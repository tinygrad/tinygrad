# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include', '-x', 'c++']
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
5838
ixFIXED_PATTERN_PERF_COUNTER_1 = 0x10c
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_10 = 0x115
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_2 = 0x10d
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_3 = 0x10e
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_4 = 0x10f
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_5 = 0x110
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_6 = 0x111
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_7 = 0x112
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_8 = 0x113
# 	PERF_COUNTER 0 16
ixFIXED_PATTERN_PERF_COUNTER_9 = 0x114
# 	PERF_COUNTER 0 16
ixGC_CAC_ACC_CHC0 = 0x61
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_CHC1 = 0x62
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_CHC2 = 0x63
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_CP0 = 0x10
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_CP1 = 0x11
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_CP2 = 0x12
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA0 = 0x13
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA1 = 0x14
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA2 = 0x15
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA3 = 0x16
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA4 = 0x17
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_EA5 = 0x18
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GDS0 = 0x2d
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GDS1 = 0x2e
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GDS2 = 0x2f
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GDS3 = 0x30
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GDS4 = 0x31
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE0 = 0x32
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE1 = 0x33
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE10 = 0x3c
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE11 = 0x3d
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE12 = 0x3e
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE13 = 0x3f
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE14 = 0x40
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE15 = 0x41
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE16 = 0x42
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE17 = 0x43
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE18 = 0x44
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE19 = 0x45
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE2 = 0x34
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE20 = 0x46
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE3 = 0x35
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE4 = 0x36
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE5 = 0x37
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE6 = 0x38
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE7 = 0x39
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE8 = 0x3a
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GE9 = 0x3b
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GL2C0 = 0x48
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GL2C1 = 0x49
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GL2C2 = 0x4a
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GL2C3 = 0x4b
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GL2C4 = 0x4c
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GUS0 = 0x64
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GUS1 = 0x65
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_GUS2 = 0x66
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH0 = 0x4d
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH1 = 0x4e
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH2 = 0x4f
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH3 = 0x50
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH4 = 0x51
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH5 = 0x52
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH6 = 0x53
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PH7 = 0x54
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_PMM0 = 0x47
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_RLC0 = 0x67
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA0 = 0x55
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA1 = 0x56
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA10 = 0x5f
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA11 = 0x60
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA2 = 0x57
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA3 = 0x58
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA4 = 0x59
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA5 = 0x5a
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA6 = 0x5b
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA7 = 0x5c
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA8 = 0x5d
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_SDMA9 = 0x5e
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER0 = 0x19
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER1 = 0x1a
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER2 = 0x1b
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER3 = 0x1c
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER4 = 0x1d
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER5 = 0x1e
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER6 = 0x1f
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER7 = 0x20
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER8 = 0x21
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_ROUTER9 = 0x22
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_VML20 = 0x23
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_VML21 = 0x24
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_VML22 = 0x25
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_VML23 = 0x26
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_VML24 = 0x27
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_WALKER0 = 0x28
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_WALKER1 = 0x29
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_WALKER2 = 0x2a
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_WALKER3 = 0x2b
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_ACC_UTCL2_WALKER4 = 0x2c
# 	ACCUMULATOR_31_0 0 31
ixGC_CAC_CNTL = 0x1
# 	CAC_THRESHOLD 0 15
ixGC_CAC_ID = 0x0
# 	CAC_BLOCK_ID 0 5
# 	CAC_SIGNAL_ID 6 13
ixHW_LUT_UPDATE_STATUS = 0x116
# 	UPDATE_TABLE_1_DONE 0 0
# 	UPDATE_TABLE_1_ERROR 1 1
# 	UPDATE_TABLE_1_ERROR_STEP 2 4
# 	UPDATE_TABLE_2_DONE 5 5
# 	UPDATE_TABLE_2_ERROR 6 6
# 	UPDATE_TABLE_2_ERROR_STEP 7 9
# 	UPDATE_TABLE_3_DONE 10 10
# 	UPDATE_TABLE_3_ERROR 11 11
# 	UPDATE_TABLE_3_ERROR_STEP 12 16
# 	UPDATE_TABLE_4_DONE 17 17
# 	UPDATE_TABLE_4_ERROR 18 18
# 	UPDATE_TABLE_4_ERROR_STEP 19 21
# 	UPDATE_TABLE_5_DONE 22 22
# 	UPDATE_TABLE_5_ERROR 23 23
# 	UPDATE_TABLE_5_ERROR_STEP 24 28
ixPWRBRK_RELEASE_TO_STALL_LUT_17_20 = 0x10b
# 	FIRST_PATTERN_17 0 2
# 	FIRST_PATTERN_18 4 6
# 	FIRST_PATTERN_19 8 10
# 	FIRST_PATTERN_20 12 14
ixPWRBRK_RELEASE_TO_STALL_LUT_1_8 = 0x109
# 	FIRST_PATTERN_1 0 2
# 	FIRST_PATTERN_2 4 6
# 	FIRST_PATTERN_3 8 10
# 	FIRST_PATTERN_4 12 14
# 	FIRST_PATTERN_5 16 18
# 	FIRST_PATTERN_6 20 22
# 	FIRST_PATTERN_7 24 26
# 	FIRST_PATTERN_8 28 30
ixPWRBRK_RELEASE_TO_STALL_LUT_9_16 = 0x10a
# 	FIRST_PATTERN_9 0 2
# 	FIRST_PATTERN_10 4 6
# 	FIRST_PATTERN_11 8 10
# 	FIRST_PATTERN_12 12 14
# 	FIRST_PATTERN_13 16 18
# 	FIRST_PATTERN_14 20 22
# 	FIRST_PATTERN_15 24 26
# 	FIRST_PATTERN_16 28 30
ixPWRBRK_STALL_TO_RELEASE_LUT_1_4 = 0x107
# 	FIRST_PATTERN_1 0 4
# 	FIRST_PATTERN_2 8 12
# 	FIRST_PATTERN_3 16 20
# 	FIRST_PATTERN_4 24 28
ixPWRBRK_STALL_TO_RELEASE_LUT_5_7 = 0x108
# 	FIRST_PATTERN_5 0 4
# 	FIRST_PATTERN_6 8 12
# 	FIRST_PATTERN_7 16 20
ixRELEASE_TO_STALL_LUT_17_20 = 0x102
# 	FIRST_PATTERN_17 0 2
# 	FIRST_PATTERN_18 4 6
# 	FIRST_PATTERN_19 8 10
# 	FIRST_PATTERN_20 12 14
ixRELEASE_TO_STALL_LUT_1_8 = 0x100
# 	FIRST_PATTERN_1 0 2
# 	FIRST_PATTERN_2 4 6
# 	FIRST_PATTERN_3 8 10
# 	FIRST_PATTERN_4 12 14
# 	FIRST_PATTERN_5 16 18
# 	FIRST_PATTERN_6 20 22
# 	FIRST_PATTERN_7 24 26
# 	FIRST_PATTERN_8 28 30
ixRELEASE_TO_STALL_LUT_9_16 = 0x101
# 	FIRST_PATTERN_9 0 2
# 	FIRST_PATTERN_10 4 6
# 	FIRST_PATTERN_11 8 10
# 	FIRST_PATTERN_12 12 14
# 	FIRST_PATTERN_13 16 18
# 	FIRST_PATTERN_14 20 22
# 	FIRST_PATTERN_15 24 26
# 	FIRST_PATTERN_16 28 30
ixRTAVFS_REG0 = 0x0
# 	RTAVFSZONE0STARTCNT 0 15
# 	RTAVFSZONE0STOPCNT 16 31
ixRTAVFS_REG1 = 0x1
# 	RTAVFSZONE1STARTCNT 0 15
# 	RTAVFSZONE1STOPCNT 16 31
ixRTAVFS_REG10 = 0xa
# 	RTAVFSZONE2EN1 0 31
ixRTAVFS_REG100 = 0x64
# 	RTAVFSCPO46_STARTCNT 0 15
# 	RTAVFSCPO46_STOPCNT 16 31
ixRTAVFS_REG101 = 0x65
# 	RTAVFSCPO47_STARTCNT 0 15
# 	RTAVFSCPO47_STOPCNT 16 31
ixRTAVFS_REG102 = 0x66
# 	RTAVFSCPO48_STARTCNT 0 15
# 	RTAVFSCPO48_STOPCNT 16 31
ixRTAVFS_REG103 = 0x67
# 	RTAVFSCPO49_STARTCNT 0 15
# 	RTAVFSCPO49_STOPCNT 16 31
ixRTAVFS_REG104 = 0x68
# 	RTAVFSCPO50_STARTCNT 0 15
# 	RTAVFSCPO50_STOPCNT 16 31
ixRTAVFS_REG105 = 0x69
# 	RTAVFSCPO51_STARTCNT 0 15
# 	RTAVFSCPO51_STOPCNT 16 31
ixRTAVFS_REG106 = 0x6a
# 	RTAVFSCPO52_STARTCNT 0 15
# 	RTAVFSCPO52_STOPCNT 16 31
ixRTAVFS_REG107 = 0x6b
# 	RTAVFSCPO53_STARTCNT 0 15
# 	RTAVFSCPO53_STOPCNT 16 31
ixRTAVFS_REG108 = 0x6c
# 	RTAVFSCPO54_STARTCNT 0 15
# 	RTAVFSCPO54_STOPCNT 16 31
ixRTAVFS_REG109 = 0x6d
# 	RTAVFSCPO55_STARTCNT 0 15
# 	RTAVFSCPO55_STOPCNT 16 31
ixRTAVFS_REG11 = 0xb
# 	RTAVFSZONE3EN0 0 31
ixRTAVFS_REG110 = 0x6e
# 	RTAVFSCPO56_STARTCNT 0 15
# 	RTAVFSCPO56_STOPCNT 16 31
ixRTAVFS_REG111 = 0x6f
# 	RTAVFSCPO57_STARTCNT 0 15
# 	RTAVFSCPO57_STOPCNT 16 31
ixRTAVFS_REG112 = 0x70
# 	RTAVFSCPO58_STARTCNT 0 15
# 	RTAVFSCPO58_STOPCNT 16 31
ixRTAVFS_REG113 = 0x71
# 	RTAVFSCPO59_STARTCNT 0 15
# 	RTAVFSCPO59_STOPCNT 16 31
ixRTAVFS_REG114 = 0x72
# 	RTAVFSCPO60_STARTCNT 0 15
# 	RTAVFSCPO60_STOPCNT 16 31
ixRTAVFS_REG115 = 0x73
# 	RTAVFSCPO61_STARTCNT 0 15
# 	RTAVFSCPO61_STOPCNT 16 31
ixRTAVFS_REG116 = 0x74
# 	RTAVFSCPO62_STARTCNT 0 15
# 	RTAVFSCPO62_STOPCNT 16 31
ixRTAVFS_REG117 = 0x75
# 	RTAVFSCPO63_STARTCNT 0 15
# 	RTAVFSCPO63_STOPCNT 16 31
ixRTAVFS_REG118 = 0x76
# 	RTAVFSCPOEN0 0 31
ixRTAVFS_REG119 = 0x77
# 	RTAVFSCPOEN1 0 31
ixRTAVFS_REG12 = 0xc
# 	RTAVFSZONE3EN1 0 31
ixRTAVFS_REG120 = 0x78
# 	RTAVFSCPOAVGDIV0 0 1
# 	RTAVFSCPOAVGDIV1 2 3
# 	RTAVFSCPOAVGDIV2 4 5
# 	RTAVFSCPOAVGDIV3 6 7
# 	RTAVFSCPOAVGDIV4 8 9
# 	RTAVFSCPOAVGDIV5 10 11
# 	RTAVFSCPOAVGDIV6 12 13
# 	RTAVFSCPOAVGDIV7 14 15
# 	RTAVFSCPOAVGDIVFINAL 16 17
# 	RESERVED 18 31
ixRTAVFS_REG121 = 0x79
# 	RTAVFSZONE0INUSE 0 0
# 	RTAVFSZONE1INUSE 1 1
# 	RTAVFSZONE2INUSE 2 2
# 	RTAVFSZONE3INUSE 3 3
# 	RTAVFSZONE4INUSE 4 4
# 	RTAVFSRESERVED 5 27
# 	RTAVFSERRORCODE 28 31
ixRTAVFS_REG122 = 0x7a
# 	RTAVFSCPO0_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG123 = 0x7b
# 	RTAVFSCPO1_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG124 = 0x7c
# 	RTAVFSCPO2_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG125 = 0x7d
# 	RTAVFSCPO3_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG126 = 0x7e
# 	RTAVFSCPO4_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG127 = 0x7f
# 	RTAVFSCPO5_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG128 = 0x80
# 	RTAVFSCPO6_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG129 = 0x81
# 	RTAVFSCPO7_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG13 = 0xd
# 	RTAVFSZONE4EN0 0 31
ixRTAVFS_REG130 = 0x82
# 	RTAVFSCPO8_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG131 = 0x83
# 	RTAVFSCPO9_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG132 = 0x84
# 	RTAVFSCPO10_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG133 = 0x85
# 	RTAVFSCPO11_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG134 = 0x86
# 	RTAVFSCPO12_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG135 = 0x87
# 	RTAVFSCPO13_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG136 = 0x88
# 	RTAVFSCPO14_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG137 = 0x89
# 	RTAVFSCPO15_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG138 = 0x8a
# 	RTAVFSCPO16_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG139 = 0x8b
# 	RTAVFSCPO17_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG14 = 0xe
# 	RTAVFSZONE4EN1 0 31
ixRTAVFS_REG140 = 0x8c
# 	RTAVFSCPO18_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG141 = 0x8d
# 	RTAVFSCPO19_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG142 = 0x8e
# 	RTAVFSCPO20_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG143 = 0x8f
# 	RTAVFSCPO21_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG144 = 0x90
# 	RTAVFSCPO22_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG145 = 0x91
# 	RTAVFSCPO23_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG146 = 0x92
# 	RTAVFSCPO24_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG147 = 0x93
# 	RTAVFSCPO25_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG148 = 0x94
# 	RTAVFSCPO26_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG149 = 0x95
# 	RTAVFSCPO27_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG15 = 0xf
# 	RTAVFSVF0FREQCOUNT 0 15
# 	RTAVFSVF0VOLTCODE 16 31
ixRTAVFS_REG150 = 0x96
# 	RTAVFSCPO28_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG151 = 0x97
# 	RTAVFSCPO29_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG152 = 0x98
# 	RTAVFSCPO30_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG153 = 0x99
# 	RTAVFSCPO31_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG154 = 0x9a
# 	RTAVFSCPO32_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG155 = 0x9b
# 	RTAVFSCPO33_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG156 = 0x9c
# 	RTAVFSCPO34_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG157 = 0x9d
# 	RTAVFSCPO35_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG158 = 0x9e
# 	RTAVFSCPO36_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG159 = 0x9f
# 	RTAVFSCPO37_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG16 = 0x10
# 	RTAVFSVF1FREQCOUNT 0 15
# 	RTAVFSVF1VOLTCODE 16 31
ixRTAVFS_REG160 = 0xa0
# 	RTAVFSCPO38_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG161 = 0xa1
# 	RTAVFSCPO39_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG162 = 0xa2
# 	RTAVFSCPO40_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG163 = 0xa3
# 	RTAVFSCPO41_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG164 = 0xa4
# 	RTAVFSCPO42_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG165 = 0xa5
# 	RTAVFSCPO43_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG166 = 0xa6
# 	RTAVFSCPO44_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG167 = 0xa7
# 	RTAVFSCPO45_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG168 = 0xa8
# 	RTAVFSCPO46_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG169 = 0xa9
# 	RTAVFSCPO47_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG17 = 0x11
# 	RTAVFSVF2FREQCOUNT 0 15
# 	RTAVFSVF2VOLTCODE 16 31
ixRTAVFS_REG170 = 0xaa
# 	RTAVFSCPO48_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG171 = 0xab
# 	RTAVFSCPO49_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG172 = 0xac
# 	RTAVFSCPO50_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG173 = 0xad
# 	RTAVFSCPO51_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG174 = 0xae
# 	RTAVFSCPO52_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG175 = 0xaf
# 	RTAVFSCPO53_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG176 = 0xb0
# 	RTAVFSCPO54_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG177 = 0xb1
# 	RTAVFSCPO55_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG178 = 0xb2
# 	RTAVFSCPO56_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG179 = 0xb3
# 	RTAVFSCPO57_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG18 = 0x12
# 	RTAVFSVF3FREQCOUNT 0 15
# 	RTAVFSVF3VOLTCODE 16 31
ixRTAVFS_REG180 = 0xb4
# 	RTAVFSCPO58_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG181 = 0xb5
# 	RTAVFSCPO59_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG182 = 0xb6
# 	RTAVFSCPO60_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG183 = 0xb7
# 	RTAVFSCPO61_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG184 = 0xb8
# 	RTAVFSCPO62_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG185 = 0xb9
# 	RTAVFSCPO63_RIPPLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG186 = 0xba
# 	RTAVFSTARGETFREQCNTOVERRIDE 0 15
# 	RTAVFSTARGETFREQCNTOVERRIDESEL 16 16
# 	RESERVED 17 31
ixRTAVFS_REG187 = 0xbb
# 	RTAVFSCURRENTFREQCNTOVERRIDE 0 15
# 	RTAVFSCURRENTFREQCNTOVERRIDESEL 16 16
# 	RESERVED 17 31
ixRTAVFS_REG188 = 0xbc
# 	RESERVED 22 31
ixRTAVFS_REG189 = 0xbd
# 	RTAVFSVOLTCODEFROMPI 0 9
# 	RTAVFSVOLTCODEFROMBINARYSEARCH 10 19
# 	RTAVFSVDDREGON 20 20
# 	RTAVFSVDDABOVEVDDRET 21 21
# 	RESERVED 22 31
ixRTAVFS_REG19 = 0x13
# 	RTAVFSGB_ZONE0 0 5
# 	RTAVFSGB_ZONE1 6 11
# 	RTAVFSGB_ZONE2 12 17
# 	RTAVFSGB_ZONE3 18 24
# 	RTAVFSGB_ZONE4 25 31
ixRTAVFS_REG190 = 0xbe
# 	RTAVFSIGNORERLCREQ 0 0
# 	RTAVFSRIPPLECOUNTEROUTSEL 1 5
# 	RTAVFSRUNLOOP 6 6
# 	RTAVFSSAVECPOWEIGHTS 7 7
# 	RTAVFSRESTORECPOWEIGHTS 8 8
# 	RTAVFSRESETRETENTIONREGS 9 9
# 	RESERVED 10 31
ixRTAVFS_REG191 = 0xbf
# 	RTAVFSSTOPATSTARTUP 0 0
# 	RTAVFSSTOPATIDLE 1 1
# 	RTAVFSSTOPATRESETCPORIPPLECOUNTERS 2 2
# 	RTAVFSSTOPATSTARTCPOS 3 3
# 	RTAVFSSTOPATSTARTRIPPLECOUNTERS 4 4
# 	RTAVFSSTOPATRIPPLECOUNTERSDONE 5 5
# 	RTAVFSSTOPATCPOFINALRESULTREADY 6 6
# 	RTAVFSSTOPATVOLTCODEREADY 7 7
# 	RTAVFSSTOPATTARGETVOLATGEREADY 8 8
# 	RTAVFSSTOPATSTOPCPOS 9 9
# 	RTAVFSSTOPATWAITFORACK 10 10
# 	RESERVED 11 31
ixRTAVFS_REG192 = 0xc0
# 	RTAVFSAVFSSCALEDCPOCOUNT 0 15
# 	RTAVFSAVFSFINALMINCPOCOUNT 16 31
ixRTAVFS_REG193 = 0xc1
# 	RTAVFSFSMSTATE 0 15
# 	RESERVED 16 31
ixRTAVFS_REG194 = 0xc2
# 	RTAVFSRIPPLECNTREAD 0 31
ixRTAVFS_REG2 = 0x2
# 	RTAVFSZONE2STARTCNT 0 15
# 	RTAVFSZONE2STOPCNT 16 31
ixRTAVFS_REG20 = 0x14
# 	RTAVFSZONE0CPOAVGDIV0 0 1
# 	RTAVFSZONE0CPOAVGDIV1 2 3
# 	RTAVFSZONE0CPOAVGDIV2 4 5
# 	RTAVFSZONE0CPOAVGDIV3 6 7
# 	RTAVFSZONE0CPOAVGDIV4 8 9
# 	RTAVFSZONE0CPOAVGDIV5 10 11
# 	RTAVFSZONE0CPOAVGDIV6 12 13
# 	RTAVFSZONE0CPOAVGDIV7 14 15
# 	RTAVFSZONE0CPOAVGDIVFINAL 16 17
# 	RTAVFSZONE0RESERVED 18 31
ixRTAVFS_REG21 = 0x15
# 	RTAVFSZONE1CPOAVGDIV0 0 1
# 	RTAVFSZONE1CPOAVGDIV1 2 3
# 	RTAVFSZONE1CPOAVGDIV2 4 5
# 	RTAVFSZONE1CPOAVGDIV3 6 7
# 	RTAVFSZONE1CPOAVGDIV4 8 9
# 	RTAVFSZONE1CPOAVGDIV5 10 11
# 	RTAVFSZONE1CPOAVGDIV6 12 13
# 	RTAVFSZONE1CPOAVGDIV7 14 15
# 	RTAVFSZONE1CPOAVGDIVFINAL 16 17
# 	RTAVFSZONE1RESERVED 18 31
ixRTAVFS_REG22 = 0x16
# 	RTAVFSZONE2CPOAVGDIV0 0 1
# 	RTAVFSZONE2CPOAVGDIV1 2 3
# 	RTAVFSZONE2CPOAVGDIV2 4 5
# 	RTAVFSZONE2CPOAVGDIV3 6 7
# 	RTAVFSZONE2CPOAVGDIV4 8 9
# 	RTAVFSZONE2CPOAVGDIV5 10 11
# 	RTAVFSZONE2CPOAVGDIV6 12 13
# 	RTAVFSZONE2CPOAVGDIV7 14 15
# 	RTAVFSZONE2CPOAVGDIVFINAL 16 17
# 	RTAVFSZONE2RESERVED 18 31
ixRTAVFS_REG23 = 0x17
# 	RTAVFSZONE3CPOAVGDIV0 0 1
# 	RTAVFSZONE3CPOAVGDIV1 2 3
# 	RTAVFSZONE3CPOAVGDIV2 4 5
# 	RTAVFSZONE3CPOAVGDIV3 6 7
# 	RTAVFSZONE3CPOAVGDIV4 8 9
# 	RTAVFSZONE3CPOAVGDIV5 10 11
# 	RTAVFSZONE3CPOAVGDIV6 12 13
# 	RTAVFSZONE3CPOAVGDIV7 14 15
# 	RTAVFSZONE3CPOAVGDIVFINAL 16 17
# 	RTAVFSZONE3RESERVED 18 31
ixRTAVFS_REG24 = 0x18
# 	RTAVFSZONE4CPOAVGDIV0 0 1
# 	RTAVFSZONE4CPOAVGDIV1 2 3
# 	RTAVFSZONE4CPOAVGDIV2 4 5
# 	RTAVFSZONE4CPOAVGDIV3 6 7
# 	RTAVFSZONE4CPOAVGDIV4 8 9
# 	RTAVFSZONE4CPOAVGDIV5 10 11
# 	RTAVFSZONE4CPOAVGDIV6 12 13
# 	RTAVFSZONE4CPOAVGDIV7 14 15
# 	RTAVFSZONE4CPOAVGDIVFINAL 16 17
# 	RTAVFSZONE4RESERVED 18 31
ixRTAVFS_REG25 = 0x19
# 	RTAVFSRESERVED0 0 31
ixRTAVFS_REG26 = 0x1a
# 	RTAVFSRESERVED1 0 31
ixRTAVFS_REG27 = 0x1b
# 	RTAVFSRESERVED2 0 31
ixRTAVFS_REG28 = 0x1c
# 	RTAVFSZONE0INTERCEPT 0 15
# 	RTAVFSZONE1INTERCEPT 16 31
ixRTAVFS_REG29 = 0x1d
# 	RTAVFSZONE2INTERCEPT 0 15
# 	RTAVFSZONE3INTERCEPT 16 31
ixRTAVFS_REG3 = 0x3
# 	RTAVFSZONE3STARTCNT 0 15
# 	RTAVFSZONE3STOPCNT 16 31
ixRTAVFS_REG30 = 0x1e
# 	RTAVFSZONE4INTERCEPT 0 15
# 	RTAVFSRESERVEDINTERCEPT 16 31
ixRTAVFS_REG31 = 0x1f
# 	RTAVFSCPOCLKDIV0 0 1
# 	RTAVFSCPOCLKDIV1 2 3
# 	RTAVFSCPOCLKDIV2 4 5
# 	RTAVFSCPOCLKDIV3 6 7
# 	RTAVFSCPOCLKDIV4 8 9
# 	RTAVFSCPOCLKDIV5 10 11
# 	RTAVFSCPOCLKDIV6 12 13
# 	RTAVFSCPOCLKDIV7 14 15
# 	RESERVED 16 31
ixRTAVFS_REG32 = 0x20
# 	RTAVFSFSMSTARTUPCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG33 = 0x21
# 	RTAVFSFSMIDLECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG34 = 0x22
# 	RTAVFSFSMRESETCPORIPPLECOUNTERSCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG35 = 0x23
# 	RTAVFSFSMSTARTCPOSCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG36 = 0x24
# 	RTAVFSFSMSTARTRIPPLECOUNTERSCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG37 = 0x25
# 	RTAVFSFSMRIPPLECOUNTERSDONECNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG38 = 0x26
# 	RTAVFSFSMCPOFINALRESULTREADYCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG39 = 0x27
# 	RTAVFSFSMVOLTCODEREADYCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG4 = 0x4
# 	RTAVFSZONE4STARTCNT 0 15
# 	RTAVFSZONE4STOPCNT 16 31
ixRTAVFS_REG40 = 0x28
# 	RTAVFSFSMTARGETVOLTAGEREADYCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG41 = 0x29
# 	RTAVFSFSMSTOPCPOSCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG42 = 0x2a
# 	RTAVFSFSMWAITFORACKCNT 0 15
# 	RESERVED 16 31
ixRTAVFS_REG43 = 0x2b
# 	RTAVFSKP0 0 3
# 	RTAVFSKP1 4 7
# 	RTAVFSKP2 8 11
# 	RTAVFSKP3 12 15
# 	RTAVFSKI0 16 19
# 	RTAVFSKI1 20 23
# 	RTAVFSKI2 24 27
# 	RTAVFSKI3 28 31
ixRTAVFS_REG44 = 0x2c
# 	RTAVFSV1 0 9
# 	RTAVFSV2 10 19
# 	RTAVFSV3 20 29
# 	RTAVFSUSEBINARYSEARCH 30 30
# 	RTAVFSVOLTCODEHWCAL 31 31
ixRTAVFS_REG45 = 0x2d
# 	RTAVFSVRBLEEDCNTRL 0 0
# 	RTAVFSVRENABLE 1 1
# 	RTAVFSVOLTCODEOVERRIDE 2 11
# 	RTAVFSVOLTCODEOVERRIDESEL 12 12
# 	RTAVFSLOWPWREN 13 13
# 	RTAVFSUREGENABLE 14 14
# 	RTAVFSBGENABLE 15 15
# 	RTAVFSENABLEVDDRETSENSING 16 16
# 	RESERVED 17 31
ixRTAVFS_REG46 = 0x2e
# 	RTAVFSKP 0 3
# 	RTAVFSKI 4 7
# 	RTAVFSPIENABLEANTIWINDUP 8 8
# 	RTAVFSPISHIFT 9 12
# 	RTAVFSPIERREN 13 13
# 	RTAVFSPISHIFTOUT 14 17
# 	RTAVFSUSELUTKPKI 18 18
# 	RESERVED 19 31
ixRTAVFS_REG47 = 0x2f
# 	RTAVFSVOLTCODEPIMIN 0 9
# 	RTAVFSVOLTCODEPIMAX 10 19
# 	RTAVFSPIERRMASK 20 26
# 	RTAVFSFORCEDISABLEPI 27 27
# 	RESERVED 28 31
ixRTAVFS_REG48 = 0x30
# 	RTAVFSPILOOPNITERATIONS 0 15
# 	RTAVFSPIERRTHRESHOLD 16 31
ixRTAVFS_REG49 = 0x31
# 	RTAVFSPSMRSTAVGVDD 0 0
# 	RTAVFSPSMMEASMAXVDD 1 1
# 	RTAVFSPSMCLKDIVVDD 2 3
# 	RTAVFSPSMAVGDIVVDD 4 9
# 	RTAVFSPSMOSCENVDD 10 10
# 	RTAVFSPSMAVGENVDD 11 11
# 	RTAVFSPSMRSTMINMAXVDD 12 12
# 	RESERVED 13 31
ixRTAVFS_REG5 = 0x5
# 	RTAVFSZONE0EN0 0 31
ixRTAVFS_REG50 = 0x32
# 	RTAVFSPSMRSTAVGVREG 0 0
# 	RTAVFSPSMMEASMAXVREG 1 1
# 	RTAVFSPSMCLKDIVVREG 2 3
# 	RTAVFSPSMAVGDIVVREG 4 9
# 	RTAVFSPSMOSCENVREG 10 10
# 	RTAVFSPSMAVGENVREG 11 11
# 	RTAVFSPSMRSTMINMAXVREG 12 12
# 	RESERVED 13 31
ixRTAVFS_REG51 = 0x33
# 	RTAVFSAVFSENABLE 0 0
# 	RTAVFSCPOTURNONDELAY 1 4
# 	RTAVFSSELECTMINMAX 5 5
# 	RTAVFSSELECTPERPATHSCALING 6 6
# 	RTAVFSADDVOLTCODEGUARDBAND 7 7
# 	RTAVFSSENDAVGPSMTOPSMOUT 8 8
# 	RTAVFSUPDATEANCHORVOLTAGES 9 9
# 	RTAVFSSENDVDDTOPSMOUT 10 10
# 	RESERVED 11 31
ixRTAVFS_REG52 = 0x34
# 	RTAVFSMINMAXPSMVDD 0 13
# 	RTAVFSAVGPSMVDD 14 27
# 	RESERVED 28 31
ixRTAVFS_REG53 = 0x35
# 	RTAVFSMINMAXPSMVREG 0 13
# 	RTAVFSAVGPSMVREG 14 27
# 	RESERVED 28 31
ixRTAVFS_REG54 = 0x36
# 	RTAVFSCPO0_STARTCNT 0 15
# 	RTAVFSCPO0_STOPCNT 16 31
ixRTAVFS_REG55 = 0x37
# 	RTAVFSCPO1_STARTCNT 0 15
# 	RTAVFSCPO1_STOPCNT 16 31
ixRTAVFS_REG56 = 0x38
# 	RTAVFSCPO2_STARTCNT 0 15
# 	RTAVFSCPO2_STOPCNT 16 31
ixRTAVFS_REG57 = 0x39
# 	RTAVFSCPO3_STARTCNT 0 15
# 	RTAVFSCPO3_STOPCNT 16 31
ixRTAVFS_REG58 = 0x3a
# 	RTAVFSCPO4_STARTCNT 0 15
# 	RTAVFSCPO4_STOPCNT 16 31
ixRTAVFS_REG59 = 0x3b
# 	RTAVFSCPO5_STARTCNT 0 15
# 	RTAVFSCPO5_STOPCNT 16 31
ixRTAVFS_REG6 = 0x6
# 	RTAVFSZONE0EN1 0 31
ixRTAVFS_REG60 = 0x3c
# 	RTAVFSCPO6_STARTCNT 0 15
# 	RTAVFSCPO6_STOPCNT 16 31
ixRTAVFS_REG61 = 0x3d
# 	RTAVFSCPO7_STARTCNT 0 15
# 	RTAVFSCPO7_STOPCNT 16 31
ixRTAVFS_REG62 = 0x3e
# 	RTAVFSCPO8_STARTCNT 0 15
# 	RTAVFSCPO8_STOPCNT 16 31
ixRTAVFS_REG63 = 0x3f
# 	RTAVFSCPO9_STARTCNT 0 15
# 	RTAVFSCPO9_STOPCNT 16 31
ixRTAVFS_REG64 = 0x40
# 	RTAVFSCPO10_STARTCNT 0 15
# 	RTAVFSCPO10_STOPCNT 16 31
ixRTAVFS_REG65 = 0x41
# 	RTAVFSCPO11_STARTCNT 0 15
# 	RTAVFSCPO11_STOPCNT 16 31
ixRTAVFS_REG66 = 0x42
# 	RTAVFSCPO12_STARTCNT 0 15
# 	RTAVFSCPO12_STOPCNT 16 31
ixRTAVFS_REG67 = 0x43
# 	RTAVFSCPO13_STARTCNT 0 15
# 	RTAVFSCPO13_STOPCNT 16 31
ixRTAVFS_REG68 = 0x44
# 	RTAVFSCPO14_STARTCNT 0 15
# 	RTAVFSCPO14_STOPCNT 16 31
ixRTAVFS_REG69 = 0x45
# 	RTAVFSCPO15_STARTCNT 0 15
# 	RTAVFSCPO15_STOPCNT 16 31
ixRTAVFS_REG7 = 0x7
# 	RTAVFSZONE1EN0 0 31
ixRTAVFS_REG70 = 0x46
# 	RTAVFSCPO16_STARTCNT 0 15
# 	RTAVFSCPO16_STOPCNT 16 31
ixRTAVFS_REG71 = 0x47
# 	RTAVFSCPO17_STARTCNT 0 15
# 	RTAVFSCPO17_STOPCNT 16 31
ixRTAVFS_REG72 = 0x48
# 	RTAVFSCPO18_STARTCNT 0 15
# 	RTAVFSCPO18_STOPCNT 16 31
ixRTAVFS_REG73 = 0x49
# 	RTAVFSCPO19_STARTCNT 0 15
# 	RTAVFSCPO19_STOPCNT 16 31
ixRTAVFS_REG74 = 0x4a
# 	RTAVFSCPO20_STARTCNT 0 15
# 	RTAVFSCPO20_STOPCNT 16 31
ixRTAVFS_REG75 = 0x4b
# 	RTAVFSCPO21_STARTCNT 0 15
# 	RTAVFSCPO21_STOPCNT 16 31
ixRTAVFS_REG76 = 0x4c
# 	RTAVFSCPO22_STARTCNT 0 15
# 	RTAVFSCPO22_STOPCNT 16 31
ixRTAVFS_REG77 = 0x4d
# 	RTAVFSCPO23_STARTCNT 0 15
# 	RTAVFSCPO23_STOPCNT 16 31
ixRTAVFS_REG78 = 0x4e
# 	RTAVFSCPO24_STARTCNT 0 15
# 	RTAVFSCPO24_STOPCNT 16 31
ixRTAVFS_REG79 = 0x4f
# 	RTAVFSCPO25_STARTCNT 0 15
# 	RTAVFSCPO25_STOPCNT 16 31
ixRTAVFS_REG8 = 0x8
# 	RTAVFSZONE1EN1 0 31
ixRTAVFS_REG80 = 0x50
# 	RTAVFSCPO26_STARTCNT 0 15
# 	RTAVFSCPO26_STOPCNT 16 31
ixRTAVFS_REG81 = 0x51
# 	RTAVFSCPO27_STARTCNT 0 15
# 	RTAVFSCPO27_STOPCNT 16 31
ixRTAVFS_REG82 = 0x52
# 	RTAVFSCPO28_STARTCNT 0 15
# 	RTAVFSCPO28_STOPCNT 16 31
ixRTAVFS_REG83 = 0x53
# 	RTAVFSCPO29_STARTCNT 0 15
# 	RTAVFSCPO29_STOPCNT 16 31
ixRTAVFS_REG84 = 0x54
# 	RTAVFSCPO30_STARTCNT 0 15
# 	RTAVFSCPO30_STOPCNT 16 31
ixRTAVFS_REG85 = 0x55
# 	RTAVFSCPO31_STARTCNT 0 15
# 	RTAVFSCPO31_STOPCNT 16 31
ixRTAVFS_REG86 = 0x56
# 	RTAVFSCPO32_STARTCNT 0 15
# 	RTAVFSCPO32_STOPCNT 16 31
ixRTAVFS_REG87 = 0x57
# 	RTAVFSCPO33_STARTCNT 0 15
# 	RTAVFSCPO33_STOPCNT 16 31
ixRTAVFS_REG88 = 0x58
# 	RTAVFSCPO34_STARTCNT 0 15
# 	RTAVFSCPO34_STOPCNT 16 31
ixRTAVFS_REG89 = 0x59
# 	RTAVFSCPO35_STARTCNT 0 15
# 	RTAVFSCPO35_STOPCNT 16 31
ixRTAVFS_REG9 = 0x9
# 	RTAVFSZONE2EN0 0 31
ixRTAVFS_REG90 = 0x5a
# 	RTAVFSCPO36_STARTCNT 0 15
# 	RTAVFSCPO36_STOPCNT 16 31
ixRTAVFS_REG91 = 0x5b
# 	RTAVFSCPO37_STARTCNT 0 15
# 	RTAVFSCPO37_STOPCNT 16 31
ixRTAVFS_REG92 = 0x5c
# 	RTAVFSCPO38_STARTCNT 0 15
# 	RTAVFSCPO38_STOPCNT 16 31
ixRTAVFS_REG93 = 0x5d
# 	RTAVFSCPO39_STARTCNT 0 15
# 	RTAVFSCPO39_STOPCNT 16 31
ixRTAVFS_REG94 = 0x5e
# 	RTAVFSCPO40_STARTCNT 0 15
# 	RTAVFSCPO40_STOPCNT 16 31
ixRTAVFS_REG95 = 0x5f
# 	RTAVFSCPO41_STARTCNT 0 15
# 	RTAVFSCPO41_STOPCNT 16 31
ixRTAVFS_REG96 = 0x60
# 	RTAVFSCPO42_STARTCNT 0 15
# 	RTAVFSCPO42_STOPCNT 16 31
ixRTAVFS_REG97 = 0x61
# 	RTAVFSCPO43_STARTCNT 0 15
# 	RTAVFSCPO43_STOPCNT 16 31
ixRTAVFS_REG98 = 0x62
# 	RTAVFSCPO44_STARTCNT 0 15
# 	RTAVFSCPO44_STOPCNT 16 31
ixRTAVFS_REG99 = 0x63
# 	RTAVFSCPO45_STARTCNT 0 15
# 	RTAVFSCPO45_STOPCNT 16 31
ixSE_CAC_CNTL = 0x1
# 	CAC_THRESHOLD 0 15
ixSE_CAC_ID = 0x0
# 	CAC_BLOCK_ID 0 5
# 	CAC_SIGNAL_ID 6 13
ixSQ_DEBUG_CTRL_LOCAL = 0x9
# 	UNUSED 0 7
ixSQ_DEBUG_STS_LOCAL = 0x8
# 	BUSY 0 0
# 	WAVE_LEVEL 4 9
# 	SQ_BUSY 12 12
# 	IS_BUSY 13 13
# 	IB_BUSY 14 14
# 	ARB_BUSY 15 15
# 	EXP_BUSY 16 16
# 	BRMSG_BUSY 17 17
# 	VM_BUSY 18 18
ixSQ_WAVE_ACTIVE = 0xa
# 	WAVE_SLOT 0 19
ixSQ_WAVE_EXEC_HI = 0x27f
# 	EXEC_HI 0 31
ixSQ_WAVE_EXEC_LO = 0x27e
# 	EXEC_LO 0 31
ixSQ_WAVE_FLAT_SCRATCH_HI = 0x115
# 	DATA 0 31
ixSQ_WAVE_FLAT_SCRATCH_LO = 0x114
# 	DATA 0 31
ixSQ_WAVE_FLUSH_IB = 0x10e
# 	UNUSED 0 31
ixSQ_WAVE_GPR_ALLOC = 0x105
# 	VGPR_BASE 0 8
# 	VGPR_SIZE 12 19
ixSQ_WAVE_HW_ID1 = 0x117
# 	WAVE_ID 0 4
# 	SIMD_ID 8 9
# 	WGP_ID 10 13
# 	SA_ID 16 16
# 	SE_ID 18 20
# 	DP_RATE 29 31
ixSQ_WAVE_HW_ID2 = 0x118
# 	QUEUE_ID 0 3
# 	PIPE_ID 4 5
# 	ME_ID 8 9
# 	STATE_ID 12 14
# 	WG_ID 16 20
# 	VM_ID 24 27
ixSQ_WAVE_IB_DBG1 = 0x10d
# 	WAVE_IDLE 24 24
# 	MISC_CNT 25 31
ixSQ_WAVE_IB_STS = 0x107
# 	EXP_CNT 0 2
# 	LGKM_CNT 4 9
# 	VM_CNT 10 15
# 	VS_CNT 26 31
ixSQ_WAVE_IB_STS2 = 0x11c
# 	INST_PREFETCH 0 1
# 	MEM_ORDER 8 9
# 	FWD_PROGRESS 10 10
# 	WAVE64 11 11
ixSQ_WAVE_LDS_ALLOC = 0x106
# 	LDS_BASE 0 8
# 	LDS_SIZE 12 20
# 	VGPR_SHARED_SIZE 24 27
ixSQ_WAVE_M0 = 0x27d
# 	M0 0 31
ixSQ_WAVE_MODE = 0x101
# 	FP_ROUND 0 3
# 	FP_DENORM 4 7
# 	DX10_CLAMP 8 8
# 	IEEE 9 9
# 	LOD_CLAMPED 10 10
# 	TRAP_AFTER_INST_EN 11 11
# 	EXCP_EN 12 20
# 	WAVE_END 21 21
# 	FP16_OVFL 23 23
# 	DISABLE_PERF 27 27
ixSQ_WAVE_PC_HI = 0x109
# 	PC_HI 0 15
ixSQ_WAVE_PC_LO = 0x108
# 	PC_LO 0 31
ixSQ_WAVE_POPS_PACKER = 0x119
# 	POPS_EN 0 0
# 	POPS_PACKER_ID 1 2
ixSQ_WAVE_SCHED_MODE = 0x11a
# 	DEP_MODE 0 1
ixSQ_WAVE_SHADER_CYCLES = 0x11d
# 	CYCLES 0 19
ixSQ_WAVE_STATUS = 0x102
# 	SCC 0 0
# 	SPI_PRIO 1 2
# 	USER_PRIO 3 4
# 	PRIV 5 5
# 	TRAP_EN 6 6
# 	TTRACE_EN 7 7
# 	EXPORT_RDY 8 8
# 	EXECZ 9 9
# 	VCCZ 10 10
# 	IN_TG 11 11
# 	IN_BARRIER 12 12
# 	HALT 13 13
# 	TRAP 14 14
# 	TTRACE_SIMD_EN 15 15
# 	VALID 16 16
# 	ECC_ERR 17 17
# 	SKIP_EXPORT 18 18
# 	PERF_EN 19 19
# 	OREO_CONFLICT 22 22
# 	FATAL_HALT 23 23
# 	NO_VGPRS 24 24
# 	LDS_PARAM_READY 25 25
# 	MUST_GS_ALLOC 26 26
# 	MUST_EXPORT 27 27
# 	IDLE 28 28
# 	SCRATCH_EN 29 29
ixSQ_WAVE_TRAPSTS = 0x103
# 	EXCP 0 8
# 	SAVECTX 10 10
# 	ILLEGAL_INST 11 11
# 	EXCP_HI 12 14
# 	BUFFER_OOB 15 15
# 	HOST_TRAP 16 16
# 	WAVESTART 17 17
# 	WAVE_END 18 18
# 	PERF_SNAPSHOT 19 19
# 	TRAP_AFTER_INST 20 20
# 	UTC_ERROR 28 28
ixSQ_WAVE_TTMP0 = 0x26c
# 	DATA 0 31
ixSQ_WAVE_TTMP1 = 0x26d
# 	DATA 0 31
ixSQ_WAVE_TTMP10 = 0x276
# 	DATA 0 31
ixSQ_WAVE_TTMP11 = 0x277
# 	DATA 0 31
ixSQ_WAVE_TTMP12 = 0x278
# 	DATA 0 31
ixSQ_WAVE_TTMP13 = 0x279
# 	DATA 0 31
ixSQ_WAVE_TTMP14 = 0x27a
# 	DATA 0 31
ixSQ_WAVE_TTMP15 = 0x27b
# 	DATA 0 31
ixSQ_WAVE_TTMP3 = 0x26f
# 	DATA 0 31
ixSQ_WAVE_TTMP4 = 0x270
# 	DATA 0 31
ixSQ_WAVE_TTMP5 = 0x271
# 	DATA 0 31
ixSQ_WAVE_TTMP6 = 0x272
# 	DATA 0 31
ixSQ_WAVE_TTMP7 = 0x273
# 	DATA 0 31
ixSQ_WAVE_TTMP8 = 0x274
# 	DATA 0 31
ixSQ_WAVE_TTMP9 = 0x275
# 	DATA 0 31
ixSQ_WAVE_VALID_AND_IDLE = 0xb
# 	WAVE_SLOT 0 19
ixSTALL_TO_PWRBRK_LUT_1_4 = 0x105
# 	FIRST_PATTERN_1 0 2
# 	FIRST_PATTERN_2 8 10
# 	FIRST_PATTERN_3 16 18
# 	FIRST_PATTERN_4 24 26
ixSTALL_TO_PWRBRK_LUT_5_7 = 0x106
# 	FIRST_PATTERN_5 0 2
# 	FIRST_PATTERN_6 8 10
# 	FIRST_PATTERN_7 16 18
ixSTALL_TO_RELEASE_LUT_1_4 = 0x103
# 	FIRST_PATTERN_1 0 4
# 	FIRST_PATTERN_2 8 12
# 	FIRST_PATTERN_3 16 20
# 	FIRST_PATTERN_4 24 28
ixSTALL_TO_RELEASE_LUT_5_7 = 0x104
# 	FIRST_PATTERN_5 0 4
# 	FIRST_PATTERN_6 8 12
# 	FIRST_PATTERN_7 16 20
regCB_BLEND0_CONTROL = 0x1e0
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND1_CONTROL = 0x1e1
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND2_CONTROL = 0x1e2
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND3_CONTROL = 0x1e3
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND4_CONTROL = 0x1e4
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND5_CONTROL = 0x1e5
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND6_CONTROL = 0x1e6
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND7_CONTROL = 0x1e7
# 	COLOR_SRCBLEND 0 4
# 	COLOR_COMB_FCN 5 7
# 	COLOR_DESTBLEND 8 12
# 	ALPHA_SRCBLEND 16 20
# 	ALPHA_COMB_FCN 21 23
# 	ALPHA_DESTBLEND 24 28
# 	SEPARATE_ALPHA_BLEND 29 29
# 	ENABLE 30 30
# 	DISABLE_ROP3 31 31
regCB_BLEND_ALPHA = 0x108
# 	BLEND_ALPHA 0 31
regCB_BLEND_BLUE = 0x107
# 	BLEND_BLUE 0 31
regCB_BLEND_GREEN = 0x106
# 	BLEND_GREEN 0 31
regCB_BLEND_RED = 0x105
# 	BLEND_RED 0 31
regCB_CACHE_EVICT_POINTS = 0x142e
# 	CC_COLOR_EVICT_POINT 0 7
# 	CC_FMASK_EVICT_POINT 8 15
# 	DCC_CACHE_EVICT_POINT 16 23
# 	CC_CACHE_EVICT_POINT 24 31
regCB_CGTT_SCLK_CTRL = 0x50a8
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	SOFT_OVERRIDE1 30 30
# 	SOFT_OVERRIDE0 31 31
regCB_COLOR0_ATTRIB = 0x31d
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR0_ATTRIB2 = 0x3b0
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR0_ATTRIB3 = 0x3b8
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR0_BASE = 0x318
# 	BASE_256B 0 31
regCB_COLOR0_BASE_EXT = 0x390
# 	BASE_256B 0 7
regCB_COLOR0_DCC_BASE = 0x325
# 	BASE_256B 0 31
regCB_COLOR0_DCC_BASE_EXT = 0x3a8
# 	BASE_256B 0 7
regCB_COLOR0_FDCC_CONTROL = 0x31e
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR0_INFO = 0x31c
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR0_VIEW = 0x31b
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR1_ATTRIB = 0x32c
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR1_ATTRIB2 = 0x3b1
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR1_ATTRIB3 = 0x3b9
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR1_BASE = 0x327
# 	BASE_256B 0 31
regCB_COLOR1_BASE_EXT = 0x391
# 	BASE_256B 0 7
regCB_COLOR1_DCC_BASE = 0x334
# 	BASE_256B 0 31
regCB_COLOR1_DCC_BASE_EXT = 0x3a9
# 	BASE_256B 0 7
regCB_COLOR1_FDCC_CONTROL = 0x32d
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR1_INFO = 0x32b
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR1_VIEW = 0x32a
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR2_ATTRIB = 0x33b
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR2_ATTRIB2 = 0x3b2
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR2_ATTRIB3 = 0x3ba
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR2_BASE = 0x336
# 	BASE_256B 0 31
regCB_COLOR2_BASE_EXT = 0x392
# 	BASE_256B 0 7
regCB_COLOR2_DCC_BASE = 0x343
# 	BASE_256B 0 31
regCB_COLOR2_DCC_BASE_EXT = 0x3aa
# 	BASE_256B 0 7
regCB_COLOR2_FDCC_CONTROL = 0x33c
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR2_INFO = 0x33a
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR2_VIEW = 0x339
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR3_ATTRIB = 0x34a
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR3_ATTRIB2 = 0x3b3
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR3_ATTRIB3 = 0x3bb
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR3_BASE = 0x345
# 	BASE_256B 0 31
regCB_COLOR3_BASE_EXT = 0x393
# 	BASE_256B 0 7
regCB_COLOR3_DCC_BASE = 0x352
# 	BASE_256B 0 31
regCB_COLOR3_DCC_BASE_EXT = 0x3ab
# 	BASE_256B 0 7
regCB_COLOR3_FDCC_CONTROL = 0x34b
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR3_INFO = 0x349
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR3_VIEW = 0x348
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR4_ATTRIB = 0x359
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR4_ATTRIB2 = 0x3b4
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR4_ATTRIB3 = 0x3bc
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR4_BASE = 0x354
# 	BASE_256B 0 31
regCB_COLOR4_BASE_EXT = 0x394
# 	BASE_256B 0 7
regCB_COLOR4_DCC_BASE = 0x361
# 	BASE_256B 0 31
regCB_COLOR4_DCC_BASE_EXT = 0x3ac
# 	BASE_256B 0 7
regCB_COLOR4_FDCC_CONTROL = 0x35a
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR4_INFO = 0x358
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR4_VIEW = 0x357
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR5_ATTRIB = 0x368
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR5_ATTRIB2 = 0x3b5
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR5_ATTRIB3 = 0x3bd
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR5_BASE = 0x363
# 	BASE_256B 0 31
regCB_COLOR5_BASE_EXT = 0x395
# 	BASE_256B 0 7
regCB_COLOR5_DCC_BASE = 0x370
# 	BASE_256B 0 31
regCB_COLOR5_DCC_BASE_EXT = 0x3ad
# 	BASE_256B 0 7
regCB_COLOR5_FDCC_CONTROL = 0x369
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR5_INFO = 0x367
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR5_VIEW = 0x366
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR6_ATTRIB = 0x377
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR6_ATTRIB2 = 0x3b6
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR6_ATTRIB3 = 0x3be
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR6_BASE = 0x372
# 	BASE_256B 0 31
regCB_COLOR6_BASE_EXT = 0x396
# 	BASE_256B 0 7
regCB_COLOR6_DCC_BASE = 0x37f
# 	BASE_256B 0 31
regCB_COLOR6_DCC_BASE_EXT = 0x3ae
# 	BASE_256B 0 7
regCB_COLOR6_FDCC_CONTROL = 0x378
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR6_INFO = 0x376
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR6_VIEW = 0x375
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR7_ATTRIB = 0x386
# 	NUM_FRAGMENTS 0 1
# 	FORCE_DST_ALPHA_1 2 2
# 	DISABLE_FMASK_NOALLOC_OPT 3 3
# 	LIMIT_COLOR_FETCH_TO_256B_MAX 4 4
# 	FORCE_LIMIT_COLOR_SECTOR_TO_256B_MAX 5 5
regCB_COLOR7_ATTRIB2 = 0x3b7
# 	MIP0_HEIGHT 0 13
# 	MIP0_WIDTH 14 27
# 	MAX_MIP 28 31
regCB_COLOR7_ATTRIB3 = 0x3bf
# 	MIP0_DEPTH 0 12
# 	META_LINEAR 13 13
# 	COLOR_SW_MODE 14 18
# 	RESOURCE_TYPE 24 25
# 	DCC_PIPE_ALIGNED 30 30
regCB_COLOR7_BASE = 0x381
# 	BASE_256B 0 31
regCB_COLOR7_BASE_EXT = 0x397
# 	BASE_256B 0 7
regCB_COLOR7_DCC_BASE = 0x38e
# 	BASE_256B 0 31
regCB_COLOR7_DCC_BASE_EXT = 0x3af
# 	BASE_256B 0 7
regCB_COLOR7_FDCC_CONTROL = 0x387
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_FEA_FORCE 1 1
# 	MAX_UNCOMPRESSED_BLOCK_SIZE 2 3
# 	MIN_COMPRESSED_BLOCK_SIZE 4 4
# 	MAX_COMPRESSED_BLOCK_SIZE 5 6
# 	COLOR_TRANSFORM 7 8
# 	INDEPENDENT_64B_BLOCKS 9 9
# 	INDEPENDENT_128B_BLOCKS 10 10
# 	DISABLE_CONSTANT_ENCODE_REG 18 18
# 	ENABLE_CONSTANT_ENCODE_REG_WRITE 19 19
# 	SKIP_LOW_COMP_RATIO 21 21
# 	FDCC_ENABLE 22 22
# 	DCC_COMPRESS_DISABLE 23 23
# 	FRAGMENT_COMPRESS_DISABLE 24 24
regCB_COLOR7_INFO = 0x385
# 	FORMAT 0 4
# 	LINEAR_GENERAL 7 7
# 	NUMBER_TYPE 8 10
# 	COMP_SWAP 11 12
# 	BLEND_CLAMP 15 15
# 	BLEND_BYPASS 16 16
# 	SIMPLE_FLOAT 17 17
# 	ROUND_MODE 18 18
# 	BLEND_OPT_DONT_RD_DST 20 22
# 	BLEND_OPT_DISCARD_PIXEL 23 25
regCB_COLOR7_VIEW = 0x384
# 	SLICE_START 0 12
# 	SLICE_MAX 13 25
# 	MIP_LEVEL 26 29
regCB_COLOR_CONTROL = 0x202
# 	DISABLE_DUAL_QUAD 0 0
# 	ENABLE_1FRAG_PS_INVOKE 1 1
# 	DEGAMMA_ENABLE 3 3
# 	MODE 4 6
# 	ROP3 16 23
regCB_COVERAGE_OUT_CONTROL = 0x10a
# 	COVERAGE_OUT_ENABLE 0 0
# 	COVERAGE_OUT_MRT 1 3
# 	COVERAGE_OUT_CHANNEL 4 5
# 	COVERAGE_OUT_SAMPLES 8 11
regCB_DCC_CONFIG = 0x1427
# 	SAMPLE_MASK_TRACKER_DEPTH 0 4
# 	SAMPLE_MASK_TRACKER_DISABLE 5 5
# 	SPARE_13 6 6
# 	DISABLE_CONSTANT_ENCODE 7 7
# 	SPARE_14 8 15
# 	READ_RETURN_SKID_FIFO_DEPTH 16 24
# 	DCC_CACHE_NUM_TAGS 25 31
regCB_DCC_CONFIG2 = 0x142b
regCB_FDCC_CONTROL = 0x109
# 	SAMPLE_MASK_TRACKER_DISABLE 0 0
# 	SAMPLE_MASK_TRACKER_WATERMARK 2 6
# 	DISABLE_CONSTANT_ENCODE_AC01 8 8
# 	DISABLE_CONSTANT_ENCODE_SINGLE 9 9
# 	DISABLE_CONSTANT_ENCODE_REG 10 10
# 	DISABLE_ELIMFC_SKIP_OF_AC01 12 12
# 	DISABLE_ELIMFC_SKIP_OF_SINGLE 13 13
# 	ENABLE_ELIMFC_SKIP_OF_REG 14 14
regCB_FGCG_SRAM_OVERRIDE = 0x142a
# 	DISABLE_FGCG 0 19
regCB_HW_CONTROL = 0x1424
# 	ALLOW_MRT_WITH_DUAL_SOURCE 0 0
# 	DISABLE_VRS_FILLRATE_OPTIMIZATION 1 1
# 	DISABLE_SMT_WHEN_NO_FDCC_FIX 2 2
# 	RMI_CREDITS 6 11
# 	NUM_CCC_SKID_FIFO_ENTRIES 12 14
# 	FORCE_FEA_HIGH 15 15
# 	FORCE_EVICT_ALL_VALID 16 16
# 	DISABLE_DCC_CACHE_BYTEMASKING 17 17
# 	FORCE_NEEDS_DST 19 19
# 	DISABLE_BLEND_OPT_RESULT_EQ_DEST 21 21
# 	SPARE_2 22 22
# 	DISABLE_BLEND_OPT_DONT_RD_DST 24 24
# 	DISABLE_BLEND_OPT_BYPASS 25 25
# 	DISABLE_BLEND_OPT_DISCARD_PIXEL 26 26
# 	DISABLE_BLEND_OPT_WHEN_DISABLED_SRCALPHA_IS_USED 27 27
# 	SPARE_3 29 29
# 	DISABLE_CC_IB_SERIALIZER_STATE_OPT 30 30
# 	DISABLE_PIXEL_IN_QUAD_FIX_FOR_LINEAR_SURFACE 31 31
regCB_HW_CONTROL_1 = 0x1425
# 	CC_CACHE_NUM_TAGS 0 5
regCB_HW_CONTROL_2 = 0x1426
# 	SPARE_4 0 7
# 	DRR_ASSUMED_FIFO_DEPTH_DIV8 8 13
# 	SPARE 14 31
regCB_HW_CONTROL_3 = 0x1423
# 	SPARE_5 0 0
# 	RAM_ADDRESS_CONFLICTS_DISALLOWED 1 1
# 	SPARE_6 2 2
# 	SPARE_7 3 3
# 	DISABLE_CC_CACHE_OVWR_STATUS_ACCUM 4 4
# 	DISABLE_CC_CACHE_PANIC_GATING 5 5
# 	SPLIT_ALL_FAST_MODE_TRANSFERS 6 6
# 	DISABLE_SHADER_BLEND_OPTS 7 7
# 	FORCE_RMI_LAST_HIGH 11 11
# 	FORCE_RMI_CLKEN_HIGH 12 12
# 	DISABLE_EARLY_WRACKS_CC 13 13
# 	DISABLE_EARLY_WRACKS_DC 14 14
# 	DISABLE_NACK_PROCESSING_CC 15 15
# 	DISABLE_NACK_PROCESSING_DC 16 16
# 	SPARE_8 17 17
# 	SPARE_9 18 18
# 	DISABLE_DCC_VRS_OPT 20 20
# 	DISABLE_FMASK_NOALLOC_OPT 21 21
regCB_HW_CONTROL_4 = 0x1422
# 	COLOR_CACHE_FETCH_NUM_QB_LOG2 0 2
# 	COLOR_CACHE_FETCH_ALGORITHM 3 4
# 	DISABLE_USE_OF_SMT_SCORE 5 5
# 	SPARE_10 6 6
# 	SPARE_11 7 7
# 	SPARE_12 8 8
# 	DISABLE_MA_WAIT_FOR_LAST 9 9
# 	SMT_TIMEOUT_THRESHOLD 10 12
# 	SMT_QPFIFO_THRESHOLD 13 15
# 	ENABLE_FRAGOP_STALLING_ON_RAW_HAZARD 16 16
# 	ENABLE_FRAGOP_STALLING_ON_COARSE_RAW_HAZARD 17 17
# 	ENABLE_FRAGOP_STALLING_ON_DS_RAW_HAZARD 18 18
regCB_HW_MEM_ARBITER_RD = 0x1428
# 	MODE 0 1
# 	IGNORE_URGENT_AGE 2 5
# 	BREAK_GROUP_AGE 6 9
# 	WEIGHT_CC 10 11
# 	WEIGHT_DC 12 13
# 	WEIGHT_DECAY_REQS 14 15
# 	WEIGHT_DECAY_NOREQS 16 17
# 	WEIGHT_IGNORE_NUM_TIDS 18 18
# 	SCALE_AGE 19 21
# 	SCALE_WEIGHT 22 24
# 	SEND_LASTS_WITHIN_GROUPS 25 25
regCB_HW_MEM_ARBITER_WR = 0x1429
# 	MODE 0 1
# 	IGNORE_URGENT_AGE 2 5
# 	BREAK_GROUP_AGE 6 9
# 	WEIGHT_CC 10 11
# 	WEIGHT_DC 12 13
# 	WEIGHT_DECAY_REQS 14 15
# 	WEIGHT_DECAY_NOREQS 16 17
# 	WEIGHT_IGNORE_BYTE_MASK 18 18
# 	SCALE_AGE 19 21
# 	SCALE_WEIGHT 22 24
# 	SEND_LASTS_WITHIN_GROUPS 25 25
regCB_PERFCOUNTER0_HI = 0x3407
# 	PERFCOUNTER_HI 0 31
regCB_PERFCOUNTER0_LO = 0x3406
# 	PERFCOUNTER_LO 0 31
regCB_PERFCOUNTER0_SELECT = 0x3c01
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regCB_PERFCOUNTER0_SELECT1 = 0x3c02
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regCB_PERFCOUNTER1_HI = 0x3409
# 	PERFCOUNTER_HI 0 31
regCB_PERFCOUNTER1_LO = 0x3408
# 	PERFCOUNTER_LO 0 31
regCB_PERFCOUNTER1_SELECT = 0x3c03
# 	PERF_SEL 0 9
# 	PERF_MODE 28 31
regCB_PERFCOUNTER2_HI = 0x340b
# 	PERFCOUNTER_HI 0 31
regCB_PERFCOUNTER2_LO = 0x340a
# 	PERFCOUNTER_LO 0 31
regCB_PERFCOUNTER2_SELECT = 0x3c04
# 	PERF_SEL 0 9
# 	PERF_MODE 28 31
regCB_PERFCOUNTER3_HI = 0x340d
# 	PERFCOUNTER_HI 0 31
regCB_PERFCOUNTER3_LO = 0x340c
# 	PERFCOUNTER_LO 0 31
regCB_PERFCOUNTER3_SELECT = 0x3c05
# 	PERF_SEL 0 9
# 	PERF_MODE 28 31
regCB_PERFCOUNTER_FILTER = 0x3c00
# 	OP_FILTER_ENABLE 0 0
# 	OP_FILTER_SEL 1 3
# 	FORMAT_FILTER_ENABLE 4 4
# 	FORMAT_FILTER_SEL 5 9
# 	CLEAR_FILTER_ENABLE 10 10
# 	CLEAR_FILTER_SEL 11 11
# 	MRT_FILTER_ENABLE 12 12
# 	MRT_FILTER_SEL 13 15
# 	NUM_SAMPLES_FILTER_ENABLE 17 17
# 	NUM_SAMPLES_FILTER_SEL 18 20
# 	NUM_FRAGMENTS_FILTER_ENABLE 21 21
# 	NUM_FRAGMENTS_FILTER_SEL 22 23
regCB_RMI_GL2_CACHE_CONTROL = 0x104
# 	DCC_WR_POLICY 0 1
# 	COLOR_WR_POLICY 2 3
# 	DCC_RD_POLICY 20 21
# 	COLOR_RD_POLICY 22 23
# 	DCC_L3_BYPASS 26 26
# 	COLOR_L3_BYPASS 27 27
# 	COLOR_BIG_PAGE 31 31
regCB_SHADER_MASK = 0x8f
# 	OUTPUT0_ENABLE 0 3
# 	OUTPUT1_ENABLE 4 7
# 	OUTPUT2_ENABLE 8 11
# 	OUTPUT3_ENABLE 12 15
# 	OUTPUT4_ENABLE 16 19
# 	OUTPUT5_ENABLE 20 23
# 	OUTPUT6_ENABLE 24 27
# 	OUTPUT7_ENABLE 28 31
regCB_TARGET_MASK = 0x8e
# 	TARGET0_ENABLE 0 3
# 	TARGET1_ENABLE 4 7
# 	TARGET2_ENABLE 8 11
# 	TARGET3_ENABLE 12 15
# 	TARGET4_ENABLE 16 19
# 	TARGET5_ENABLE 20 23
# 	TARGET6_ENABLE 24 27
# 	TARGET7_ENABLE 28 31
regCC_GC_EDC_CONFIG = 0x1e38
# 	DIS_EDC 1 1
regCC_GC_PRIM_CONFIG = 0xfe0
# 	INACTIVE_PA 4 19
regCC_GC_SA_UNIT_DISABLE = 0xfe9
# 	SA_DISABLE 8 23
regCC_GC_SHADER_ARRAY_CONFIG = 0x100f
# 	INACTIVE_WGPS 16 31
regCC_GC_SHADER_RATE_CONFIG = 0x10bc
# 	DPFP_RATE 1 2
regCC_RB_BACKEND_DISABLE = 0x13dd
# 	RESERVED 2 3
# 	BACKEND_DISABLE 4 31
regCC_RB_DAISY_CHAIN = 0x13e1
# 	RB_0 0 3
# 	RB_1 4 7
# 	RB_2 8 11
# 	RB_3 12 15
# 	RB_4 16 19
# 	RB_5 20 23
# 	RB_6 24 27
# 	RB_7 28 31
regCC_RB_REDUNDANCY = 0x13dc
# 	FAILED_RB0 8 11
# 	EN_REDUNDANCY0 12 12
# 	FAILED_RB1 16 19
# 	EN_REDUNDANCY1 20 20
regCC_RMI_REDUNDANCY = 0x18a2
# 	REPAIR_EN_IN_0 1 1
# 	REPAIR_EN_IN_1 2 2
# 	REPAIR_RMI_OVERRIDE 3 3
# 	REPAIR_ID_SWAP 4 4
regCGTS_TCC_DISABLE = 0x5006
# 	HI_TCC_DISABLE 8 15
# 	TCC_DISABLE 16 31
regCGTS_USER_TCC_DISABLE = 0x5b96
# 	HI_TCC_DISABLE 8 15
# 	TCC_DISABLE 16 31
regCGTT_CPC_CLK_CTRL = 0x50b2
# 	OFF_HYSTERESIS 4 11
# 	MGLS_OVERRIDE 15 15
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE_PERFMON 29 29
# 	SOFT_OVERRIDE_DYN 30 30
# 	SOFT_OVERRIDE_REG 31 31
regCGTT_CPF_CLK_CTRL = 0x50b1
# 	OFF_HYSTERESIS 4 11
# 	MGLS_OVERRIDE 15 15
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE_PERFMON 26 26
# 	SOFT_OVERRIDE_PRT 27 27
# 	SOFT_OVERRIDE_CMP 28 28
# 	SOFT_OVERRIDE_GFX 29 29
# 	SOFT_OVERRIDE_DYN 30 30
# 	SOFT_OVERRIDE_REG 31 31
regCGTT_CP_CLK_CTRL = 0x50b0
# 	OFF_HYSTERESIS 4 11
# 	MGLS_OVERRIDE 15 15
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE_PERFMON 29 29
# 	SOFT_OVERRIDE_DYN 30 30
# 	SOFT_OVERRIDE_REG 31 31
regCGTT_GS_NGG_CLK_CTRL = 0x5087
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	PERF_ENABLE 15 15
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	PERF_OVERRIDE 27 27
# 	PRIMGEN_OVERRIDE 28 28
# 	REG_OVERRIDE 31 31
regCGTT_PA_CLK_CTRL = 0x5088
# 	CLIP_SU_PRIM_FIFO_CLK_OVERRIDE 12 12
# 	SXIFCCG_CLK_OVERRIDE 13 13
# 	AG_CLK_OVERRIDE 14 14
# 	VE_VTE_REC_CLK_OVERRIDE 15 15
# 	ENGG_CLK_OVERRIDE 16 16
# 	CL_VTE_CLK_OVERRIDE 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	AG_REG_CLK_OVERRIDE 20 20
# 	CL_VTE_REG_CLK_OVERRIDE 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	VTE_REG_CLK_OVERRIDE 24 24
# 	PERFMON_CLK_OVERRIDE 25 25
# 	SOFT_OVERRIDE5 26 26
# 	NGG_INDEX_CLK_OVERRIDE 27 27
# 	NGG_CSB_CLK_OVERRIDE 28 28
# 	SU_CLK_OVERRIDE 29 29
# 	CL_CLK_OVERRIDE 30 30
# 	SU_CL_REG_CLK_OVERRIDE 31 31
regCGTT_PH_CLK_CTRL0 = 0x50f8
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	PERFMON_CLK_OVERRIDE 30 30
# 	REG_CLK_OVERRIDE 31 31
regCGTT_PH_CLK_CTRL1 = 0x50f9
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	SOFT_OVERRIDE1 30 30
regCGTT_PH_CLK_CTRL2 = 0x50fa
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	SOFT_OVERRIDE1 30 30
regCGTT_PH_CLK_CTRL3 = 0x50fb
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	SOFT_OVERRIDE1 30 30
regCGTT_RLC_CLK_CTRL = 0x50b5
# 	RESERVED 0 31
regCGTT_SC_CLK_CTRL0 = 0x5089
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	PFF_ZFF_MEM_CLK_STALL_OVERRIDE 16 16
# 	SOFT_STALL_OVERRIDE5 17 17
# 	SOFT_STALL_OVERRIDE4 18 18
# 	SOFT_STALL_OVERRIDE3 19 19
# 	SOFT_STALL_OVERRIDE2 20 20
# 	SOFT_STALL_OVERRIDE1 21 21
# 	SOFT_STALL_OVERRIDE0 22 22
# 	REG_CLK_STALL_OVERRIDE 23 23
# 	PFF_ZFF_MEM_CLK_OVERRIDE 24 24
# 	SOFT_OVERRIDE5 25 25
# 	SOFT_OVERRIDE4 26 26
# 	SOFT_OVERRIDE3 27 27
# 	SOFT_OVERRIDE2 28 28
# 	SOFT_OVERRIDE1 29 29
# 	SOFT_OVERRIDE0 30 30
# 	REG_CLK_OVERRIDE 31 31
regCGTT_SC_CLK_CTRL1 = 0x508a
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	PBB_BINNING_CLK_STALL_OVERRIDE0 16 16
# 	PBB_BINNING_CLK_STALL_OVERRIDE 17 17
# 	PBB_SCISSOR_CLK_STALL_OVERRIDE 18 18
# 	OTHER_SPECIAL_SC_REG_CLK_STALL_OVERRIDE 19 19
# 	SCREEN_EXT_REG_CLK_STALL_OVERRIDE 20 20
# 	VPORT_REG_MEM_CLK_STALL_OVERRIDE 21 21
# 	PBB_CLK_STALL_OVERRIDE 22 22
# 	PBB_WARP_CLK_STALL_OVERRIDE 23 23
# 	PBB_BINNING_CLK_OVERRIDE0 24 24
# 	PBB_BINNING_CLK_OVERRIDE 25 25
# 	PBB_SCISSOR_CLK_OVERRIDE 26 26
# 	OTHER_SPECIAL_SC_REG_CLK_OVERRIDE 27 27
# 	SCREEN_EXT_REG_CLK_OVERRIDE 28 28
# 	VPORT_REG_MEM_CLK_OVERRIDE 29 29
# 	PBB_CLK_OVERRIDE 30 30
# 	PBB_WARP_CLK_OVERRIDE 31 31
regCGTT_SC_CLK_CTRL2 = 0x508b
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SCF_SCB_VRS_INTF_CLK_OVERRIDE 16 16
# 	SC_DB_COURSE_MGCG_BUSY_ENABLE 17 17
# 	SC_DB_STAGE_IN_TP_PFFB_WR_OVERRIDE 18 18
# 	SC_DB_QUADMASK_OVERRIDE 19 19
# 	SC_DB_QUADMASK_Z_OVERRIDE 20 20
# 	SC_DB_QUAD_PROC_OVERRIDE 21 21
# 	SC_DB_QUAD_ACCUM_OVERRIDE 22 22
# 	SC_DB_PFFB_RP_OVERRIDE 23 23
# 	SC_DB_PKR_OVERRIDE 24 24
# 	SC_DB_SC_FREE_WAVE_CLK_OVERRIDE 25 25
# 	SC_DB_SC_WAVE_2_SC_SPI_WAVE_CLK_OVERRIDE 26 26
# 	SCF_SCB_INTF_CLK_OVERRIDE 27 27
# 	SC_PKR_INTF_CLK_OVERRIDE 28 28
# 	SC_DB_INTF_CLK_OVERRIDE 29 29
# 	PA_SC_INTF_CLK_OVERRIDE 30 30
regCGTT_SC_CLK_CTRL3 = 0x50bc
# 	PBB_WARPBINROWWARP_CLK_STALL_OVERRIDE 0 0
# 	PBB_WARPBINWARP_CLK_STALL_OVERRIDE 1 1
# 	PBB_WARPFBWBINWARP_CLK_STALL_OVERRIDE 2 2
# 	PBB_WARPSCISSORUNWARP_CLK_STALL_OVERRIDE 4 4
# 	PBB_FBWBACK_CLK_STALL_OVERRIDE 5 5
# 	PBB_FBWBACKREPEATER_CLK_STALL_OVERRIDE 6 6
# 	PBB_FBWFRONT_CLK_STALL_OVERRIDE 7 7
# 	PBB_FBWFRONTREPEATER_CLK_STALL_OVERRIDE 8 8
# 	PBB_FBWSCALER_CLK_STALL_OVERRIDE 9 9
# 	PBB_FRONT_CLK_STALL_OVERRIDE 10 10
# 	PBB_BATCHIN_CLK_STALL_OVERRIDE 11 11
# 	PBB_VRASTER_CLK_STALL_OVERRIDE 12 12
# 	PBB_VGATHER_CLK_STALL_OVERRIDE 13 13
# 	PBB_WARPBINROWWARP_CLK_OVERRIDE 18 18
# 	PBB_WARPBINWARP_CLK_OVERRIDE 19 19
# 	PBB_WARPFBWBINWARP_CLK_OVERRIDE 20 20
# 	PBB_WARPSCISSORUNWARP_CLK_OVERRIDE 22 22
# 	PBB_FBWBACK_CLK_OVERRIDE 23 23
# 	PBB_FBWBACKREPEATER_CLK_OVERRIDE 24 24
# 	PBB_FBWFRONT_CLK_OVERRIDE 25 25
# 	PBB_FBWFRONTREPEATER_CLK_OVERRIDE 26 26
# 	PBB_FBWSCALER_CLK_OVERRIDE 27 27
# 	PBB_FRONT_CLK_OVERRIDE 28 28
# 	PBB_BATCHIN_CLK_OVERRIDE 29 29
# 	PBB_VRASTER_CLK_OVERRIDE 30 30
# 	PBB_VGATHER_CLK_OVERRIDE 31 31
regCGTT_SC_CLK_CTRL4 = 0x50bd
# 	PBB_VCOARSE_CLK_STALL_OVERRIDE 0 0
# 	PBB_VDETAIL_CLK_STALL_OVERRIDE 1 1
# 	PBB_HRASTER_CLK_STALL_OVERRIDE 2 2
# 	PBB_HCONFIG_CLK_STALL_OVERRIDE 3 3
# 	PBB_HGATHER_CLK_STALL_OVERRIDE 4 4
# 	PBB_HCOARSE_CLK_STALL_OVERRIDE 5 5
# 	PBB_HDETAIL_CLK_STALL_OVERRIDE 6 6
# 	PBB_HREPEAT_CLK_STALL_OVERRIDE 7 7
# 	PBB_BATCHOUT_CLK_STALL_OVERRIDE 8 8
# 	PBB_OUTPUT_CLK_STALL_OVERRIDE 9 9
# 	PBB_OUTMUX_CLK_STALL_OVERRIDE 10 10
# 	PBB_BATCHINFO_CLK_STALL_OVERRIDE 11 11
# 	PBB_EVENTINFO_CLK_STALL_OVERRIDE 12 12
# 	PBB_VCOARSE_CLK_OVERRIDE 19 19
# 	PBB_VDETAIL_CLK_OVERRIDE 20 20
# 	PBB_HRASTER_CLK_OVERRIDE 21 21
# 	PBB_HCONFIG_CLK_OVERRIDE 22 22
# 	PBB_HGATHER_CLK_OVERRIDE 23 23
# 	PBB_HCOARSE_CLK_OVERRIDE 24 24
# 	PBB_HDETAIL_CLK_OVERRIDE 25 25
# 	PBB_HREPEAT_CLK_OVERRIDE 26 26
# 	PBB_BATCHOUT_CLK_OVERRIDE 27 27
# 	PBB_OUTPUT_CLK_OVERRIDE 28 28
# 	PBB_OUTMUX_CLK_OVERRIDE 29 29
# 	PBB_BATCHINFO_CLK_OVERRIDE 30 30
# 	PBB_EVENTINFO_CLK_OVERRIDE 31 31
regCGTT_SQG_CLK_CTRL = 0x508d
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	FORCE_GL1H_CLKEN 23 23
# 	FORCE_EXPALLOC_FGCG 24 24
# 	FORCE_EXPGRANT_FGCG 25 25
# 	FORCE_EXPREQ_FGCG 26 26
# 	FORCE_CMD_FGCG 27 27
# 	TTRACE_OVERRIDE 28 28
# 	PERFMON_OVERRIDE 29 29
# 	CORE_OVERRIDE 30 30
# 	REG_OVERRIDE 31 31
regCHA_CHC_CREDITS = 0x2d88
# 	CHC_REQ_CREDITS 0 7
# 	CHCG_REQ_CREDITS 8 15
regCHA_CLIENT_FREE_DELAY = 0x2d89
# 	CLIENT_TYPE_0_FREE_DELAY 0 2
# 	CLIENT_TYPE_1_FREE_DELAY 3 5
# 	CLIENT_TYPE_2_FREE_DELAY 6 8
# 	CLIENT_TYPE_3_FREE_DELAY 9 11
# 	CLIENT_TYPE_4_FREE_DELAY 12 14
regCHA_PERFCOUNTER0_HI = 0x3601
# 	PERFCOUNTER_HI 0 31
regCHA_PERFCOUNTER0_LO = 0x3600
# 	PERFCOUNTER_LO 0 31
regCHA_PERFCOUNTER0_SELECT = 0x3de0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regCHA_PERFCOUNTER0_SELECT1 = 0x3de1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regCHA_PERFCOUNTER1_HI = 0x3603
# 	PERFCOUNTER_HI 0 31
regCHA_PERFCOUNTER1_LO = 0x3602
# 	PERFCOUNTER_LO 0 31
regCHA_PERFCOUNTER1_SELECT = 0x3de2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHA_PERFCOUNTER2_HI = 0x3605
# 	PERFCOUNTER_HI 0 31
regCHA_PERFCOUNTER2_LO = 0x3604
# 	PERFCOUNTER_LO 0 31
regCHA_PERFCOUNTER2_SELECT = 0x3de3
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHA_PERFCOUNTER3_HI = 0x3607
# 	PERFCOUNTER_HI 0 31
regCHA_PERFCOUNTER3_LO = 0x3606
# 	PERFCOUNTER_LO 0 31
regCHA_PERFCOUNTER3_SELECT = 0x3de4
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHCG_CTRL = 0x2dc2
# 	BUFFER_DEPTH_MAX 0 3
# 	VC0_BUFFER_DEPTH_MAX 4 7
# 	GL2_REQ_CREDITS 8 14
# 	GL2_DATA_CREDITS 15 21
# 	TO_L1_REPEATER_FGCG_DISABLE 22 22
# 	TO_L2_REPEATER_FGCG_DISABLE 23 23
regCHCG_PERFCOUNTER0_HI = 0x33c9
# 	PERFCOUNTER_HI 0 31
regCHCG_PERFCOUNTER0_LO = 0x33c8
# 	PERFCOUNTER_LO 0 31
regCHCG_PERFCOUNTER0_SELECT = 0x3bc6
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regCHCG_PERFCOUNTER0_SELECT1 = 0x3bc7
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regCHCG_PERFCOUNTER1_HI = 0x33cb
# 	PERFCOUNTER_HI 0 31
regCHCG_PERFCOUNTER1_LO = 0x33ca
# 	PERFCOUNTER_LO 0 31
regCHCG_PERFCOUNTER1_SELECT = 0x3bc8
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHCG_PERFCOUNTER2_HI = 0x33cd
# 	PERFCOUNTER_HI 0 31
regCHCG_PERFCOUNTER2_LO = 0x33cc
# 	PERFCOUNTER_LO 0 31
regCHCG_PERFCOUNTER2_SELECT = 0x3bc9
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHCG_PERFCOUNTER3_HI = 0x33cf
# 	PERFCOUNTER_HI 0 31
regCHCG_PERFCOUNTER3_LO = 0x33ce
# 	PERFCOUNTER_LO 0 31
regCHCG_PERFCOUNTER3_SELECT = 0x3bca
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHCG_STATUS = 0x2dc3
# 	INPUT_BUFFER_VC0_FIFO_FULL 0 0
# 	OUTPUT_FIFOS_BUSY 1 1
# 	SRC_DATA_FIFO_VC0_FULL 2 2
# 	GL2_REQ_VC0_STALL 3 3
# 	GL2_DATA_VC0_STALL 4 4
# 	GL2_REQ_VC1_STALL 5 5
# 	GL2_DATA_VC1_STALL 6 6
# 	INPUT_BUFFER_VC0_BUSY 7 7
# 	SRC_DATA_FIFO_VC0_BUSY 8 8
# 	GL2_RH_BUSY 9 9
# 	NUM_REQ_PENDING_FROM_L2 10 19
# 	VIRTUAL_FIFO_FULL_STALL 20 20
# 	REQUEST_TRACKER_BUFFER_STALL 21 21
# 	REQUEST_TRACKER_BUSY 22 22
# 	BUFFER_FULL 23 23
# 	INPUT_BUFFER_VC1_BUSY 24 24
# 	SRC_DATA_FIFO_VC1_BUSY 25 25
# 	INPUT_BUFFER_VC1_FIFO_FULL 26 26
# 	SRC_DATA_FIFO_VC1_FULL 27 27
regCHC_CTRL = 0x2dc0
# 	BUFFER_DEPTH_MAX 0 3
# 	GL2_REQ_CREDITS 4 10
# 	GL2_DATA_CREDITS 11 17
# 	TO_L1_REPEATER_FGCG_DISABLE 18 18
# 	TO_L2_REPEATER_FGCG_DISABLE 19 19
# 	DISABLE_PERF_WR_DATA_ALLOC_COUNT 29 29
regCHC_PERFCOUNTER0_HI = 0x33c1
# 	PERFCOUNTER_HI 0 31
regCHC_PERFCOUNTER0_LO = 0x33c0
# 	PERFCOUNTER_LO 0 31
regCHC_PERFCOUNTER0_SELECT = 0x3bc0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regCHC_PERFCOUNTER0_SELECT1 = 0x3bc1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regCHC_PERFCOUNTER1_HI = 0x33c3
# 	PERFCOUNTER_HI 0 31
regCHC_PERFCOUNTER1_LO = 0x33c2
# 	PERFCOUNTER_LO 0 31
regCHC_PERFCOUNTER1_SELECT = 0x3bc2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHC_PERFCOUNTER2_HI = 0x33c5
# 	PERFCOUNTER_HI 0 31
regCHC_PERFCOUNTER2_LO = 0x33c4
# 	PERFCOUNTER_LO 0 31
regCHC_PERFCOUNTER2_SELECT = 0x3bc3
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHC_PERFCOUNTER3_HI = 0x33c7
# 	PERFCOUNTER_HI 0 31
regCHC_PERFCOUNTER3_LO = 0x33c6
# 	PERFCOUNTER_LO 0 31
regCHC_PERFCOUNTER3_SELECT = 0x3bc4
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regCHC_STATUS = 0x2dc1
# 	INPUT_BUFFER_VC0_FIFO_FULL 0 0
# 	OUTPUT_FIFOS_BUSY 1 1
# 	SRC_DATA_FIFO_VC0_FULL 2 2
# 	GL2_REQ_VC0_STALL 3 3
# 	GL2_DATA_VC0_STALL 4 4
# 	GL2_REQ_VC1_STALL 5 5
# 	GL2_DATA_VC1_STALL 6 6
# 	INPUT_BUFFER_VC0_BUSY 7 7
# 	SRC_DATA_FIFO_VC0_BUSY 8 8
# 	GL2_RH_BUSY 9 9
# 	NUM_REQ_PENDING_FROM_L2 10 19
# 	VIRTUAL_FIFO_FULL_STALL 20 20
# 	REQUEST_TRACKER_BUFFER_STALL 21 21
# 	REQUEST_TRACKER_BUSY 22 22
# 	BUFFER_FULL 23 23
regCHICKEN_BITS = 0x142d
# 	SPARE 0 31
regCHI_CHR_MGCG_OVERRIDE = 0x50e9
# 	CHA_CHIR_MGCG_SCLK_OVERRIDE 0 0
# 	CHA_CHIR_MGCG_RET_DCLK_OVERRIDE 1 1
# 	CHA_CHIW_MGCG_SCLK_OVERRIDE 2 2
# 	CHA_CHIW_MGCG_RET_DCLK_OVERRIDE 3 3
# 	CHA_CHIW_MGCG_SRC_DCLK_OVERRIDE 4 4
# 	CHA_CHR_RET_MGCG_SCLK_OVERRIDE 5 5
# 	CHA_CHR_SRC_MGCG_SCLK_OVERRIDE 6 6
regCHI_CHR_REP_FGCG_OVERRIDE = 0x2d8c
# 	CHA_CHIW_REP_FGCG_OVERRIDE 0 0
# 	CHA_CHIR_REP_FGCG_OVERRIDE 1 1
# 	CHA_CHR_SRC_REP_FGCG_OVERRIDE 2 2
# 	CHA_CHR_RET_REP_FGCG_OVERRIDE 3 3
regCH_ARB_CTRL = 0x2d80
# 	NUM_MEM_PIPES 0 1
# 	UC_IO_WR_PATH 2 2
# 	FGCG_DISABLE 3 3
# 	PERF_CNTR_EN_OVERRIDE 4 4
# 	CHICKEN_BITS 5 12
regCH_ARB_STATUS = 0x2d83
# 	REQ_ARB_BUSY 0 0
# 	RET_ARB_BUSY 1 1
regCH_DRAM_BURST_CTRL = 0x2d84
# 	MAX_DRAM_BURST 0 2
# 	BURST_DISABLE 3 3
# 	GATHER_64B_MEMORY_BURST_DISABLE 4 4
# 	GATHER_64B_IO_BURST_DISABLE 5 5
# 	GATHER_32B_MEMORY_BURST_DISABLE 6 6
# 	GATHER_32B_IO_BURST_DISABLE 7 7
# 	WRITE_BURSTABLE_STALL_DISABLE 8 8
regCH_DRAM_BURST_MASK = 0x2d82
# 	DRAM_BURST_ADDR_MASK 0 7
regCH_PIPE_STEER = 0x5b88
# 	PIPE0 0 1
# 	PIPE1 2 3
# 	PIPE2 4 5
# 	PIPE3 6 7
regCH_VC5_ENABLE = 0x2d94
# 	UTCL2_VC5_ENABLE 1 1
regCOHER_DEST_BASE_0 = 0x92
# 	DEST_BASE_256B 0 31
regCOHER_DEST_BASE_1 = 0x93
# 	DEST_BASE_256B 0 31
regCOHER_DEST_BASE_2 = 0x7e
# 	DEST_BASE_256B 0 31
regCOHER_DEST_BASE_3 = 0x7f
# 	DEST_BASE_256B 0 31
regCOHER_DEST_BASE_HI_0 = 0x7a
# 	DEST_BASE_HI_256B 0 7
regCOHER_DEST_BASE_HI_1 = 0x7b
# 	DEST_BASE_HI_256B 0 7
regCOHER_DEST_BASE_HI_2 = 0x7c
# 	DEST_BASE_HI_256B 0 7
regCOHER_DEST_BASE_HI_3 = 0x7d
# 	DEST_BASE_HI_256B 0 7
regCOMPUTE_DDID_INDEX = 0x1bc9
# 	INDEX 0 10
regCOMPUTE_DESTINATION_EN_SE0 = 0x1bb6
# 	CU_EN 0 31
regCOMPUTE_DESTINATION_EN_SE1 = 0x1bb7
# 	CU_EN 0 31
regCOMPUTE_DESTINATION_EN_SE2 = 0x1bb9
# 	CU_EN 0 31
regCOMPUTE_DESTINATION_EN_SE3 = 0x1bba
# 	CU_EN 0 31
regCOMPUTE_DIM_X = 0x1ba1
# 	SIZE 0 31
regCOMPUTE_DIM_Y = 0x1ba2
# 	SIZE 0 31
regCOMPUTE_DIM_Z = 0x1ba3
# 	SIZE 0 31
regCOMPUTE_DISPATCH_END = 0x1c1e
# 	DATA 0 31
regCOMPUTE_DISPATCH_ID = 0x1bc0
# 	DISPATCH_ID 0 31
regCOMPUTE_DISPATCH_INITIATOR = 0x1ba0
# 	COMPUTE_SHADER_EN 0 0
# 	PARTIAL_TG_EN 1 1
# 	FORCE_START_AT_000 2 2
# 	ORDERED_APPEND_ENBL 3 3
# 	ORDERED_APPEND_MODE 4 4
# 	USE_THREAD_DIMENSIONS 5 5
# 	ORDER_MODE 6 6
# 	SCALAR_L1_INV_VOL 10 10
# 	VECTOR_L1_INV_VOL 11 11
# 	RESERVED 12 12
# 	TUNNEL_ENABLE 13 13
# 	RESTORE 14 14
# 	CS_W32_EN 15 15
# 	AMP_SHADER_EN 16 16
# 	DISABLE_DISP_PREMPT_EN 17 17
regCOMPUTE_DISPATCH_INTERLEAVE = 0x1bcf
# 	INTERLEAVE 0 9
regCOMPUTE_DISPATCH_PKT_ADDR_HI = 0x1baf
# 	DATA 0 7
regCOMPUTE_DISPATCH_PKT_ADDR_LO = 0x1bae
# 	DATA 0 31
regCOMPUTE_DISPATCH_SCRATCH_BASE_HI = 0x1bb1
# 	DATA 0 7
regCOMPUTE_DISPATCH_SCRATCH_BASE_LO = 0x1bb0
# 	DATA 0 31
regCOMPUTE_DISPATCH_TUNNEL = 0x1c1d
# 	OFF_DELAY 0 9
# 	IMMEDIATE 10 10
regCOMPUTE_MISC_RESERVED = 0x1bbf
# 	SEND_SEID 0 2
# 	RESERVED3 3 3
# 	RESERVED4 4 4
# 	WAVE_ID_BASE 5 16
regCOMPUTE_NOWHERE = 0x1c1f
# 	DATA 0 31
regCOMPUTE_NUM_THREAD_X = 0x1ba7
# 	NUM_THREAD_FULL 0 15
# 	NUM_THREAD_PARTIAL 16 31
regCOMPUTE_NUM_THREAD_Y = 0x1ba8
# 	NUM_THREAD_FULL 0 15
# 	NUM_THREAD_PARTIAL 16 31
regCOMPUTE_NUM_THREAD_Z = 0x1ba9
# 	NUM_THREAD_FULL 0 15
# 	NUM_THREAD_PARTIAL 16 31
regCOMPUTE_PERFCOUNT_ENABLE = 0x1bab
# 	PERFCOUNT_ENABLE 0 0
regCOMPUTE_PGM_HI = 0x1bad
# 	DATA 0 7
regCOMPUTE_PGM_LO = 0x1bac
# 	DATA 0 31
regCOMPUTE_PGM_RSRC1 = 0x1bb2
# 	VGPRS 0 5
# 	SGPRS 6 9
# 	PRIORITY 10 11
# 	FLOAT_MODE 12 19
# 	PRIV 20 20
# 	DX10_CLAMP 21 21
# 	IEEE_MODE 23 23
# 	BULKY 24 24
# 	FP16_OVFL 26 26
# 	WGP_MODE 29 29
# 	MEM_ORDERED 30 30
# 	FWD_PROGRESS 31 31
regCOMPUTE_PGM_RSRC2 = 0x1bb3
# 	SCRATCH_EN 0 0
# 	USER_SGPR 1 5
# 	TRAP_PRESENT 6 6
# 	TGID_X_EN 7 7
# 	TGID_Y_EN 8 8
# 	TGID_Z_EN 9 9
# 	TG_SIZE_EN 10 10
# 	TIDIG_COMP_CNT 11 12
# 	EXCP_EN_MSB 13 14
# 	LDS_SIZE 15 23
# 	EXCP_EN 24 30
regCOMPUTE_PGM_RSRC3 = 0x1bc8
# 	SHARED_VGPR_CNT 0 3
# 	INST_PREF_SIZE 4 9
# 	TRAP_ON_START 10 10
# 	TRAP_ON_END 11 11
# 	IMAGE_OP 31 31
regCOMPUTE_PIPELINESTAT_ENABLE = 0x1baa
# 	PIPELINESTAT_ENABLE 0 0
regCOMPUTE_RELAUNCH = 0x1bd0
# 	PAYLOAD 0 29
# 	IS_EVENT 30 30
# 	IS_STATE 31 31
regCOMPUTE_RELAUNCH2 = 0x1bd3
# 	PAYLOAD 0 29
# 	IS_EVENT 30 30
# 	IS_STATE 31 31
regCOMPUTE_REQ_CTRL = 0x1bc2
# 	SOFT_GROUPING_EN 0 0
# 	NUMBER_OF_REQUESTS_PER_CU 1 4
# 	SOFT_GROUPING_ALLOCATION_TIMEOUT 5 8
# 	HARD_LOCK_HYSTERESIS 9 9
# 	HARD_LOCK_LOW_THRESHOLD 10 14
# 	PRODUCER_REQUEST_LOCKOUT 15 15
# 	GLOBAL_SCANNING_EN 16 16
# 	ALLOCATION_RATE_THROTTLING_THRESHOLD 17 19
# 	DEDICATED_PREALLOCATION_BUFFER_LIMIT 20 26
regCOMPUTE_RESOURCE_LIMITS = 0x1bb5
# 	WAVES_PER_SH 0 9
# 	TG_PER_CU 12 15
# 	LOCK_THRESHOLD 16 21
# 	SIMD_DEST_CNTL 22 22
# 	FORCE_SIMD_DIST 23 23
# 	CU_GROUP_COUNT 24 26
regCOMPUTE_RESTART_X = 0x1bbb
# 	RESTART 0 31
regCOMPUTE_RESTART_Y = 0x1bbc
# 	RESTART 0 31
regCOMPUTE_RESTART_Z = 0x1bbd
# 	RESTART 0 31
regCOMPUTE_SHADER_CHKSUM = 0x1bca
# 	CHECKSUM 0 31
regCOMPUTE_START_X = 0x1ba4
# 	START 0 31
regCOMPUTE_START_Y = 0x1ba5
# 	START 0 31
regCOMPUTE_START_Z = 0x1ba6
# 	START 0 31
regCOMPUTE_STATIC_THREAD_MGMT_SE0 = 0x1bb6
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE1 = 0x1bb7
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE2 = 0x1bb9
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE3 = 0x1bba
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE4 = 0x1bcb
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE5 = 0x1bcc
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE6 = 0x1bcd
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_STATIC_THREAD_MGMT_SE7 = 0x1bce
# 	SA0_CU_EN 0 15
# 	SA1_CU_EN 16 31
regCOMPUTE_THREADGROUP_ID = 0x1bc1
# 	THREADGROUP_ID 0 31
regCOMPUTE_THREAD_TRACE_ENABLE = 0x1bbe
# 	THREAD_TRACE_ENABLE 0 0
regCOMPUTE_TMPRING_SIZE = 0x1bb8
# 	WAVES 0 11
# 	WAVESIZE 12 26
regCOMPUTE_USER_ACCUM_0 = 0x1bc4
# 	CONTRIBUTION 0 6
regCOMPUTE_USER_ACCUM_1 = 0x1bc5
# 	CONTRIBUTION 0 6
regCOMPUTE_USER_ACCUM_2 = 0x1bc6
# 	CONTRIBUTION 0 6
regCOMPUTE_USER_ACCUM_3 = 0x1bc7
# 	CONTRIBUTION 0 6
regCOMPUTE_USER_DATA_0 = 0x1be0
# 	DATA 0 31
regCOMPUTE_USER_DATA_1 = 0x1be1
# 	DATA 0 31
regCOMPUTE_USER_DATA_10 = 0x1bea
# 	DATA 0 31
regCOMPUTE_USER_DATA_11 = 0x1beb
# 	DATA 0 31
regCOMPUTE_USER_DATA_12 = 0x1bec
# 	DATA 0 31
regCOMPUTE_USER_DATA_13 = 0x1bed
# 	DATA 0 31
regCOMPUTE_USER_DATA_14 = 0x1bee
# 	DATA 0 31
regCOMPUTE_USER_DATA_15 = 0x1bef
# 	DATA 0 31
regCOMPUTE_USER_DATA_2 = 0x1be2
# 	DATA 0 31
regCOMPUTE_USER_DATA_3 = 0x1be3
# 	DATA 0 31
regCOMPUTE_USER_DATA_4 = 0x1be4
# 	DATA 0 31
regCOMPUTE_USER_DATA_5 = 0x1be5
# 	DATA 0 31
regCOMPUTE_USER_DATA_6 = 0x1be6
# 	DATA 0 31
regCOMPUTE_USER_DATA_7 = 0x1be7
# 	DATA 0 31
regCOMPUTE_USER_DATA_8 = 0x1be8
# 	DATA 0 31
regCOMPUTE_USER_DATA_9 = 0x1be9
# 	DATA 0 31
regCOMPUTE_VMID = 0x1bb4
# 	DATA 0 3
regCOMPUTE_WAVE_RESTORE_ADDR_HI = 0x1bd2
# 	ADDR 0 15
regCOMPUTE_WAVE_RESTORE_ADDR_LO = 0x1bd1
# 	ADDR 0 31
regCONFIG_RESERVED_REG0 = 0x800
# 	DATA 0 31
regCONFIG_RESERVED_REG1 = 0x801
# 	DATA 0 31
regCONTEXT_RESERVED_REG0 = 0xdb
# 	DATA 0 31
regCONTEXT_RESERVED_REG1 = 0xdc
# 	DATA 0 31
regCPC_DDID_BASE_ADDR_HI = 0x1e6c
# 	BASE_ADDR_HI 0 15
regCPC_DDID_BASE_ADDR_LO = 0x1e6b
# 	BASE_ADDR_LO 6 31
regCPC_DDID_CNTL = 0x1e6d
# 	THRESHOLD 0 7
# 	SIZE 16 16
# 	NO_RING_MEMORY 19 19
# 	POLICY 28 29
# 	MODE 30 30
# 	ENABLE 31 31
regCPC_INT_ADDR = 0x1dd9
# 	ADDR 0 31
regCPC_INT_CNTL = 0x1e54
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCPC_INT_CNTX_ID = 0x1e57
# 	CNTX_ID 0 31
regCPC_INT_INFO = 0x1dd7
# 	ADDR_HI 0 15
# 	TYPE 16 16
# 	VMID 20 23
# 	QUEUE_ID 28 30
regCPC_INT_PASID = 0x1dda
# 	PASID 0 15
# 	BYPASS_PASID 16 16
regCPC_INT_STATUS = 0x1e55
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCPC_LATENCY_STATS_DATA = 0x300e
# 	DATA 0 31
regCPC_LATENCY_STATS_SELECT = 0x380e
# 	INDEX 0 3
# 	CLEAR 30 30
# 	ENABLE 31 31
regCPC_OS_PIPES = 0x1e67
# 	OS_PIPES 0 7
regCPC_PERFCOUNTER0_HI = 0x3007
# 	PERFCOUNTER_HI 0 31
regCPC_PERFCOUNTER0_LO = 0x3006
# 	PERFCOUNTER_LO 0 31
regCPC_PERFCOUNTER0_SELECT = 0x3809
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	SPM_MODE 20 23
# 	CNTR_MODE1 24 27
# 	CNTR_MODE0 28 31
regCPC_PERFCOUNTER0_SELECT1 = 0x3804
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	CNTR_MODE3 24 27
# 	CNTR_MODE2 28 31
regCPC_PERFCOUNTER1_HI = 0x3005
# 	PERFCOUNTER_HI 0 31
regCPC_PERFCOUNTER1_LO = 0x3004
# 	PERFCOUNTER_LO 0 31
regCPC_PERFCOUNTER1_SELECT = 0x3803
# 	PERF_SEL 0 9
# 	SPM_MODE 20 23
# 	CNTR_MODE 28 31
regCPC_PSP_DEBUG = 0x5c11
# 	PRIV_VIOLATION_CNTL 0 1
# 	GPA_OVERRIDE 3 3
# 	UCODE_VF_OVERRIDE 4 4
# 	MTYPE_TMZ_OVERRIDE 5 5
# 	SECURE_REG_OVERRIDE 6 6
regCPC_SUSPEND_CNTL_STACK_OFFSET = 0x1e63
# 	OFFSET 2 15
regCPC_SUSPEND_CNTL_STACK_SIZE = 0x1e64
# 	SIZE 12 15
regCPC_SUSPEND_CTX_SAVE_BASE_ADDR_HI = 0x1e61
# 	ADDR_HI 0 15
regCPC_SUSPEND_CTX_SAVE_BASE_ADDR_LO = 0x1e60
# 	ADDR 12 31
regCPC_SUSPEND_CTX_SAVE_CONTROL = 0x1e62
# 	POLICY 3 4
# 	EXE_DISABLE 23 23
regCPC_SUSPEND_CTX_SAVE_SIZE = 0x1e66
# 	SIZE 12 25
regCPC_SUSPEND_WG_STATE_OFFSET = 0x1e65
# 	OFFSET 2 25
regCPC_TC_PERF_COUNTER_WINDOW_SELECT = 0x380f
# 	INDEX 0 4
# 	ALWAYS 30 30
# 	ENABLE 31 31
regCPC_UTCL1_CNTL = 0x1ddd
# 	XNACK_REDO_TIMER_CNT 0 19
# 	DROP_MODE 24 24
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	IGNORE_PTE_PERMISSION 29 29
# 	MTYPE_NO_PTE_MODE 30 30
regCPC_UTCL1_ERROR = 0x1dff
# 	ERROR_DETECTED_HALT 0 0
regCPC_UTCL1_STATUS = 0x1f55
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29
regCPF_GCR_CNTL = 0x1f53
# 	GCR_GL_CMD 0 18
regCPF_LATENCY_STATS_DATA = 0x300c
# 	DATA 0 31
regCPF_LATENCY_STATS_SELECT = 0x380c
# 	INDEX 0 3
# 	CLEAR 30 30
# 	ENABLE 31 31
regCPF_PERFCOUNTER0_HI = 0x300b
# 	PERFCOUNTER_HI 0 31
regCPF_PERFCOUNTER0_LO = 0x300a
# 	PERFCOUNTER_LO 0 31
regCPF_PERFCOUNTER0_SELECT = 0x3807
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	SPM_MODE 20 23
# 	CNTR_MODE1 24 27
# 	CNTR_MODE0 28 31
regCPF_PERFCOUNTER0_SELECT1 = 0x3806
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	CNTR_MODE3 24 27
# 	CNTR_MODE2 28 31
regCPF_PERFCOUNTER1_HI = 0x3009
# 	PERFCOUNTER_HI 0 31
regCPF_PERFCOUNTER1_LO = 0x3008
# 	PERFCOUNTER_LO 0 31
regCPF_PERFCOUNTER1_SELECT = 0x3805
# 	PERF_SEL 0 9
# 	SPM_MODE 20 23
# 	CNTR_MODE 28 31
regCPF_TC_PERF_COUNTER_WINDOW_SELECT = 0x380a
# 	INDEX 0 2
# 	ALWAYS 30 30
# 	ENABLE 31 31
regCPF_UTCL1_CNTL = 0x1dde
# 	XNACK_REDO_TIMER_CNT 0 19
# 	VMID_RESET_MODE 23 23
# 	DROP_MODE 24 24
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	IGNORE_PTE_PERMISSION 29 29
# 	MTYPE_NO_PTE_MODE 30 30
# 	FORCE_NO_EXE 31 31
regCPF_UTCL1_STATUS = 0x1f56
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29
regCPG_LATENCY_STATS_DATA = 0x300d
# 	DATA 0 31
regCPG_LATENCY_STATS_SELECT = 0x380d
# 	INDEX 0 4
# 	CLEAR 30 30
# 	ENABLE 31 31
regCPG_PERFCOUNTER0_HI = 0x3003
# 	PERFCOUNTER_HI 0 31
regCPG_PERFCOUNTER0_LO = 0x3002
# 	PERFCOUNTER_LO 0 31
regCPG_PERFCOUNTER0_SELECT = 0x3802
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	SPM_MODE 20 23
# 	CNTR_MODE1 24 27
# 	CNTR_MODE0 28 31
regCPG_PERFCOUNTER0_SELECT1 = 0x3801
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	CNTR_MODE3 24 27
# 	CNTR_MODE2 28 31
regCPG_PERFCOUNTER1_HI = 0x3001
# 	PERFCOUNTER_HI 0 31
regCPG_PERFCOUNTER1_LO = 0x3000
# 	PERFCOUNTER_LO 0 31
regCPG_PERFCOUNTER1_SELECT = 0x3800
# 	PERF_SEL 0 9
# 	SPM_MODE 20 23
# 	CNTR_MODE 28 31
regCPG_PSP_DEBUG = 0x5c10
# 	PRIV_VIOLATION_CNTL 0 1
# 	VMID_VIOLATION_CNTL 2 2
# 	GPA_OVERRIDE 3 3
# 	UCODE_VF_OVERRIDE 4 4
# 	MTYPE_TMZ_OVERRIDE 5 5
# 	SECURE_REG_OVERRIDE 6 6
regCPG_RCIU_CAM_DATA = 0x1f45
# 	DATA 0 31
regCPG_RCIU_CAM_DATA_PHASE0 = 0x1f45
# 	ADDR 0 17
# 	PIPE0_EN 24 24
# 	PIPE1_EN 25 25
# 	SKIP_WR 31 31
regCPG_RCIU_CAM_DATA_PHASE1 = 0x1f45
# 	MASK 0 31
regCPG_RCIU_CAM_DATA_PHASE2 = 0x1f45
# 	VALUE 0 31
regCPG_RCIU_CAM_INDEX = 0x1f44
# 	INDEX 0 4
regCPG_TC_PERF_COUNTER_WINDOW_SELECT = 0x380b
# 	INDEX 0 4
# 	ALWAYS 30 30
# 	ENABLE 31 31
regCPG_UTCL1_CNTL = 0x1ddc
# 	XNACK_REDO_TIMER_CNT 0 19
# 	VMID_RESET_MODE 23 23
# 	DROP_MODE 24 24
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	IGNORE_PTE_PERMISSION 29 29
# 	MTYPE_NO_PTE_MODE 30 30
regCPG_UTCL1_ERROR = 0x1dfe
# 	ERROR_DETECTED_HALT 0 0
regCPG_UTCL1_STATUS = 0x1f54
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29
regCP_APPEND_ADDR_HI = 0x2059
# 	MEM_ADDR_HI 0 15
# 	CS_PS_SEL 16 17
# 	FENCE_SIZE 18 18
# 	PWS_ENABLE 19 19
# 	CACHE_POLICY 25 26
# 	COMMAND 29 31
regCP_APPEND_ADDR_LO = 0x2058
# 	MEM_ADDR_LO 2 31
regCP_APPEND_CMD_ADDR_HI = 0x20a1
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_APPEND_CMD_ADDR_LO = 0x20a0
# 	RSVD 0 1
# 	ADDR_LO 2 31
regCP_APPEND_DATA = 0x205a
# 	DATA 0 31
regCP_APPEND_DATA_HI = 0x204c
# 	DATA 0 31
regCP_APPEND_DATA_LO = 0x205a
# 	DATA 0 31
regCP_APPEND_DDID_CNT = 0x204b
# 	DATA 0 7
regCP_APPEND_LAST_CS_FENCE = 0x205b
# 	LAST_FENCE 0 31
regCP_APPEND_LAST_CS_FENCE_HI = 0x204d
# 	LAST_FENCE 0 31
regCP_APPEND_LAST_CS_FENCE_LO = 0x205b
# 	LAST_FENCE 0 31
regCP_APPEND_LAST_PS_FENCE = 0x205c
# 	LAST_FENCE 0 31
regCP_APPEND_LAST_PS_FENCE_HI = 0x204e
# 	LAST_FENCE 0 31
regCP_APPEND_LAST_PS_FENCE_LO = 0x205c
# 	LAST_FENCE 0 31
regCP_AQL_SMM_STATUS = 0x1ddf
# 	AQL_QUEUE_SMM 0 31
regCP_ATOMIC_PREOP_HI = 0x205e
# 	ATOMIC_PREOP_HI 0 31
regCP_ATOMIC_PREOP_LO = 0x205d
# 	ATOMIC_PREOP_LO 0 31
regCP_BUSY_STAT = 0xf3f
# 	REG_BUS_FIFO_BUSY 0 0
# 	COHER_CNT_NEQ_ZERO 6 6
# 	PFP_PARSING_PACKETS 7 7
# 	ME_PARSING_PACKETS 8 8
# 	RCIU_PFP_BUSY 9 9
# 	RCIU_ME_BUSY 10 10
# 	SEM_CMDFIFO_NOT_EMPTY 12 12
# 	SEM_FAILED_AND_HOLDING 13 13
# 	SEM_POLLING_FOR_PASS 14 14
# 	GFX_CONTEXT_BUSY 15 15
# 	ME_PARSER_BUSY 17 17
# 	EOP_DONE_BUSY 18 18
# 	STRM_OUT_BUSY 19 19
# 	PIPE_STATS_BUSY 20 20
# 	RCIU_CE_BUSY 21 21
# 	CE_PARSING_PACKETS 22 22
regCP_CMD_DATA = 0xf7f
# 	CMD_DATA 0 31
regCP_CMD_INDEX = 0xf7e
# 	CMD_INDEX 0 10
# 	CMD_ME_SEL 12 13
# 	CMD_QUEUE_SEL 16 18
regCP_CNTX_STAT = 0xf58
# 	ACTIVE_HP3D_CONTEXTS 0 7
# 	CURRENT_HP3D_CONTEXT 8 10
# 	ACTIVE_GFX_CONTEXTS 20 27
# 	CURRENT_GFX_CONTEXT 28 30
regCP_CONTEXT_CNTL = 0x1e4d
# 	ME0PIPE0_MAX_GE_CNTX 0 2
# 	ME0PIPE0_MAX_PIPE_CNTX 4 6
# 	ME0PIPE1_MAX_GE_CNTX 16 18
# 	ME0PIPE1_MAX_PIPE_CNTX 20 22
regCP_CPC_BUSY_HYSTERESIS = 0x1edb
# 	CAC_ACTIVE 0 7
# 	CPC_BUSY 8 15
regCP_CPC_BUSY_STAT = 0xe25
# 	MEC1_LOAD_BUSY 0 0
# 	MEC1_SEMAPHORE_BUSY 1 1
# 	MEC1_MUTEX_BUSY 2 2
# 	MEC1_MESSAGE_BUSY 3 3
# 	MEC1_EOP_QUEUE_BUSY 4 4
# 	MEC1_IQ_QUEUE_BUSY 5 5
# 	MEC1_IB_QUEUE_BUSY 6 6
# 	MEC1_TC_BUSY 7 7
# 	MEC1_DMA_BUSY 8 8
# 	MEC1_PARTIAL_FLUSH_BUSY 9 9
# 	MEC1_PIPE0_BUSY 10 10
# 	MEC1_PIPE1_BUSY 11 11
# 	MEC1_PIPE2_BUSY 12 12
# 	MEC1_PIPE3_BUSY 13 13
# 	MEC2_LOAD_BUSY 16 16
# 	MEC2_SEMAPHORE_BUSY 17 17
# 	MEC2_MUTEX_BUSY 18 18
# 	MEC2_MESSAGE_BUSY 19 19
# 	MEC2_EOP_QUEUE_BUSY 20 20
# 	MEC2_IQ_QUEUE_BUSY 21 21
# 	MEC2_IB_QUEUE_BUSY 22 22
# 	MEC2_TC_BUSY 23 23
# 	MEC2_DMA_BUSY 24 24
# 	MEC2_PARTIAL_FLUSH_BUSY 25 25
# 	MEC2_PIPE0_BUSY 26 26
# 	MEC2_PIPE1_BUSY 27 27
# 	MEC2_PIPE2_BUSY 28 28
# 	MEC2_PIPE3_BUSY 29 29
regCP_CPC_BUSY_STAT2 = 0xe2a
# 	MES_LOAD_BUSY 0 0
# 	MES_MUTEX_BUSY 2 2
# 	MES_MESSAGE_BUSY 3 3
# 	MES_TC_BUSY 7 7
# 	MES_DMA_BUSY 8 8
# 	MES_PIPE0_BUSY 10 10
# 	MES_PIPE1_BUSY 11 11
# 	MES_PIPE2_BUSY 12 12
# 	MES_PIPE3_BUSY 13 13
regCP_CPC_DEBUG = 0x1e21
# 	PIPE_SELECT 0 1
# 	ME_SELECT 2 2
# 	ADC_INTERLEAVE_DISABLE 4 4
# 	DEBUG_BUS_FLOP_EN 14 14
# 	CPC_REPEATER_FGCG_OVERRIDE 15 15
# 	CPC_CHIU_NOALLOC_OVERRIDE 16 16
# 	CPC_GCR_CNTL_BYPASS 17 17
# 	CPC_RAM_CLK_GATING_DISABLE 18 18
# 	PRIV_VIOLATION_WRITE_DISABLE 20 20
# 	UCODE_ECC_ERROR_DISABLE 21 21
# 	INTERRUPT_DISABLE 22 22
# 	CPC_CHIU_RO_DISABLE 23 23
# 	UNDERFLOW_BUSY_DISABLE 24 24
# 	OVERFLOW_BUSY_DISABLE 25 25
# 	EVENT_FILT_DISABLE 26 26
# 	CPC_CHIU_GUS_DISABLE 27 27
# 	CPC_TC_ONE_CYCLE_WRITE_DISABLE 28 28
# 	CS_STATE_FILT_DISABLE 29 29
# 	CPC_CHIU_MTYPE_OVERRIDE 30 30
# 	ME2_UCODE_RAM_ENABLE 31 31
regCP_CPC_DEBUG_CNTL = 0xe20
# 	DEBUG_INDX 0 6
regCP_CPC_DEBUG_DATA = 0xe21
# 	DEBUG_DATA 0 31
regCP_CPC_GFX_CNTL = 0x1f5a
# 	QUEUEID 0 2
# 	PIPEID 3 4
# 	MEID 5 6
# 	VALID 7 7
regCP_CPC_GRBM_FREE_COUNT = 0xe2b
# 	FREE_COUNT 0 5
regCP_CPC_HALT_HYST_COUNT = 0xe47
# 	COUNT 0 3
regCP_CPC_IC_BASE_CNTL = 0x584e
# 	VMID 0 3
# 	ADDRESS_CLAMP 4 4
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
regCP_CPC_IC_BASE_HI = 0x584d
# 	IC_BASE_HI 0 15
regCP_CPC_IC_BASE_LO = 0x584c
# 	IC_BASE_LO 12 31
regCP_CPC_IC_OP_CNTL = 0x297a
# 	INVALIDATE_CACHE 0 0
# 	INVALIDATE_CACHE_COMPLETE 1 1
# 	PRIME_ICACHE 4 4
# 	ICACHE_PRIMED 5 5
regCP_CPC_MGCG_SYNC_CNTL = 0x1dd6
# 	COOLDOWN_PERIOD 0 7
# 	WARMUP_PERIOD 8 15
regCP_CPC_PRIV_VIOLATION_ADDR = 0xe2c
# 	PRIV_VIOLATION_ADDR 0 17
regCP_CPC_SCRATCH_DATA = 0xe31
# 	SCRATCH_DATA 0 31
regCP_CPC_SCRATCH_INDEX = 0xe30
# 	SCRATCH_INDEX 0 8
# 	SCRATCH_INDEX_64BIT_MODE 31 31
regCP_CPC_STALLED_STAT1 = 0xe26
# 	RCIU_TX_FREE_STALL 3 3
# 	RCIU_PRIV_VIOLATION 4 4
# 	TCIU_TX_FREE_STALL 6 6
# 	TCIU_WAITING_ON_TAGS 7 7
# 	MEC1_DECODING_PACKET 8 8
# 	MEC1_WAIT_ON_RCIU 9 9
# 	MEC1_WAIT_ON_RCIU_READ 10 10
# 	MEC1_WAIT_ON_ROQ_DATA 13 13
# 	MEC2_DECODING_PACKET 16 16
# 	MEC2_WAIT_ON_RCIU 17 17
# 	MEC2_WAIT_ON_RCIU_READ 18 18
# 	MEC2_WAIT_ON_ROQ_DATA 21 21
# 	UTCL2IU_WAITING_ON_FREE 22 22
# 	UTCL2IU_WAITING_ON_TAGS 23 23
# 	UTCL1_WAITING_ON_TRANS 24 24
# 	GCRIU_WAITING_ON_FREE 25 25
regCP_CPC_STATUS = 0xe24
# 	MEC1_BUSY 0 0
# 	MEC2_BUSY 1 1
# 	DC0_BUSY 2 2
# 	DC1_BUSY 3 3
# 	RCIU1_BUSY 4 4
# 	RCIU2_BUSY 5 5
# 	ROQ1_BUSY 6 6
# 	ROQ2_BUSY 7 7
# 	TCIU_BUSY 10 10
# 	SCRATCH_RAM_BUSY 11 11
# 	QU_BUSY 12 12
# 	UTCL2IU_BUSY 13 13
# 	SAVE_RESTORE_BUSY 14 14
# 	GCRIU_BUSY 15 15
# 	MES_BUSY 16 16
# 	MES_SCRATCH_RAM_BUSY 17 17
# 	RCIU3_BUSY 18 18
# 	MES_INSTRUCTION_CACHE_BUSY 19 19
# 	MES_DATA_CACHE_BUSY 20 20
# 	MEC_DATA_CACHE_BUSY 21 21
# 	CPG_CPC_BUSY 29 29
# 	CPF_CPC_BUSY 30 30
# 	CPC_BUSY 31 31
regCP_CPF_BUSY_HYSTERESIS1 = 0x1edc
# 	CAC_ACTIVE 0 7
# 	CPF_BUSY 8 15
# 	CORE_BUSY 16 23
# 	GFX_BUSY 24 31
regCP_CPF_BUSY_HYSTERESIS2 = 0x1edd
# 	CMP_BUSY 0 7
regCP_CPF_BUSY_STAT = 0xe28
# 	REG_BUS_FIFO_BUSY 0 0
# 	CSF_RING_BUSY 1 1
# 	CSF_INDIRECT1_BUSY 2 2
# 	CSF_INDIRECT2_BUSY 3 3
# 	CSF_STATE_BUSY 4 4
# 	CSF_CE_INDR1_BUSY 5 5
# 	CSF_CE_INDR2_BUSY 6 6
# 	CSF_ARBITER_BUSY 7 7
# 	CSF_INPUT_BUSY 8 8
# 	CSF_DATA_BUSY 9 9
# 	CSF_CE_DATA_BUSY 10 10
# 	HPD_PROCESSING_EOP_BUSY 11 11
# 	HQD_DISPATCH_BUSY 12 12
# 	HQD_IQ_TIMER_BUSY 13 13
# 	HQD_DMA_OFFLOAD_BUSY 14 14
# 	HQD_WAIT_SEMAPHORE_BUSY 15 15
# 	HQD_SIGNAL_SEMAPHORE_BUSY 16 16
# 	HQD_MESSAGE_BUSY 17 17
# 	HQD_PQ_FETCHER_BUSY 18 18
# 	HQD_IB_FETCHER_BUSY 19 19
# 	HQD_IQ_FETCHER_BUSY 20 20
# 	HQD_EOP_FETCHER_BUSY 21 21
# 	HQD_CONSUMED_RPTR_BUSY 22 22
# 	HQD_FETCHER_ARB_BUSY 23 23
# 	HQD_ROQ_ALIGN_BUSY 24 24
# 	HQD_ROQ_EOP_BUSY 25 25
# 	HQD_ROQ_IQ_BUSY 26 26
# 	HQD_ROQ_PQ_BUSY 27 27
# 	HQD_ROQ_IB_BUSY 28 28
# 	HQD_WPTR_POLL_BUSY 29 29
# 	HQD_PQ_BUSY 30 30
# 	HQD_IB_BUSY 31 31
regCP_CPF_BUSY_STAT2 = 0xe33
# 	CP_SDMA_CPG_BUSY 0 0
# 	CP_SDMA_CPC_BUSY 1 1
# 	MES_HQD_DISPATCH_BUSY 12 12
# 	MES_HQD_DMA_OFFLOAD_BUSY 14 14
# 	MES_HQD_MESSAGE_BUSY 17 17
# 	MES_HQD_PQ_FETCHER_BUSY 18 18
# 	MES_HQD_CONSUMED_RPTR_BUSY 22 22
# 	MES_HQD_FETCHER_ARB_BUSY 23 23
# 	MES_HQD_ROQ_ALIGN_BUSY 24 24
# 	MES_HQD_ROQ_PQ_BUSY 27 27
# 	MES_HQD_PQ_BUSY 30 30
regCP_CPF_GRBM_FREE_COUNT = 0xe32
# 	FREE_COUNT 0 2
regCP_CPF_STALLED_STAT1 = 0xe29
# 	RING_FETCHING_DATA 0 0
# 	INDR1_FETCHING_DATA 1 1
# 	INDR2_FETCHING_DATA 2 2
# 	STATE_FETCHING_DATA 3 3
# 	TCIU_WAITING_ON_FREE 5 5
# 	TCIU_WAITING_ON_TAGS 6 6
# 	UTCL2IU_WAITING_ON_FREE 7 7
# 	UTCL2IU_WAITING_ON_TAGS 8 8
# 	GFX_UTCL1_WAITING_ON_TRANS 9 9
# 	CMP_UTCL1_WAITING_ON_TRANS 10 10
# 	RCIU_WAITING_ON_FREE 11 11
# 	DATA_FETCHING_DATA 12 12
# 	GCRIU_WAIT_ON_FREE 13 13
regCP_CPF_STATUS = 0xe27
# 	POST_WPTR_GFX_BUSY 0 0
# 	CSF_BUSY 1 1
# 	ROQ_ALIGN_BUSY 4 4
# 	ROQ_RING_BUSY 5 5
# 	ROQ_INDIRECT1_BUSY 6 6
# 	ROQ_INDIRECT2_BUSY 7 7
# 	ROQ_STATE_BUSY 8 8
# 	ROQ_CE_RING_BUSY 9 9
# 	ROQ_CE_INDIRECT1_BUSY 10 10
# 	ROQ_CE_INDIRECT2_BUSY 11 11
# 	SEMAPHORE_BUSY 12 12
# 	INTERRUPT_BUSY 13 13
# 	TCIU_BUSY 14 14
# 	HQD_BUSY 15 15
# 	PRT_BUSY 16 16
# 	UTCL2IU_BUSY 17 17
# 	RCIU_BUSY 18 18
# 	RCIU_GFX_BUSY 19 19
# 	RCIU_CMP_BUSY 20 20
# 	ROQ_DATA_BUSY 21 21
# 	ROQ_CE_DATA_BUSY 22 22
# 	GCRIU_BUSY 23 23
# 	MES_HQD_BUSY 24 24
# 	CPF_GFX_BUSY 26 26
# 	CPF_CMP_BUSY 27 27
# 	GRBM_CPF_STAT_BUSY 28 29
# 	CPC_CPF_BUSY 30 30
# 	CPF_BUSY 31 31
regCP_CPG_BUSY_HYSTERESIS1 = 0x1ede
# 	CAC_ACTIVE 0 7
# 	CP_BUSY 8 15
# 	DMA_BUSY 16 23
# 	GFX_BUSY 24 31
regCP_CPG_BUSY_HYSTERESIS2 = 0x1edf
# 	CMP_BUSY 0 7
# 	SPI_CLOCK_0 8 15
# 	SPI_CLOCK_1 16 23
regCP_CSF_STAT = 0xf54
# 	BUFFER_REQUEST_COUNT 8 16
regCP_CU_MASK_ADDR_HI = 0x1dd3
# 	ADDR_HI 0 31
regCP_CU_MASK_ADDR_LO = 0x1dd2
# 	ADDR_LO 2 31
regCP_CU_MASK_CNTL = 0x1dd4
# 	POLICY 0 0
regCP_DB_BASE_HI = 0x20d9
# 	DB_BASE_HI 0 15
regCP_DB_BASE_LO = 0x20d8
# 	DB_BASE_LO 2 31
regCP_DB_BUFSZ = 0x20da
# 	DB_BUFSZ 0 19
regCP_DB_CMD_BUFSZ = 0x20db
# 	DB_CMD_REQSZ 0 19
regCP_DDID_BASE_ADDR_HI = 0x1e6c
# 	BASE_ADDR_HI 0 15
regCP_DDID_BASE_ADDR_LO = 0x1e6b
# 	BASE_ADDR_LO 6 31
regCP_DDID_CNTL = 0x1e6d
# 	THRESHOLD 0 7
# 	SIZE 16 16
# 	NO_RING_MEMORY 19 19
# 	VMID 20 23
# 	VMID_SEL 24 24
# 	POLICY 28 29
# 	MODE 30 30
# 	ENABLE 31 31
regCP_DEBUG = 0x1e1f
# 	PERFMON_RING_SEL 0 1
# 	DEBUG_BUS_SELECT_BITS 2 7
# 	DEBUG_BUS_FLOP_EN 8 8
# 	CPG_REPEATER_FGCG_OVERRIDE 9 9
# 	PACKET_FILTER_DISABLE 10 10
# 	NOT_EOP_PREEMPT_DISABLE 11 11
# 	CPG_CHIU_RO_DISABLE 12 12
# 	CPG_GCR_CNTL_BYPASS 13 13
# 	CPG_RAM_CLK_GATING_DISABLE 14 14
# 	CPG_UTCL1_ERROR_HALT_DISABLE 15 15
# 	SURFSYNC_CNTX_RDADDR 16 18
# 	PRIV_VIOLATION_WRITE_DISABLE 20 20
# 	CPG_CHIU_GUS_DISABLE 21 21
# 	INTERRUPT_DISABLE 22 22
# 	PREDICATE_DISABLE 23 23
# 	UNDERFLOW_BUSY_DISABLE 24 24
# 	OVERFLOW_BUSY_DISABLE 25 25
# 	EVENT_FILT_DISABLE 26 26
# 	CPG_CHIU_MTYPE_OVERRIDE 27 27
# 	CPG_TC_ONE_CYCLE_WRITE_DISABLE 28 28
# 	CS_STATE_FILT_DISABLE 29 29
# 	CS_PIPELINE_RESET_DISABLE 30 30
# 	IB_PACKET_INJECTOR_DISABLE 31 31
regCP_DEBUG_2 = 0x1800
# 	CHIU_NOALLOC_OVERRIDE 12 12
# 	RCIU_SECURE_CHECK_DISABLE 13 13
# 	RB_PACKET_INJECTOR_DISABLE 14 14
# 	CNTX_DONE_COPY_STATE_DISABLE 15 15
# 	NOP_DISCARD_DISABLE 16 16
# 	DC_INTERLEAVE_DISABLE 17 17
# 	BC_LOOKUP_CB_DB_FLUSH_DISABLE 27 27
# 	DC_FORCE_CLK_EN 28 28
# 	DC_DISABLE_BROADCAST 29 29
# 	NOT_EOP_HW_DETECT_DISABLE 30 30
# 	PFP_DDID_HW_DETECT_DISABLE 31 31
regCP_DEBUG_CNTL = 0xf98
# 	DEBUG_INDX 0 6
regCP_DEBUG_DATA = 0xf99
# 	DEBUG_DATA 0 31
regCP_DEVICE_ID = 0x1deb
# 	DEVICE_ID 0 7
regCP_DISPATCH_INDR_ADDR = 0x20f6
# 	ADDR_LO 0 31
regCP_DISPATCH_INDR_ADDR_HI = 0x20f7
# 	ADDR_HI 0 15
regCP_DMA_CNTL = 0x208a
# 	UTCL1_FAULT_CONTROL 0 0
# 	WATCH_CONTROL 1 1
# 	MIN_AVAILSZ 4 5
# 	BUFFER_DEPTH 16 24
# 	PIO_FIFO_EMPTY 28 28
# 	PIO_FIFO_FULL 29 29
# 	PIO_COUNT 30 31
regCP_DMA_ME_CMD_ADDR_HI = 0x209d
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_ME_CMD_ADDR_LO = 0x209c
# 	RSVD 0 1
# 	ADDR_LO 2 31
regCP_DMA_ME_COMMAND = 0x2084
# 	BYTE_COUNT 0 25
# 	SAS 26 26
# 	DAS 27 27
# 	SAIC 28 28
# 	DAIC 29 29
# 	RAW_WAIT 30 30
# 	DIS_WC 31 31
regCP_DMA_ME_CONTROL = 0x2078
# 	VMID 0 3
# 	TMZ 4 4
# 	MEMLOG_CLEAR 10 10
# 	SRC_CACHE_POLICY 13 14
# 	SRC_VOLATLE 15 15
# 	DST_SELECT 20 21
# 	DST_CACHE_POLICY 25 26
# 	DST_VOLATLE 27 27
# 	SRC_SELECT 29 30
regCP_DMA_ME_DST_ADDR = 0x2082
# 	DST_ADDR 0 31
regCP_DMA_ME_DST_ADDR_HI = 0x2083
# 	DST_ADDR_HI 0 15
regCP_DMA_ME_SRC_ADDR = 0x2080
# 	SRC_ADDR 0 31
regCP_DMA_ME_SRC_ADDR_HI = 0x2081
# 	SRC_ADDR_HI 0 15
regCP_DMA_PFP_CMD_ADDR_HI = 0x209f
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_PFP_CMD_ADDR_LO = 0x209e
# 	RSVD 0 1
# 	ADDR_LO 2 31
regCP_DMA_PFP_COMMAND = 0x2089
# 	BYTE_COUNT 0 25
# 	SAS 26 26
# 	DAS 27 27
# 	SAIC 28 28
# 	DAIC 29 29
# 	RAW_WAIT 30 30
# 	DIS_WC 31 31
regCP_DMA_PFP_CONTROL = 0x2077
# 	VMID 0 3
# 	TMZ 4 4
# 	MEMLOG_CLEAR 10 10
# 	SRC_CACHE_POLICY 13 14
# 	SRC_VOLATLE 15 15
# 	DST_SELECT 20 21
# 	DST_CACHE_POLICY 25 26
# 	DST_VOLATLE 27 27
# 	SRC_SELECT 29 30
regCP_DMA_PFP_DST_ADDR = 0x2087
# 	DST_ADDR 0 31
regCP_DMA_PFP_DST_ADDR_HI = 0x2088
# 	DST_ADDR_HI 0 15
regCP_DMA_PFP_SRC_ADDR = 0x2085
# 	SRC_ADDR 0 31
regCP_DMA_PFP_SRC_ADDR_HI = 0x2086
# 	SRC_ADDR_HI 0 15
regCP_DMA_READ_TAGS = 0x208b
# 	DMA_READ_TAG 0 25
# 	DMA_READ_TAG_VALID 28 28
regCP_DMA_WATCH0_ADDR_HI = 0x1ec1
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_WATCH0_ADDR_LO = 0x1ec0
# 	RSVD 0 6
# 	ADDR_LO 7 31
regCP_DMA_WATCH0_CNTL = 0x1ec3
# 	VMID 0 3
# 	RSVD1 4 7
# 	WATCH_READS 8 8
# 	WATCH_WRITES 9 9
# 	ANY_VMID 10 10
# 	RSVD2 11 31
regCP_DMA_WATCH0_MASK = 0x1ec2
# 	RSVD 0 6
# 	MASK 7 31
regCP_DMA_WATCH1_ADDR_HI = 0x1ec5
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_WATCH1_ADDR_LO = 0x1ec4
# 	RSVD 0 6
# 	ADDR_LO 7 31
regCP_DMA_WATCH1_CNTL = 0x1ec7
# 	VMID 0 3
# 	RSVD1 4 7
# 	WATCH_READS 8 8
# 	WATCH_WRITES 9 9
# 	ANY_VMID 10 10
# 	RSVD2 11 31
regCP_DMA_WATCH1_MASK = 0x1ec6
# 	RSVD 0 6
# 	MASK 7 31
regCP_DMA_WATCH2_ADDR_HI = 0x1ec9
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_WATCH2_ADDR_LO = 0x1ec8
# 	RSVD 0 6
# 	ADDR_LO 7 31
regCP_DMA_WATCH2_CNTL = 0x1ecb
# 	VMID 0 3
# 	RSVD1 4 7
# 	WATCH_READS 8 8
# 	WATCH_WRITES 9 9
# 	ANY_VMID 10 10
# 	RSVD2 11 31
regCP_DMA_WATCH2_MASK = 0x1eca
# 	RSVD 0 6
# 	MASK 7 31
regCP_DMA_WATCH3_ADDR_HI = 0x1ecd
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_DMA_WATCH3_ADDR_LO = 0x1ecc
# 	RSVD 0 6
# 	ADDR_LO 7 31
regCP_DMA_WATCH3_CNTL = 0x1ecf
# 	VMID 0 3
# 	RSVD1 4 7
# 	WATCH_READS 8 8
# 	WATCH_WRITES 9 9
# 	ANY_VMID 10 10
# 	RSVD2 11 31
regCP_DMA_WATCH3_MASK = 0x1ece
# 	RSVD 0 6
# 	MASK 7 31
regCP_DMA_WATCH_STAT = 0x1ed2
# 	VMID 0 3
# 	QUEUE_ID 4 6
# 	CLIENT_ID 8 10
# 	PIPE 12 13
# 	WATCH_ID 16 17
# 	RD_WR 20 20
# 	TRAP_FLAG 31 31
regCP_DMA_WATCH_STAT_ADDR_HI = 0x1ed1
# 	ADDR_HI 0 15
regCP_DMA_WATCH_STAT_ADDR_LO = 0x1ed0
# 	ADDR_LO 2 31
regCP_DRAW_INDX_INDR_ADDR = 0x20f4
# 	ADDR_LO 0 31
regCP_DRAW_INDX_INDR_ADDR_HI = 0x20f5
# 	ADDR_HI 0 15
regCP_DRAW_OBJECT = 0x3810
# 	OBJECT 0 31
regCP_DRAW_OBJECT_COUNTER = 0x3811
# 	COUNT 0 15
regCP_DRAW_WINDOW_CNTL = 0x3815
# 	DISABLE_DRAW_WINDOW_LO_MAX 0 0
# 	DISABLE_DRAW_WINDOW_LO_MIN 1 1
# 	DISABLE_DRAW_WINDOW_HI 2 2
# 	MODE 8 8
regCP_DRAW_WINDOW_HI = 0x3813
# 	WINDOW_HI 0 31
regCP_DRAW_WINDOW_LO = 0x3814
# 	MIN 0 15
# 	MAX 16 31
regCP_DRAW_WINDOW_MASK_HI = 0x3812
# 	WINDOW_MASK_HI 0 31
regCP_ECC_FIRSTOCCURRENCE = 0x1e1a
# 	INTERFACE 0 1
# 	CLIENT 4 7
# 	ME 8 9
# 	PIPE 10 11
# 	VMID 16 19
regCP_ECC_FIRSTOCCURRENCE_RING0 = 0x1e1b
# 	OBSOLETE 0 31
regCP_ECC_FIRSTOCCURRENCE_RING1 = 0x1e1c
# 	OBSOLETE 0 31
regCP_EOPQ_WAIT_TIME = 0x1dd5
# 	WAIT_TIME 0 9
# 	SCALE_COUNT 10 17
regCP_EOP_DONE_ADDR_HI = 0x2001
# 	ADDR_HI 0 15
regCP_EOP_DONE_ADDR_LO = 0x2000
# 	ADDR_LO 2 31
regCP_EOP_DONE_CNTX_ID = 0x20d7
# 	CNTX_ID 0 31
regCP_EOP_DONE_DATA_CNTL = 0x20d6
# 	DST_SEL 16 17
# 	SEMAPHORE_SIGNAL_TYPE 19 19
# 	ACTION_PIPE_ID 20 21
# 	ACTION_ID 22 23
# 	INT_SEL 24 26
# 	DATA_SEL 29 31
regCP_EOP_DONE_DATA_HI = 0x2003
# 	DATA_HI 0 31
regCP_EOP_DONE_DATA_LO = 0x2002
# 	DATA_LO 0 31
regCP_EOP_DONE_EVENT_CNTL = 0x20d5
# 	GCR_CNTL 12 24
# 	CACHE_POLICY 25 26
# 	EOP_VOLATILE 27 27
# 	EXECUTE 28 28
# 	GLK_INV 30 30
# 	PWS_ENABLE 31 31
regCP_EOP_LAST_FENCE_HI = 0x2005
# 	LAST_FENCE_HI 0 31
regCP_EOP_LAST_FENCE_LO = 0x2004
# 	LAST_FENCE_LO 0 31
regCP_FATAL_ERROR = 0x1df0
# 	CPF_FATAL_ERROR 0 0
# 	CPG_FATAL_ERROR 1 1
# 	GFX_HALT_PROC 2 2
# 	DIS_CPG_FATAL_ERROR 3 3
# 	CPG_TAG_FATAL_ERROR_EN 4 4
regCP_FETCHER_SOURCE = 0x1801
# 	ME_SRC 0 0
regCP_GDS_ATOMIC0_PREOP_HI = 0x2060
# 	GDS_ATOMIC0_PREOP_HI 0 31
regCP_GDS_ATOMIC0_PREOP_LO = 0x205f
# 	GDS_ATOMIC0_PREOP_LO 0 31
regCP_GDS_ATOMIC1_PREOP_HI = 0x2062
# 	GDS_ATOMIC1_PREOP_HI 0 31
regCP_GDS_ATOMIC1_PREOP_LO = 0x2061
# 	GDS_ATOMIC1_PREOP_LO 0 31
regCP_GDS_BKUP_ADDR = 0x20fb
# 	ADDR_LO 0 31
regCP_GDS_BKUP_ADDR_HI = 0x20fc
# 	ADDR_HI 0 15
regCP_GE_MSINVOC_COUNT_HI = 0x20a7
# 	MSINVOC_COUNT_HI 0 31
regCP_GE_MSINVOC_COUNT_LO = 0x20a6
# 	MSINVOC_COUNT_LO 0 31
regCP_GFX_CNTL = 0x2a00
# 	ENGINE_SEL 0 0
# 	CONFIG 1 2
regCP_GFX_DDID_DELTA_RPT_COUNT = 0x1e71
# 	COUNT 0 7
regCP_GFX_DDID_INFLIGHT_COUNT = 0x1e6e
# 	COUNT 0 15
regCP_GFX_DDID_RPTR = 0x1e70
# 	COUNT 0 15
regCP_GFX_DDID_WPTR = 0x1e6f
# 	COUNT 0 15
regCP_GFX_ERROR = 0x1ddb
# 	ME_INSTR_CACHE_UTCL1_ERROR 0 0
# 	PFP_INSTR_CACHE_UTCL1_ERROR 1 1
# 	DDID_DRAW_UTCL1_ERROR 2 2
# 	DDID_DISPATCH_UTCL1_ERROR 3 3
# 	SUA_ERROR 4 4
# 	DATA_FETCHER_UTCL1_ERROR 6 6
# 	SEM_UTCL1_ERROR 7 7
# 	QU_EOP_UTCL1_ERROR 9 9
# 	QU_PIPE_UTCL1_ERROR 10 10
# 	QU_READ_UTCL1_ERROR 11 11
# 	SYNC_MEMRD_UTCL1_ERROR 12 12
# 	SYNC_MEMWR_UTCL1_ERROR 13 13
# 	SHADOW_UTCL1_ERROR 14 14
# 	APPEND_UTCL1_ERROR 15 15
# 	DMA_SRC_UTCL1_ERROR 18 18
# 	DMA_DST_UTCL1_ERROR 19 19
# 	PFP_TC_UTCL1_ERROR 20 20
# 	ME_TC_UTCL1_ERROR 21 21
# 	PRT_LOD_UTCL1_ERROR 23 23
# 	RDPTR_RPT_UTCL1_ERROR 24 24
# 	RB_FETCHER_UTCL1_ERROR 25 25
# 	I1_FETCHER_UTCL1_ERROR 26 26
# 	I2_FETCHER_UTCL1_ERROR 27 27
# 	ST_FETCHER_UTCL1_ERROR 30 30
# 	RESERVED 31 31
regCP_GFX_HPD_CONTROL0 = 0x1e73
# 	SUSPEND_ENABLE 0 0
# 	PIPE_HOLDING 4 4
# 	RB_CE_ROQ_CNTL 8 8
regCP_GFX_HPD_OSPRE_FENCE_ADDR_HI = 0x1e75
# 	ADDR_HI 0 15
# 	RSVD 16 31
regCP_GFX_HPD_OSPRE_FENCE_ADDR_LO = 0x1e74
# 	ADDR_LO 2 31
regCP_GFX_HPD_OSPRE_FENCE_DATA_HI = 0x1e77
# 	DATA_HI 0 31
regCP_GFX_HPD_OSPRE_FENCE_DATA_LO = 0x1e76
# 	DATA_LO 0 31
regCP_GFX_HPD_STATUS0 = 0x1e72
# 	QUEUE_STATE 0 4
# 	MAPPED_QUEUE 5 7
# 	QUEUE_AVAILABLE 8 15
# 	FORCE_MAPPED_QUEUE 16 18
# 	FORCE_QUEUE_STATE 20 24
# 	SUSPEND_REQ 28 28
# 	ENABLE_OVERIDE_QUEUEID 29 29
# 	OVERIDE_QUEUEID 30 30
# 	FORCE_QUEUE 31 31
regCP_GFX_HQD_ACTIVE = 0x1e80
# 	ACTIVE 0 0
regCP_GFX_HQD_BASE = 0x1e86
# 	RB_BASE 0 31
regCP_GFX_HQD_BASE_HI = 0x1e87
# 	RB_BASE_HI 0 7
regCP_GFX_HQD_CNTL = 0x1e8f
# 	RB_BUFSZ 0 5
# 	TMZ_STATE 6 6
# 	TMZ_MATCH 7 7
# 	RB_BLKSZ 8 13
# 	RB_NON_PRIV 15 15
# 	BUF_SWAP 16 17
# 	MIN_AVAILSZ 20 21
# 	MIN_IB_AVAILSZ 22 23
# 	CACHE_POLICY 24 25
# 	RB_VOLATILE 26 26
# 	RB_NO_UPDATE 27 27
# 	RB_EXE 28 28
# 	KMD_QUEUE 29 29
# 	RB_RPTR_WR_ENA 31 31
regCP_GFX_HQD_CSMD_RPTR = 0x1e90
# 	RB_RPTR 0 19
regCP_GFX_HQD_DEQUEUE_REQUEST = 0x1e93
# 	DEQUEUE_REQ 0 0
# 	IQ_REQ_PEND 4 4
# 	IQ_REQ_PEND_EN 9 9
# 	DEQUEUE_REQ_EN 10 10
regCP_GFX_HQD_HQ_CONTROL0 = 0x1e99
# 	COMMAND 0 3
# 	SPARES 4 7
regCP_GFX_HQD_HQ_STATUS0 = 0x1e98
# 	DEQUEUE_STATUS 0 0
# 	OS_PREEMPT_STATUS 4 5
# 	PREEMPT_ACK 6 6
# 	QUEUE_IDLE 30 30
regCP_GFX_HQD_IQ_TIMER = 0x1e96
# 	WAIT_TIME 0 7
# 	RETRY_TYPE 8 10
# 	IMMEDIATE_EXPIRE 11 11
# 	INTERRUPT_TYPE 12 13
# 	CLOCK_COUNT 14 15
# 	QUANTUM_TIMER 22 22
# 	QUEUE_TYPE 27 27
# 	REARM_TIMER 28 28
# 	ACTIVE 31 31
regCP_GFX_HQD_MAPPED = 0x1e94
# 	MAPPED 0 0
regCP_GFX_HQD_OFFSET = 0x1e8e
# 	RB_OFFSET 0 19
# 	DISABLE_RB_OFFSET 31 31
regCP_GFX_HQD_QUANTUM = 0x1e85
# 	QUANTUM_EN 0 0
# 	QUANTUM_SCALE 3 4
# 	QUANTUM_DURATION 8 15
# 	QUANTUM_ACTIVE 31 31
regCP_GFX_HQD_QUEUE_PRIORITY = 0x1e84
# 	PRIORITY_LEVEL 0 3
regCP_GFX_HQD_QUE_MGR_CONTROL = 0x1e95
# 	DISABLE_IDLE_QUEUE_DISCONNECT 0 0
# 	DISABLE_CONNECT_HANDSHAKE 4 4
# 	DISABLE_FETCHER_DISCONNECT 5 5
# 	FORCE_QUEUE_ACTIVE_EN 6 6
# 	FORCE_ALLOW_DB_UPDATE_EN 7 7
# 	FORCE_QUEUE 8 10
# 	DISABLE_OFFSET_UPDATE 11 11
# 	PRIORITY_PREEMPT_DISABLE 13 13
# 	DISABLE_QUEUE_MGR 15 15
# 	ENABLE_IDLE_MESSAGE 16 16
# 	DISABLE_SWITCH_MESSAGE_IDLE 17 17
# 	ENABLE_SWITCH_MSG_PREEMPT 18 18
# 	DISABLE_MAPPED_QUEUE_IDLE_MSG 23 23
regCP_GFX_HQD_RPTR = 0x1e88
# 	RB_RPTR 0 19
regCP_GFX_HQD_RPTR_ADDR = 0x1e89
# 	RB_RPTR_ADDR 2 31
regCP_GFX_HQD_RPTR_ADDR_HI = 0x1e8a
# 	RB_RPTR_ADDR_HI 0 15
regCP_GFX_HQD_VMID = 0x1e81
# 	VMID 0 3
regCP_GFX_HQD_WPTR = 0x1e91
# 	RB_WPTR 0 31
regCP_GFX_HQD_WPTR_HI = 0x1e92
# 	RB_WPTR 0 31
regCP_GFX_INDEX_MUTEX = 0x1e78
# 	REQUEST 0 0
# 	CLIENTID 1 3
regCP_GFX_MQD_BASE_ADDR = 0x1e7e
# 	BASE_ADDR 2 31
regCP_GFX_MQD_BASE_ADDR_HI = 0x1e7f
# 	BASE_ADDR_HI 0 15
# 	APP_VMID 28 31
regCP_GFX_MQD_CONTROL = 0x1e9a
# 	VMID 0 3
# 	PRIV_STATE 8 8
# 	PROCESSING_MQD 12 12
# 	PROCESSING_MQD_EN 13 13
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
regCP_GFX_QUEUE_INDEX = 0x1e37
# 	QUEUE_ACCESS 0 0
# 	PIPE_ID 4 5
# 	QUEUE_ID 8 10
regCP_GFX_RS64_DC_APERTURE0_BASE0 = 0x2a49
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE0_BASE1 = 0x2a79
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE0_CNTL0 = 0x2a4b
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE0_CNTL1 = 0x2a7b
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE0_MASK0 = 0x2a4a
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE0_MASK1 = 0x2a7a
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE10_BASE0 = 0x2a67
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE10_BASE1 = 0x2a97
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE10_CNTL0 = 0x2a69
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE10_CNTL1 = 0x2a99
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE10_MASK0 = 0x2a68
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE10_MASK1 = 0x2a98
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE11_BASE0 = 0x2a6a
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE11_BASE1 = 0x2a9a
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE11_CNTL0 = 0x2a6c
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE11_CNTL1 = 0x2a9c
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE11_MASK0 = 0x2a6b
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE11_MASK1 = 0x2a9b
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE12_BASE0 = 0x2a6d
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE12_BASE1 = 0x2a9d
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE12_CNTL0 = 0x2a6f
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE12_CNTL1 = 0x2a9f
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE12_MASK0 = 0x2a6e
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE12_MASK1 = 0x2a9e
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE13_BASE0 = 0x2a70
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE13_BASE1 = 0x2aa0
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE13_CNTL0 = 0x2a72
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE13_CNTL1 = 0x2aa2
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE13_MASK0 = 0x2a71
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE13_MASK1 = 0x2aa1
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE14_BASE0 = 0x2a73
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE14_BASE1 = 0x2aa3
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE14_CNTL0 = 0x2a75
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE14_CNTL1 = 0x2aa5
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE14_MASK0 = 0x2a74
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE14_MASK1 = 0x2aa4
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE15_BASE0 = 0x2a76
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE15_BASE1 = 0x2aa6
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE15_CNTL0 = 0x2a78
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE15_CNTL1 = 0x2aa8
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE15_MASK0 = 0x2a77
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE15_MASK1 = 0x2aa7
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE1_BASE0 = 0x2a4c
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE1_BASE1 = 0x2a7c
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE1_CNTL0 = 0x2a4e
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE1_CNTL1 = 0x2a7e
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE1_MASK0 = 0x2a4d
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE1_MASK1 = 0x2a7d
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE2_BASE0 = 0x2a4f
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE2_BASE1 = 0x2a7f
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE2_CNTL0 = 0x2a51
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE2_CNTL1 = 0x2a81
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE2_MASK0 = 0x2a50
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE2_MASK1 = 0x2a80
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE3_BASE0 = 0x2a52
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE3_BASE1 = 0x2a82
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE3_CNTL0 = 0x2a54
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE3_CNTL1 = 0x2a84
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE3_MASK0 = 0x2a53
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE3_MASK1 = 0x2a83
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE4_BASE0 = 0x2a55
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE4_BASE1 = 0x2a85
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE4_CNTL0 = 0x2a57
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE4_CNTL1 = 0x2a87
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE4_MASK0 = 0x2a56
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE4_MASK1 = 0x2a86
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE5_BASE0 = 0x2a58
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE5_BASE1 = 0x2a88
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE5_CNTL0 = 0x2a5a
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE5_CNTL1 = 0x2a8a
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE5_MASK0 = 0x2a59
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE5_MASK1 = 0x2a89
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE6_BASE0 = 0x2a5b
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE6_BASE1 = 0x2a8b
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE6_CNTL0 = 0x2a5d
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE6_CNTL1 = 0x2a8d
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE6_MASK0 = 0x2a5c
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE6_MASK1 = 0x2a8c
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE7_BASE0 = 0x2a5e
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE7_BASE1 = 0x2a8e
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE7_CNTL0 = 0x2a60
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE7_CNTL1 = 0x2a90
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE7_MASK0 = 0x2a5f
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE7_MASK1 = 0x2a8f
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE8_BASE0 = 0x2a61
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE8_BASE1 = 0x2a91
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE8_CNTL0 = 0x2a63
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE8_CNTL1 = 0x2a93
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE8_MASK0 = 0x2a62
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE8_MASK1 = 0x2a92
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE9_BASE0 = 0x2a64
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE9_BASE1 = 0x2a94
# 	BASE 0 31
regCP_GFX_RS64_DC_APERTURE9_CNTL0 = 0x2a66
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE9_CNTL1 = 0x2a96
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_GFX_RS64_DC_APERTURE9_MASK0 = 0x2a65
# 	MASK 0 31
regCP_GFX_RS64_DC_APERTURE9_MASK1 = 0x2a95
# 	MASK 0 31
regCP_GFX_RS64_DC_BASE0_HI = 0x5865
# 	DC_BASE_HI 0 15
regCP_GFX_RS64_DC_BASE0_LO = 0x5863
# 	DC_BASE_LO 16 31
regCP_GFX_RS64_DC_BASE1_HI = 0x5866
# 	DC_BASE_HI 0 15
regCP_GFX_RS64_DC_BASE1_LO = 0x5864
# 	DC_BASE_LO 16 31
regCP_GFX_RS64_DC_BASE_CNTL = 0x2a08
# 	VMID 0 3
# 	CACHE_POLICY 24 25
regCP_GFX_RS64_DC_OP_CNTL = 0x2a09
# 	INVALIDATE_DCACHE 0 0
# 	INVALIDATE_DCACHE_COMPLETE 1 1
# 	BYPASS_ALL 2 2
# 	RESERVED 3 3
# 	PRIME_DCACHE 4 4
# 	DCACHE_PRIMED 5 5
regCP_GFX_RS64_DM_INDEX_ADDR = 0x5c04
# 	ADDR 0 31
regCP_GFX_RS64_DM_INDEX_DATA = 0x5c05
# 	DATA 0 31
regCP_GFX_RS64_GP0_HI0 = 0x2a26
# 	M_RET_ADDR 0 31
regCP_GFX_RS64_GP0_HI1 = 0x2a27
# 	M_RET_ADDR 0 31
regCP_GFX_RS64_GP0_LO0 = 0x2a24
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_GFX_RS64_GP0_LO1 = 0x2a25
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_GFX_RS64_GP1_HI0 = 0x2a2a
# 	RD_WR_SELECT_HI 0 31
regCP_GFX_RS64_GP1_HI1 = 0x2a2b
# 	RD_WR_SELECT_HI 0 31
regCP_GFX_RS64_GP1_LO0 = 0x2a28
# 	RD_WR_SELECT_LO 0 31
regCP_GFX_RS64_GP1_LO1 = 0x2a29
# 	RD_WR_SELECT_LO 0 31
regCP_GFX_RS64_GP2_HI0 = 0x2a2e
# 	STACK_PNTR_HI 0 31
regCP_GFX_RS64_GP2_HI1 = 0x2a2f
# 	STACK_PNTR_HI 0 31
regCP_GFX_RS64_GP2_LO0 = 0x2a2c
# 	STACK_PNTR_LO 0 31
regCP_GFX_RS64_GP2_LO1 = 0x2a2d
# 	STACK_PNTR_LO 0 31
regCP_GFX_RS64_GP3_HI0 = 0x2a32
# 	DATA 0 31
regCP_GFX_RS64_GP3_HI1 = 0x2a33
# 	DATA 0 31
regCP_GFX_RS64_GP3_LO0 = 0x2a30
# 	DATA 0 31
regCP_GFX_RS64_GP3_LO1 = 0x2a31
# 	DATA 0 31
regCP_GFX_RS64_GP4_HI0 = 0x2a36
# 	DATA 0 31
regCP_GFX_RS64_GP4_HI1 = 0x2a37
# 	DATA 0 31
regCP_GFX_RS64_GP4_LO0 = 0x2a34
# 	DATA 0 31
regCP_GFX_RS64_GP4_LO1 = 0x2a35
# 	DATA 0 31
regCP_GFX_RS64_GP5_HI0 = 0x2a3a
# 	M_RET_ADDR 0 31
regCP_GFX_RS64_GP5_HI1 = 0x2a3b
# 	M_RET_ADDR 0 31
regCP_GFX_RS64_GP5_LO0 = 0x2a38
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_GFX_RS64_GP5_LO1 = 0x2a39
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_GFX_RS64_GP6_HI = 0x2a3d
# 	RD_WR_SELECT_HI 0 31
regCP_GFX_RS64_GP6_LO = 0x2a3c
# 	RD_WR_SELECT_LO 0 31
regCP_GFX_RS64_GP7_HI = 0x2a3f
# 	STACK_PNTR_HI 0 31
regCP_GFX_RS64_GP7_LO = 0x2a3e
# 	STACK_PNTR_LO 0 31
regCP_GFX_RS64_GP8_HI = 0x2a41
# 	DATA 0 31
regCP_GFX_RS64_GP8_LO = 0x2a40
# 	DATA 0 31
regCP_GFX_RS64_GP9_HI = 0x2a43
# 	DATA 0 31
regCP_GFX_RS64_GP9_LO = 0x2a42
# 	DATA 0 31
regCP_GFX_RS64_INSTR_PNTR0 = 0x2a44
# 	INSTR_PNTR 0 19
regCP_GFX_RS64_INSTR_PNTR1 = 0x2a45
# 	INSTR_PNTR 0 19
regCP_GFX_RS64_INTERRUPT0 = 0x2a01
# 	ME_INT 0 31
regCP_GFX_RS64_INTERRUPT1 = 0x2aac
# 	ME_INT 0 31
regCP_GFX_RS64_INTR_EN0 = 0x2a02
# 	ME_INT 0 31
regCP_GFX_RS64_INTR_EN1 = 0x2a03
# 	ME_INT 0 31
regCP_GFX_RS64_LOCAL_APERTURE = 0x2a0e
# 	APERTURE 0 2
regCP_GFX_RS64_LOCAL_BASE0_HI = 0x2a0b
# 	BASE0_HI 0 15
regCP_GFX_RS64_LOCAL_BASE0_LO = 0x2a0a
# 	BASE0_LO 16 31
regCP_GFX_RS64_LOCAL_INSTR_APERTURE = 0x2a13
# 	APERTURE 0 2
regCP_GFX_RS64_LOCAL_INSTR_BASE_HI = 0x2a10
# 	BASE_HI 0 15
regCP_GFX_RS64_LOCAL_INSTR_BASE_LO = 0x2a0f
# 	BASE_LO 16 31
regCP_GFX_RS64_LOCAL_INSTR_MASK_HI = 0x2a12
# 	MASK_HI 0 15
regCP_GFX_RS64_LOCAL_INSTR_MASK_LO = 0x2a11
# 	MASK_LO 16 31
regCP_GFX_RS64_LOCAL_MASK0_HI = 0x2a0d
# 	MASK0_HI 0 15
regCP_GFX_RS64_LOCAL_MASK0_LO = 0x2a0c
# 	MASK0_LO 16 31
regCP_GFX_RS64_LOCAL_SCRATCH_APERTURE = 0x2a14
# 	APERTURE 0 2
regCP_GFX_RS64_LOCAL_SCRATCH_BASE_HI = 0x2a16
# 	BASE_HI 0 15
regCP_GFX_RS64_LOCAL_SCRATCH_BASE_LO = 0x2a15
# 	BASE_LO 16 31
regCP_GFX_RS64_MIBOUND_HI = 0x586d
# 	BOUND 0 31
regCP_GFX_RS64_MIBOUND_LO = 0x586c
# 	BOUND 0 31
regCP_GFX_RS64_MIP_HI0 = 0x2a1e
# 	MIP_HI 0 31
regCP_GFX_RS64_MIP_HI1 = 0x2a1f
# 	MIP_HI 0 31
regCP_GFX_RS64_MIP_LO0 = 0x2a1c
# 	MIP_LO 0 31
regCP_GFX_RS64_MIP_LO1 = 0x2a1d
# 	MIP_LO 0 31
regCP_GFX_RS64_MTIMECMP_HI0 = 0x2a22
# 	TIME_HI 0 31
regCP_GFX_RS64_MTIMECMP_HI1 = 0x2a23
# 	TIME_HI 0 31
regCP_GFX_RS64_MTIMECMP_LO0 = 0x2a20
# 	TIME_LO 0 31
regCP_GFX_RS64_MTIMECMP_LO1 = 0x2a21
# 	TIME_LO 0 31
regCP_GFX_RS64_PENDING_INTERRUPT0 = 0x2a46
# 	PENDING_INTERRUPT 0 31
regCP_GFX_RS64_PENDING_INTERRUPT1 = 0x2a47
# 	PENDING_INTERRUPT 0 31
regCP_GFX_RS64_PERFCOUNT_CNTL0 = 0x2a1a
# 	EVENT_SEL 0 4
regCP_GFX_RS64_PERFCOUNT_CNTL1 = 0x2a1b
# 	EVENT_SEL 0 4
regCP_GPU_TIMESTAMP_OFFSET_HI = 0x1f4d
# 	OFFSET_HI 0 31
regCP_GPU_TIMESTAMP_OFFSET_LO = 0x1f4c
# 	OFFSET_LO 0 31
regCP_GRBM_FREE_COUNT = 0xf43
# 	FREE_COUNT 0 5
# 	FREE_COUNT_GDS 8 13
# 	FREE_COUNT_PFP 16 21
regCP_HPD_MES_ROQ_OFFSETS = 0x1821
# 	IQ_OFFSET 0 2
# 	PQ_OFFSET 8 13
# 	IB_OFFSET 16 22
regCP_HPD_ROQ_OFFSETS = 0x1821
# 	IQ_OFFSET 0 2
# 	PQ_OFFSET 8 13
# 	IB_OFFSET 16 22
regCP_HPD_STATUS0 = 0x1822
# 	QUEUE_STATE 0 4
# 	MAPPED_QUEUE 5 7
# 	QUEUE_AVAILABLE 8 15
# 	FETCHING_MQD 16 16
# 	PEND_TXFER_SIZE_PQIB 17 17
# 	PEND_TXFER_SIZE_IQ 18 18
# 	FORCE_QUEUE_STATE 20 24
# 	MASTER_QUEUE_IDLE_DIS 27 27
# 	ENABLE_OFFLOAD_CHECK 28 29
# 	FREEZE_QUEUE_STATE 30 30
# 	FORCE_QUEUE 31 31
regCP_HPD_UTCL1_CNTL = 0x1fa3
# 	SELECT 0 3
# 	DISABLE_ERROR_REPORT 10 10
regCP_HPD_UTCL1_ERROR = 0x1fa7
# 	ADDR_HI 0 15
# 	TYPE 16 16
# 	VMID 20 23
regCP_HPD_UTCL1_ERROR_ADDR = 0x1fa8
# 	ADDR 12 31
regCP_HQD_ACTIVE = 0x1fab
# 	ACTIVE 0 0
# 	BUSY_GATE 1 1
regCP_HQD_AQL_CONTROL = 0x1fde
# 	CONTROL0 0 14
# 	CONTROL0_EN 15 15
# 	CONTROL1 16 30
# 	CONTROL1_EN 31 31
regCP_HQD_ATOMIC0_PREOP_HI = 0x1fc6
# 	ATOMIC0_PREOP_HI 0 31
regCP_HQD_ATOMIC0_PREOP_LO = 0x1fc5
# 	ATOMIC0_PREOP_LO 0 31
regCP_HQD_ATOMIC1_PREOP_HI = 0x1fc8
# 	ATOMIC1_PREOP_HI 0 31
regCP_HQD_ATOMIC1_PREOP_LO = 0x1fc7
# 	ATOMIC1_PREOP_LO 0 31
regCP_HQD_CNTL_STACK_OFFSET = 0x1fd7
# 	OFFSET 2 15
regCP_HQD_CNTL_STACK_SIZE = 0x1fd8
# 	SIZE 12 15
regCP_HQD_CTX_SAVE_BASE_ADDR_HI = 0x1fd5
# 	ADDR_HI 0 15
regCP_HQD_CTX_SAVE_BASE_ADDR_LO = 0x1fd4
# 	ADDR 12 31
regCP_HQD_CTX_SAVE_CONTROL = 0x1fd6
# 	POLICY 3 4
# 	EXE_DISABLE 23 23
regCP_HQD_CTX_SAVE_SIZE = 0x1fda
# 	SIZE 12 25
regCP_HQD_DDID_DELTA_RPT_COUNT = 0x1fe7
# 	COUNT 0 7
regCP_HQD_DDID_INFLIGHT_COUNT = 0x1fe6
# 	COUNT 0 15
regCP_HQD_DDID_RPTR = 0x1fe4
# 	RPTR 0 10
regCP_HQD_DDID_WPTR = 0x1fe5
# 	WPTR 0 10
regCP_HQD_DEQUEUE_REQUEST = 0x1fc1
# 	DEQUEUE_REQ 0 3
# 	IQ_REQ_PEND 4 4
# 	DEQUEUE_INT 8 8
# 	IQ_REQ_PEND_EN 9 9
# 	DEQUEUE_REQ_EN 10 10
regCP_HQD_DEQUEUE_STATUS = 0x1fe8
# 	DEQUEUE_STAT 0 3
# 	SUSPEND_REQ_PEND 4 4
# 	SUSPEND_REQ_PEND_EN 9 9
# 	DEQUEUE_STAT_EN 10 10
regCP_HQD_DMA_OFFLOAD = 0x1fc2
# 	DMA_OFFLOAD 0 0
# 	DMA_OFFLOAD_EN 1 1
# 	AQL_OFFLOAD 2 2
# 	AQL_OFFLOAD_EN 3 3
# 	EOP_OFFLOAD 4 4
# 	EOP_OFFLOAD_EN 5 5
regCP_HQD_EOP_BASE_ADDR = 0x1fce
# 	BASE_ADDR 0 31
regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf
# 	BASE_ADDR_HI 0 7
regCP_HQD_EOP_CONTROL = 0x1fd0
# 	EOP_SIZE 0 5
# 	PROCESSING_EOP 8 8
# 	PROCESS_EOP_EN 12 12
# 	PROCESSING_EOPIB 13 13
# 	PROCESS_EOPIB_EN 14 14
# 	HALT_FETCHER 21 21
# 	HALT_FETCHER_EN 22 22
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
# 	EOP_VOLATILE 26 26
# 	SIG_SEM_RESULT 29 30
# 	PEND_SIG_SEM 31 31
regCP_HQD_EOP_EVENTS = 0x1fd3
# 	EVENT_COUNT 0 11
# 	CS_PARTIAL_FLUSH_PEND 16 16
regCP_HQD_EOP_RPTR = 0x1fd1
# 	RPTR 0 12
# 	RESET_FETCHER 28 28
# 	DEQUEUE_PEND 29 29
# 	RPTR_EQ_CSMD_WPTR 30 30
# 	INIT_FETCHER 31 31
regCP_HQD_EOP_WPTR = 0x1fd2
# 	WPTR 0 12
# 	EOP_EMPTY 15 15
# 	EOP_AVAIL 16 28
regCP_HQD_EOP_WPTR_MEM = 0x1fdd
# 	WPTR 0 12
regCP_HQD_ERROR = 0x1fdc
# 	EDC_ERROR_ID 0 3
# 	SUA_ERROR 4 4
# 	AQL_ERROR 5 5
# 	PQ_UTCL1_ERROR 8 8
# 	IB_UTCL1_ERROR 9 9
# 	EOP_UTCL1_ERROR 10 10
# 	IQ_UTCL1_ERROR 11 11
# 	RRPT_UTCL1_ERROR 12 12
# 	WPP_UTCL1_ERROR 13 13
# 	SEM_UTCL1_ERROR 14 14
# 	DMA_SRC_UTCL1_ERROR 15 15
# 	DMA_DST_UTCL1_ERROR 16 16
# 	SR_UTCL1_ERROR 17 17
# 	QU_UTCL1_ERROR 18 18
# 	TC_UTCL1_ERROR 19 19
regCP_HQD_GDS_RESOURCE_STATE = 0x1fdb
# 	OA_REQUIRED 0 0
# 	OA_ACQUIRED 1 1
# 	GWS_SIZE 4 9
# 	GWS_PNTR 12 17
regCP_HQD_GFX_CONTROL = 0x1e9f
# 	MESSAGE 0 3
# 	MISC 4 14
# 	DB_UPDATED_MSG_EN 15 15
regCP_HQD_GFX_STATUS = 0x1ea0
# 	STATUS 0 15
regCP_HQD_HQ_CONTROL0 = 0x1fca
# 	CONTROL 0 31
regCP_HQD_HQ_CONTROL1 = 0x1fcd
# 	CONTROL 0 31
regCP_HQD_HQ_SCHEDULER0 = 0x1fc9
# 	CWSR 0 0
# 	SAVE_STATUS 1 1
# 	RSRV 2 2
# 	STATIC_QUEUE 3 5
# 	QUEUE_RUN_ONCE 6 6
# 	SCRATCH_RAM_INIT 7 7
# 	TCL2_DIRTY 8 8
# 	C_INHERIT_VMID 9 9
# 	QUEUE_SCHEDULER_TYPE 10 12
# 	C_QUEUE_USE_GWS 13 13
# 	QUEUE_SLOT_CONNECTED 15 15
# 	MES_INTERRUPT_ENABLED 20 20
# 	MES_INTERRUPT_PIPE 21 22
# 	CONCURRENT_PROCESS_COUNT 24 27
# 	QUEUE_IDLE 30 30
# 	DB_UPDATED_MSG_EN 31 31
regCP_HQD_HQ_SCHEDULER1 = 0x1fca
# 	SCHEDULER 0 31
regCP_HQD_HQ_STATUS0 = 0x1fc9
# 	CWSR 0 0
# 	SAVE_STATUS 1 1
# 	RSRV 2 2
# 	STATIC_QUEUE 3 5
# 	QUEUE_RUN_ONCE 6 6
# 	SCRATCH_RAM_INIT 7 7
# 	TCL2_DIRTY 8 8
# 	C_INHERIT_VMID 9 9
# 	QUEUE_SCHEDULER_TYPE 10 12
# 	C_QUEUE_USE_GWS 13 13
# 	QUEUE_SLOT_CONNECTED 15 15
# 	MES_INTERRUPT_ENABLED 20 20
# 	MES_INTERRUPT_PIPE 21 22
# 	CONCURRENT_PROCESS_COUNT 24 27
# 	QUEUE_IDLE 30 30
# 	DB_UPDATED_MSG_EN 31 31
regCP_HQD_HQ_STATUS1 = 0x1fcc
# 	STATUS 0 31
regCP_HQD_IB_BASE_ADDR = 0x1fbb
# 	IB_BASE_ADDR 2 31
regCP_HQD_IB_BASE_ADDR_HI = 0x1fbc
# 	IB_BASE_ADDR_HI 0 15
regCP_HQD_IB_CONTROL = 0x1fbe
# 	IB_SIZE 0 19
# 	MIN_IB_AVAIL_SIZE 20 21
# 	IB_EXE_DISABLE 23 23
# 	IB_CACHE_POLICY 24 25
# 	IB_VOLATILE 26 26
# 	PROCESSING_IB 31 31
regCP_HQD_IB_RPTR = 0x1fbd
# 	CONSUMED_OFFSET 0 19
regCP_HQD_IQ_RPTR = 0x1fc0
# 	OFFSET 0 5
regCP_HQD_IQ_TIMER = 0x1fbf
# 	WAIT_TIME 0 7
# 	RETRY_TYPE 8 10
# 	IMMEDIATE_EXPIRE 11 11
# 	INTERRUPT_TYPE 12 13
# 	CLOCK_COUNT 14 15
# 	INTERRUPT_SIZE 16 21
# 	QUANTUM_TIMER 22 22
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
# 	IQ_VOLATILE 26 26
# 	QUEUE_TYPE 27 27
# 	REARM_TIMER 28 28
# 	PROCESS_IQ_EN 29 29
# 	PROCESSING_IQ 30 30
# 	ACTIVE 31 31
regCP_HQD_MSG_TYPE = 0x1fc4
# 	ACTION 0 2
# 	SAVE_STATE 4 6
regCP_HQD_OFFLOAD = 0x1fc2
# 	DMA_OFFLOAD 0 0
# 	DMA_OFFLOAD_EN 1 1
# 	AQL_OFFLOAD 2 2
# 	AQL_OFFLOAD_EN 3 3
# 	EOP_OFFLOAD 4 4
# 	EOP_OFFLOAD_EN 5 5
regCP_HQD_PERSISTENT_STATE = 0x1fad
# 	PRELOAD_REQ 0 0
# 	TMZ_CONNECT_OVERRIDE 1 1
# 	SUSPEND_STATUS 7 7
# 	PRELOAD_SIZE 8 17
# 	TMZ_SWITCH_EXEMPT 18 18
# 	TMZ_MATCH_DIS 19 19
# 	WPP_CLAMP_EN 20 20
# 	WPP_SWITCH_QOS_EN 21 21
# 	IQ_SWITCH_QOS_EN 22 22
# 	IB_SWITCH_QOS_EN 23 23
# 	EOP_SWITCH_QOS_EN 24 24
# 	PQ_SWITCH_QOS_EN 25 25
# 	TC_OFFLOAD_QOS_EN 26 26
# 	CACHE_FULL_PACKET_EN 27 27
# 	RESTORE_ACTIVE 28 28
# 	RELAUNCH_WAVES 29 29
# 	QSWITCH_MODE 30 30
# 	DISP_ACTIVE 31 31
regCP_HQD_PIPE_PRIORITY = 0x1fae
# 	PIPE_PRIORITY 0 1
regCP_HQD_PQ_BASE = 0x1fb1
# 	ADDR 0 31
regCP_HQD_PQ_BASE_HI = 0x1fb2
# 	ADDR_HI 0 7
regCP_HQD_PQ_CONTROL = 0x1fba
# 	QUEUE_SIZE 0 5
# 	WPTR_CARRY 6 6
# 	RPTR_CARRY 7 7
# 	RPTR_BLOCK_SIZE 8 13
# 	QUEUE_FULL_EN 14 14
# 	PQ_EMPTY 15 15
# 	SLOT_BASED_WPTR 18 19
# 	MIN_AVAIL_SIZE 20 21
# 	TMZ 22 22
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
# 	PQ_VOLATILE 26 26
# 	NO_UPDATE_RPTR 27 27
# 	UNORD_DISPATCH 28 28
# 	TUNNEL_DISPATCH 29 29
# 	PRIV_STATE 30 30
# 	KMD_QUEUE 31 31
regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8
# 	DOORBELL_MODE 0 0
# 	DOORBELL_BIF_DROP 1 1
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_SOURCE 28 28
# 	DOORBELL_SCHD_HIT 29 29
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_HQD_PQ_RPTR = 0x1fb3
# 	CONSUMED_OFFSET 0 31
regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4
# 	RPTR_REPORT_ADDR 2 31
regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5
# 	RPTR_REPORT_ADDR_HI 0 15
regCP_HQD_PQ_WPTR_HI = 0x1fe0
# 	DATA 0 31
regCP_HQD_PQ_WPTR_LO = 0x1fdf
# 	OFFSET 0 31
regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6
# 	WPTR_ADDR 3 31
regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7
# 	WPTR_ADDR_HI 0 15
regCP_HQD_QUANTUM = 0x1fb0
# 	QUANTUM_EN 0 0
# 	QUANTUM_SCALE 4 4
# 	QUANTUM_DURATION 8 13
# 	QUANTUM_ACTIVE 31 31
regCP_HQD_QUEUE_PRIORITY = 0x1faf
# 	PRIORITY_LEVEL 0 3
regCP_HQD_SEMA_CMD = 0x1fc3
# 	RETRY 0 0
# 	RESULT 1 2
# 	POLLING_DIS 8 8
# 	MESSAGE_EN 9 9
regCP_HQD_SUSPEND_CNTL_STACK_DW_CNT = 0x1fe2
# 	CNT 0 13
regCP_HQD_SUSPEND_CNTL_STACK_OFFSET = 0x1fe1
# 	OFFSET 2 15
regCP_HQD_SUSPEND_WG_STATE_OFFSET = 0x1fe3
# 	OFFSET 2 25
regCP_HQD_VMID = 0x1fac
# 	VMID 0 3
# 	IB_VMID 8 11
# 	VQID 16 25
regCP_HQD_WG_STATE_OFFSET = 0x1fd9
# 	OFFSET 2 25
regCP_HYP_MEC1_UCODE_ADDR = 0x581a
# 	UCODE_ADDR 0 19
regCP_HYP_MEC1_UCODE_DATA = 0x581b
# 	UCODE_DATA 0 31
regCP_HYP_MEC2_UCODE_ADDR = 0x581c
# 	UCODE_ADDR 0 19
regCP_HYP_MEC2_UCODE_DATA = 0x581d
# 	UCODE_DATA 0 31
regCP_HYP_ME_UCODE_ADDR = 0x5816
# 	UCODE_ADDR 0 19
regCP_HYP_ME_UCODE_DATA = 0x5817
# 	UCODE_DATA 0 31
regCP_HYP_PFP_UCODE_ADDR = 0x5814
# 	UCODE_ADDR 0 19
regCP_HYP_PFP_UCODE_DATA = 0x5815
# 	UCODE_DATA 0 31
regCP_IB2_BASE_HI = 0x20d0
# 	IB2_BASE_HI 0 15
regCP_IB2_BASE_LO = 0x20cf
# 	IB2_BASE_LO 2 31
regCP_IB2_BUFSZ = 0x20d1
# 	IB2_BUFSZ 0 19
regCP_IB2_CMD_BUFSZ = 0x20c1
# 	IB2_CMD_REQSZ 0 19
regCP_IB2_OFFSET = 0x2093
# 	IB2_OFFSET 0 19
regCP_IB2_PREAMBLE_BEGIN = 0x2096
# 	IB2_PREAMBLE_BEGIN 0 19
regCP_IB2_PREAMBLE_END = 0x2097
# 	IB2_PREAMBLE_END 0 19
regCP_INDEX_BASE_ADDR = 0x20f8
# 	ADDR_LO 0 31
regCP_INDEX_BASE_ADDR_HI = 0x20f9
# 	ADDR_HI 0 15
regCP_INDEX_TYPE = 0x20fa
# 	INDEX_TYPE 0 1
regCP_INT_CNTL = 0x1de9
# 	RESUME_INT_ENABLE 8 8
# 	SUSPEND_INT_ENABLE 9 9
# 	DMA_WATCH_INT_ENABLE 10 10
# 	CP_VM_DOORBELL_WR_INT_ENABLE 11 11
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	CMP_BUSY_INT_ENABLE 18 18
# 	CNTX_BUSY_INT_ENABLE 19 19
# 	CNTX_EMPTY_INT_ENABLE 20 20
# 	GFX_IDLE_INT_ENABLE 21 21
# 	PRIV_INSTR_INT_ENABLE 22 22
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_INT_CNTL_RING0 = 0x1e0a
# 	RESUME_INT_ENABLE 8 8
# 	SUSPEND_INT_ENABLE 9 9
# 	DMA_WATCH_INT_ENABLE 10 10
# 	CP_VM_DOORBELL_WR_INT_ENABLE 11 11
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	CMP_BUSY_INT_ENABLE 18 18
# 	CNTX_BUSY_INT_ENABLE 19 19
# 	CNTX_EMPTY_INT_ENABLE 20 20
# 	GFX_IDLE_INT_ENABLE 21 21
# 	PRIV_INSTR_INT_ENABLE 22 22
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_INT_CNTL_RING1 = 0x1e0b
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_INSTR_INT_ENABLE 22 22
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_INT_STATUS = 0x1dea
# 	RESUME_INT_STAT 8 8
# 	SUSPEND_INT_STAT 9 9
# 	DMA_WATCH_INT_STAT 10 10
# 	CP_VM_DOORBELL_WR_INT_STAT 11 11
# 	CP_ECC_ERROR_INT_STAT 14 14
# 	GPF_INT_STAT 16 16
# 	WRM_POLL_TIMEOUT_INT_STAT 17 17
# 	CMP_BUSY_INT_STAT 18 18
# 	CNTX_BUSY_INT_STAT 19 19
# 	CNTX_EMPTY_INT_STAT 20 20
# 	GFX_IDLE_INT_STAT 21 21
# 	PRIV_INSTR_INT_STAT 22 22
# 	PRIV_REG_INT_STAT 23 23
# 	OPCODE_ERROR_INT_STAT 24 24
# 	TIME_STAMP_INT_STAT 26 26
# 	RESERVED_BIT_ERROR_INT_STAT 27 27
# 	GENERIC2_INT_STAT 29 29
# 	GENERIC1_INT_STAT 30 30
# 	GENERIC0_INT_STAT 31 31
regCP_INT_STATUS_RING0 = 0x1e0d
# 	RESUME_INT_STAT 8 8
# 	SUSPEND_INT_STAT 9 9
# 	DMA_WATCH_INT_STAT 10 10
# 	CP_VM_DOORBELL_WR_INT_STAT 11 11
# 	CP_ECC_ERROR_INT_STAT 14 14
# 	GPF_INT_STAT 16 16
# 	WRM_POLL_TIMEOUT_INT_STAT 17 17
# 	CMP_BUSY_INT_STAT 18 18
# 	GCNTX_BUSY_INT_STAT 19 19
# 	CNTX_EMPTY_INT_STAT 20 20
# 	GFX_IDLE_INT_STAT 21 21
# 	PRIV_INSTR_INT_STAT 22 22
# 	PRIV_REG_INT_STAT 23 23
# 	OPCODE_ERROR_INT_STAT 24 24
# 	TIME_STAMP_INT_STAT 26 26
# 	RESERVED_BIT_ERROR_INT_STAT 27 27
# 	GENERIC2_INT_STAT 29 29
# 	GENERIC1_INT_STAT 30 30
# 	GENERIC0_INT_STAT 31 31
regCP_INT_STATUS_RING1 = 0x1e0e
# 	CP_ECC_ERROR_INT_STAT 14 14
# 	GPF_INT_STAT 16 16
# 	WRM_POLL_TIMEOUT_INT_STAT 17 17
# 	PRIV_INSTR_INT_STAT 22 22
# 	PRIV_REG_INT_STAT 23 23
# 	OPCODE_ERROR_INT_STAT 24 24
# 	TIME_STAMP_INT_STAT 26 26
# 	RESERVED_BIT_ERROR_INT_STAT 27 27
# 	GENERIC2_INT_STAT 29 29
# 	GENERIC1_INT_STAT 30 30
# 	GENERIC0_INT_STAT 31 31
regCP_IQ_WAIT_TIME1 = 0x1e4f
# 	IB_OFFLOAD 0 7
# 	ATOMIC_OFFLOAD 8 15
# 	WRM_OFFLOAD 16 23
# 	GWS 24 31
regCP_IQ_WAIT_TIME2 = 0x1e50
# 	QUE_SLEEP 0 7
# 	SCH_WAVE 8 15
# 	SEM_REARM 16 23
# 	DEQ_RETRY 24 31
regCP_IQ_WAIT_TIME3 = 0x1e6a
# 	SUSPEND_QUE 0 7
regCP_MAX_CONTEXT = 0x1e4e
# 	MAX_CONTEXT 0 2
regCP_MAX_DRAW_COUNT = 0x1e5c
# 	MAX_DRAW_COUNT 0 31
regCP_ME0_PIPE0_PRIORITY = 0x1ded
# 	PRIORITY 0 1
regCP_ME0_PIPE0_VMID = 0x1df2
# 	VMID 0 3
regCP_ME0_PIPE1_PRIORITY = 0x1dee
# 	PRIORITY 0 1
regCP_ME0_PIPE1_VMID = 0x1df3
# 	VMID 0 3
regCP_ME0_PIPE_PRIORITY_CNTS = 0x1dec
# 	PRIORITY1_CNT 0 7
# 	PRIORITY2A_CNT 8 15
# 	PRIORITY2B_CNT 16 23
# 	PRIORITY3_CNT 24 31
regCP_ME1_PIPE0_INT_CNTL = 0x1e25
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME1_PIPE0_INT_STATUS = 0x1e2d
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME1_PIPE0_PRIORITY = 0x1e3a
# 	PRIORITY 0 1
regCP_ME1_PIPE1_INT_CNTL = 0x1e26
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME1_PIPE1_INT_STATUS = 0x1e2e
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME1_PIPE1_PRIORITY = 0x1e3b
# 	PRIORITY 0 1
regCP_ME1_PIPE2_INT_CNTL = 0x1e27
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME1_PIPE2_INT_STATUS = 0x1e2f
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME1_PIPE2_PRIORITY = 0x1e3c
# 	PRIORITY 0 1
regCP_ME1_PIPE3_INT_CNTL = 0x1e28
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME1_PIPE3_INT_STATUS = 0x1e30
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME1_PIPE3_PRIORITY = 0x1e3d
# 	PRIORITY 0 1
regCP_ME1_PIPE_PRIORITY_CNTS = 0x1e39
# 	PRIORITY1_CNT 0 7
# 	PRIORITY2A_CNT 8 15
# 	PRIORITY2B_CNT 16 23
# 	PRIORITY3_CNT 24 31
regCP_ME2_PIPE0_INT_CNTL = 0x1e29
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME2_PIPE0_INT_STATUS = 0x1e31
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME2_PIPE0_PRIORITY = 0x1e3f
# 	PRIORITY 0 1
regCP_ME2_PIPE1_INT_CNTL = 0x1e2a
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME2_PIPE1_INT_STATUS = 0x1e32
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME2_PIPE1_PRIORITY = 0x1e40
# 	PRIORITY 0 1
regCP_ME2_PIPE2_INT_CNTL = 0x1e2b
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME2_PIPE2_INT_STATUS = 0x1e33
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME2_PIPE2_PRIORITY = 0x1e41
# 	PRIORITY 0 1
regCP_ME2_PIPE3_INT_CNTL = 0x1e2c
# 	CMP_QUERY_STATUS_INT_ENABLE 12 12
# 	DEQUEUE_REQUEST_INT_ENABLE 13 13
# 	CP_ECC_ERROR_INT_ENABLE 14 14
# 	SUA_VIOLATION_INT_ENABLE 15 15
# 	GPF_INT_ENABLE 16 16
# 	WRM_POLL_TIMEOUT_INT_ENABLE 17 17
# 	PRIV_REG_INT_ENABLE 23 23
# 	OPCODE_ERROR_INT_ENABLE 24 24
# 	TIME_STAMP_INT_ENABLE 26 26
# 	RESERVED_BIT_ERROR_INT_ENABLE 27 27
# 	GENERIC2_INT_ENABLE 29 29
# 	GENERIC1_INT_ENABLE 30 30
# 	GENERIC0_INT_ENABLE 31 31
regCP_ME2_PIPE3_INT_STATUS = 0x1e34
# 	CMP_QUERY_STATUS_INT_STATUS 12 12
# 	DEQUEUE_REQUEST_INT_STATUS 13 13
# 	CP_ECC_ERROR_INT_STATUS 14 14
# 	SUA_VIOLATION_INT_STATUS 15 15
# 	GPF_INT_STATUS 16 16
# 	WRM_POLL_TIMEOUT_INT_STATUS 17 17
# 	PRIV_REG_INT_STATUS 23 23
# 	OPCODE_ERROR_INT_STATUS 24 24
# 	TIME_STAMP_INT_STATUS 26 26
# 	RESERVED_BIT_ERROR_INT_STATUS 27 27
# 	GENERIC2_INT_STATUS 29 29
# 	GENERIC1_INT_STATUS 30 30
# 	GENERIC0_INT_STATUS 31 31
regCP_ME2_PIPE3_PRIORITY = 0x1e42
# 	PRIORITY 0 1
regCP_ME2_PIPE_PRIORITY_CNTS = 0x1e3e
# 	PRIORITY1_CNT 0 7
# 	PRIORITY2A_CNT 8 15
# 	PRIORITY2B_CNT 16 23
# 	PRIORITY3_CNT 24 31
regCP_MEC1_F32_INTERRUPT = 0x1e16
# 	EDC_ROQ_FED_INT 0 0
# 	PRIV_REG_INT 1 1
# 	RESERVED_BIT_ERR_INT 2 2
# 	EDC_TC_FED_INT 3 3
# 	EDC_GDS_FED_INT 4 4
# 	EDC_SCRATCH_FED_INT 5 5
# 	WAVE_RESTORE_INT 6 6
# 	SUA_VIOLATION_INT 7 7
# 	EDC_DMA_FED_INT 8 8
# 	IQ_TIMER_INT 9 9
# 	GPF_INT_CPF 10 10
# 	GPF_INT_DMA 11 11
# 	GPF_INT_CPC 12 12
# 	EDC_SR_MEM_FED_INT 13 13
# 	QUEUE_MESSAGE_INT 14 14
# 	FATAL_EDC_ERROR_INT 15 15
regCP_MEC1_F32_INT_DIS = 0x1e5d
# 	EDC_ROQ_FED_INT 0 0
# 	PRIV_REG_INT 1 1
# 	RESERVED_BIT_ERR_INT 2 2
# 	EDC_TC_FED_INT 3 3
# 	EDC_GDS_FED_INT 4 4
# 	EDC_SCRATCH_FED_INT 5 5
# 	WAVE_RESTORE_INT 6 6
# 	SUA_VIOLATION_INT 7 7
# 	EDC_DMA_FED_INT 8 8
# 	IQ_TIMER_INT 9 9
# 	GPF_INT_CPF 10 10
# 	GPF_INT_DMA 11 11
# 	GPF_INT_CPC 12 12
# 	EDC_SR_MEM_FED_INT 13 13
# 	QUEUE_MESSAGE_INT 14 14
# 	FATAL_EDC_ERROR_INT 15 15
regCP_MEC1_INSTR_PNTR = 0xf48
# 	INSTR_PNTR 0 15
regCP_MEC1_INTR_ROUTINE_START = 0x1e4b
# 	IR_START 0 19
regCP_MEC1_PRGRM_CNTR_START = 0x1e46
# 	IP_START 0 19
regCP_MEC2_F32_INTERRUPT = 0x1e17
# 	EDC_ROQ_FED_INT 0 0
# 	PRIV_REG_INT 1 1
# 	RESERVED_BIT_ERR_INT 2 2
# 	EDC_TC_FED_INT 3 3
# 	EDC_GDS_FED_INT 4 4
# 	EDC_SCRATCH_FED_INT 5 5
# 	WAVE_RESTORE_INT 6 6
# 	SUA_VIOLATION_INT 7 7
# 	EDC_DMA_FED_INT 8 8
# 	IQ_TIMER_INT 9 9
# 	GPF_INT_CPF 10 10
# 	GPF_INT_DMA 11 11
# 	GPF_INT_CPC 12 12
# 	EDC_SR_MEM_FED_INT 13 13
# 	QUEUE_MESSAGE_INT 14 14
# 	FATAL_EDC_ERROR_INT 15 15
regCP_MEC2_F32_INT_DIS = 0x1e5e
# 	EDC_ROQ_FED_INT 0 0
# 	PRIV_REG_INT 1 1
# 	RESERVED_BIT_ERR_INT 2 2
# 	EDC_TC_FED_INT 3 3
# 	EDC_GDS_FED_INT 4 4
# 	EDC_SCRATCH_FED_INT 5 5
# 	WAVE_RESTORE_INT 6 6
# 	SUA_VIOLATION_INT 7 7
# 	EDC_DMA_FED_INT 8 8
# 	IQ_TIMER_INT 9 9
# 	GPF_INT_CPF 10 10
# 	GPF_INT_DMA 11 11
# 	GPF_INT_CPC 12 12
# 	EDC_SR_MEM_FED_INT 13 13
# 	QUEUE_MESSAGE_INT 14 14
# 	FATAL_EDC_ERROR_INT 15 15
regCP_MEC2_INSTR_PNTR = 0xf49
# 	INSTR_PNTR 0 15
regCP_MEC2_INTR_ROUTINE_START = 0x1e4c
# 	IR_START 0 19
regCP_MEC2_PRGRM_CNTR_START = 0x1e47
# 	IP_START 0 19
regCP_MEC_CNTL = 0x802
# 	MEC_ME1_PIPE0_RESET 16 16
# 	MEC_ME1_PIPE1_RESET 17 17
# 	MEC_ME1_PIPE2_RESET 18 18
# 	MEC_ME1_PIPE3_RESET 19 19
# 	MEC_ME2_PIPE0_RESET 20 20
# 	MEC_ME2_PIPE1_RESET 21 21
# 	MEC_ME2_PIPE2_RESET 22 22
# 	MEC_ME2_PIPE3_RESET 23 23
# 	MEC_INVALIDATE_ICACHE 27 27
# 	MEC_ME2_HALT 28 28
# 	MEC_ME2_STEP 29 29
# 	MEC_ME1_HALT 30 30
# 	MEC_ME1_STEP 31 31
regCP_MEC_DC_APERTURE0_BASE = 0x294a
# 	BASE 0 31
regCP_MEC_DC_APERTURE0_CNTL = 0x294c
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE0_MASK = 0x294b
# 	MASK 0 31
regCP_MEC_DC_APERTURE10_BASE = 0x2968
# 	BASE 0 31
regCP_MEC_DC_APERTURE10_CNTL = 0x296a
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE10_MASK = 0x2969
# 	MASK 0 31
regCP_MEC_DC_APERTURE11_BASE = 0x296b
# 	BASE 0 31
regCP_MEC_DC_APERTURE11_CNTL = 0x296d
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE11_MASK = 0x296c
# 	MASK 0 31
regCP_MEC_DC_APERTURE12_BASE = 0x296e
# 	BASE 0 31
regCP_MEC_DC_APERTURE12_CNTL = 0x2970
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE12_MASK = 0x296f
# 	MASK 0 31
regCP_MEC_DC_APERTURE13_BASE = 0x2971
# 	BASE 0 31
regCP_MEC_DC_APERTURE13_CNTL = 0x2973
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE13_MASK = 0x2972
# 	MASK 0 31
regCP_MEC_DC_APERTURE14_BASE = 0x2974
# 	BASE 0 31
regCP_MEC_DC_APERTURE14_CNTL = 0x2976
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE14_MASK = 0x2975
# 	MASK 0 31
regCP_MEC_DC_APERTURE15_BASE = 0x2977
# 	BASE 0 31
regCP_MEC_DC_APERTURE15_CNTL = 0x2979
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE15_MASK = 0x2978
# 	MASK 0 31
regCP_MEC_DC_APERTURE1_BASE = 0x294d
# 	BASE 0 31
regCP_MEC_DC_APERTURE1_CNTL = 0x294f
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE1_MASK = 0x294e
# 	MASK 0 31
regCP_MEC_DC_APERTURE2_BASE = 0x2950
# 	BASE 0 31
regCP_MEC_DC_APERTURE2_CNTL = 0x2952
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE2_MASK = 0x2951
# 	MASK 0 31
regCP_MEC_DC_APERTURE3_BASE = 0x2953
# 	BASE 0 31
regCP_MEC_DC_APERTURE3_CNTL = 0x2955
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE3_MASK = 0x2954
# 	MASK 0 31
regCP_MEC_DC_APERTURE4_BASE = 0x2956
# 	BASE 0 31
regCP_MEC_DC_APERTURE4_CNTL = 0x2958
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE4_MASK = 0x2957
# 	MASK 0 31
regCP_MEC_DC_APERTURE5_BASE = 0x2959
# 	BASE 0 31
regCP_MEC_DC_APERTURE5_CNTL = 0x295b
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE5_MASK = 0x295a
# 	MASK 0 31
regCP_MEC_DC_APERTURE6_BASE = 0x295c
# 	BASE 0 31
regCP_MEC_DC_APERTURE6_CNTL = 0x295e
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE6_MASK = 0x295d
# 	MASK 0 31
regCP_MEC_DC_APERTURE7_BASE = 0x295f
# 	BASE 0 31
regCP_MEC_DC_APERTURE7_CNTL = 0x2961
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE7_MASK = 0x2960
# 	MASK 0 31
regCP_MEC_DC_APERTURE8_BASE = 0x2962
# 	BASE 0 31
regCP_MEC_DC_APERTURE8_CNTL = 0x2964
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE8_MASK = 0x2963
# 	MASK 0 31
regCP_MEC_DC_APERTURE9_BASE = 0x2965
# 	BASE 0 31
regCP_MEC_DC_APERTURE9_CNTL = 0x2967
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MEC_DC_APERTURE9_MASK = 0x2966
# 	MASK 0 31
regCP_MEC_DC_BASE_CNTL = 0x290b
# 	VMID 0 3
# 	CACHE_POLICY 24 25
regCP_MEC_DC_BASE_HI = 0x5871
# 	DC_BASE_HI 0 15
regCP_MEC_DC_BASE_LO = 0x5870
# 	DC_BASE_LO 16 31
regCP_MEC_DC_OP_CNTL = 0x290c
# 	INVALIDATE_DCACHE 0 0
# 	INVALIDATE_DCACHE_COMPLETE 1 1
# 	BYPASS_ALL 2 2
regCP_MEC_DM_INDEX_ADDR = 0x5c02
# 	ADDR 0 31
regCP_MEC_DM_INDEX_DATA = 0x5c03
# 	DATA 0 31
regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc
# 	DOORBELL_RANGE_LOWER 2 11
regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd
# 	DOORBELL_RANGE_UPPER 2 11
regCP_MEC_GP0_HI = 0x2911
# 	M_RET_ADDR 0 31
regCP_MEC_GP0_LO = 0x2910
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_MEC_GP1_HI = 0x2913
# 	RD_WR_SELECT_HI 0 31
regCP_MEC_GP1_LO = 0x2912
# 	RD_WR_SELECT_LO 0 31
regCP_MEC_GP2_HI = 0x2915
# 	STACK_PNTR_HI 0 31
regCP_MEC_GP2_LO = 0x2914
# 	STACK_PNTR_LO 0 31
regCP_MEC_GP3_HI = 0x2917
# 	DATA 0 31
regCP_MEC_GP3_LO = 0x2916
# 	DATA 0 31
regCP_MEC_GP4_HI = 0x2919
# 	DATA 0 31
regCP_MEC_GP4_LO = 0x2918
# 	DATA 0 31
regCP_MEC_GP5_HI = 0x291b
# 	M_RET_ADDR 0 31
regCP_MEC_GP5_LO = 0x291a
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_MEC_GP6_HI = 0x291d
# 	RD_WR_SELECT_HI 0 31
regCP_MEC_GP6_LO = 0x291c
# 	RD_WR_SELECT_LO 0 31
regCP_MEC_GP7_HI = 0x291f
# 	STACK_PNTR_HI 0 31
regCP_MEC_GP7_LO = 0x291e
# 	STACK_PNTR_LO 0 31
regCP_MEC_GP8_HI = 0x2921
# 	DATA 0 31
regCP_MEC_GP8_LO = 0x2920
# 	DATA 0 31
regCP_MEC_GP9_HI = 0x2923
# 	DATA 0 31
regCP_MEC_GP9_LO = 0x2922
# 	DATA 0 31
regCP_MEC_ISA_CNTL = 0x2903
# 	ISA_MODE 0 0
regCP_MEC_JT_STAT = 0x1ed5
# 	JT_LOADED 0 7
# 	WR_MASK 16 23
regCP_MEC_LOCAL_APERTURE = 0x292b
# 	APERTURE 0 2
regCP_MEC_LOCAL_BASE0_HI = 0x2928
# 	BASE0_HI 0 15
regCP_MEC_LOCAL_BASE0_LO = 0x2927
# 	BASE0_LO 16 31
regCP_MEC_LOCAL_INSTR_APERTURE = 0x2930
# 	APERTURE 0 2
regCP_MEC_LOCAL_INSTR_BASE_HI = 0x292d
# 	BASE_HI 0 15
regCP_MEC_LOCAL_INSTR_BASE_LO = 0x292c
# 	BASE_LO 16 31
regCP_MEC_LOCAL_INSTR_MASK_HI = 0x292f
# 	MASK_HI 0 15
regCP_MEC_LOCAL_INSTR_MASK_LO = 0x292e
# 	MASK_LO 16 31
regCP_MEC_LOCAL_MASK0_HI = 0x292a
# 	MASK0_HI 0 15
regCP_MEC_LOCAL_MASK0_LO = 0x2929
# 	MASK0_LO 16 31
regCP_MEC_LOCAL_SCRATCH_APERTURE = 0x2931
# 	APERTURE 0 2
regCP_MEC_LOCAL_SCRATCH_BASE_HI = 0x2933
# 	BASE_HI 0 15
regCP_MEC_LOCAL_SCRATCH_BASE_LO = 0x2932
# 	BASE_LO 16 31
regCP_MEC_MDBASE_HI = 0x5871
# 	BASE_HI 0 15
regCP_MEC_MDBASE_LO = 0x5870
# 	BASE_LO 16 31
regCP_MEC_MDBOUND_HI = 0x5875
# 	BOUND_HI 0 31
regCP_MEC_MDBOUND_LO = 0x5874
# 	BOUND_LO 0 31
regCP_MEC_ME1_HEADER_DUMP = 0xe2e
# 	HEADER_DUMP 0 31
regCP_MEC_ME1_UCODE_ADDR = 0x581a
# 	UCODE_ADDR 0 19
regCP_MEC_ME1_UCODE_DATA = 0x581b
# 	UCODE_DATA 0 31
regCP_MEC_ME2_HEADER_DUMP = 0xe2f
# 	HEADER_DUMP 0 31
regCP_MEC_ME2_UCODE_ADDR = 0x581c
# 	UCODE_ADDR 0 19
regCP_MEC_ME2_UCODE_DATA = 0x581d
# 	UCODE_DATA 0 31
regCP_MEC_MIBOUND_HI = 0x5873
# 	BOUND_HI 0 31
regCP_MEC_MIBOUND_LO = 0x5872
# 	BOUND_LO 0 31
regCP_MEC_MIE_HI = 0x2906
# 	MEC_INT 0 31
regCP_MEC_MIE_LO = 0x2905
# 	MEC_INT 0 31
regCP_MEC_MIP_HI = 0x290a
# 	MIP_HI 0 31
regCP_MEC_MIP_LO = 0x2909
# 	MIP_LO 0 31
regCP_MEC_MTIMECMP_HI = 0x290e
# 	TIME_HI 0 31
regCP_MEC_MTIMECMP_LO = 0x290d
# 	TIME_LO 0 31
regCP_MEC_MTVEC_HI = 0x2902
# 	ADDR_LO 0 31
regCP_MEC_MTVEC_LO = 0x2901
# 	ADDR_LO 0 31
regCP_MEC_RS64_CNTL = 0x2904
# 	MEC_INVALIDATE_ICACHE 4 4
# 	MEC_PIPE0_RESET 16 16
# 	MEC_PIPE1_RESET 17 17
# 	MEC_PIPE2_RESET 18 18
# 	MEC_PIPE3_RESET 19 19
# 	MEC_PIPE0_ACTIVE 26 26
# 	MEC_PIPE1_ACTIVE 27 27
# 	MEC_PIPE2_ACTIVE 28 28
# 	MEC_PIPE3_ACTIVE 29 29
# 	MEC_HALT 30 30
# 	MEC_STEP 31 31
regCP_MEC_RS64_INSTR_PNTR = 0x2908
# 	INSTR_PNTR 0 19
regCP_MEC_RS64_INTERRUPT = 0x2907
# 	MEC_INT 0 31
regCP_MEC_RS64_INTERRUPT_DATA_16 = 0x293a
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_17 = 0x293b
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_18 = 0x293c
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_19 = 0x293d
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_20 = 0x293e
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_21 = 0x293f
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_22 = 0x2940
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_23 = 0x2941
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_24 = 0x2942
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_25 = 0x2943
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_26 = 0x2944
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_27 = 0x2945
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_28 = 0x2946
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_29 = 0x2947
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_30 = 0x2948
# 	DATA 0 31
regCP_MEC_RS64_INTERRUPT_DATA_31 = 0x2949
# 	DATA 0 31
regCP_MEC_RS64_PENDING_INTERRUPT = 0x2935
# 	PENDING_INTERRUPT 0 31
regCP_MEC_RS64_PERFCOUNT_CNTL = 0x2934
# 	EVENT_SEL 0 4
regCP_MEC_RS64_PRGRM_CNTR_START = 0x2900
# 	IP_START 0 31
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
# 	IP_START 0 29
regCP_MEQ_AVAIL = 0xf7d
# 	MEQ_CNT 0 9
regCP_MEQ_STAT = 0xf85
# 	MEQ_RPTR 0 9
# 	MEQ_WPTR 16 25
regCP_MEQ_THRESHOLDS = 0xf79
# 	MEQ1_START 0 7
# 	MEQ2_START 8 15
regCP_MES_CNTL = 0x2807
# 	MES_INVALIDATE_ICACHE 4 4
# 	MES_PIPE0_RESET 16 16
# 	MES_PIPE1_RESET 17 17
# 	MES_PIPE2_RESET 18 18
# 	MES_PIPE3_RESET 19 19
# 	MES_PIPE0_ACTIVE 26 26
# 	MES_PIPE1_ACTIVE 27 27
# 	MES_PIPE2_ACTIVE 28 28
# 	MES_PIPE3_ACTIVE 29 29
# 	MES_HALT 30 30
# 	MES_STEP 31 31
regCP_MES_DC_APERTURE0_BASE = 0x28af
# 	BASE 0 31
regCP_MES_DC_APERTURE0_CNTL = 0x28b1
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE0_MASK = 0x28b0
# 	MASK 0 31
regCP_MES_DC_APERTURE10_BASE = 0x28cd
# 	BASE 0 31
regCP_MES_DC_APERTURE10_CNTL = 0x28cf
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE10_MASK = 0x28ce
# 	MASK 0 31
regCP_MES_DC_APERTURE11_BASE = 0x28d0
# 	BASE 0 31
regCP_MES_DC_APERTURE11_CNTL = 0x28d2
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE11_MASK = 0x28d1
# 	MASK 0 31
regCP_MES_DC_APERTURE12_BASE = 0x28d3
# 	BASE 0 31
regCP_MES_DC_APERTURE12_CNTL = 0x28d5
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE12_MASK = 0x28d4
# 	MASK 0 31
regCP_MES_DC_APERTURE13_BASE = 0x28d6
# 	BASE 0 31
regCP_MES_DC_APERTURE13_CNTL = 0x28d8
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE13_MASK = 0x28d7
# 	MASK 0 31
regCP_MES_DC_APERTURE14_BASE = 0x28d9
# 	BASE 0 31
regCP_MES_DC_APERTURE14_CNTL = 0x28db
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE14_MASK = 0x28da
# 	MASK 0 31
regCP_MES_DC_APERTURE15_BASE = 0x28dc
# 	BASE 0 31
regCP_MES_DC_APERTURE15_CNTL = 0x28de
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE15_MASK = 0x28dd
# 	MASK 0 31
regCP_MES_DC_APERTURE1_BASE = 0x28b2
# 	BASE 0 31
regCP_MES_DC_APERTURE1_CNTL = 0x28b4
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE1_MASK = 0x28b3
# 	MASK 0 31
regCP_MES_DC_APERTURE2_BASE = 0x28b5
# 	BASE 0 31
regCP_MES_DC_APERTURE2_CNTL = 0x28b7
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE2_MASK = 0x28b6
# 	MASK 0 31
regCP_MES_DC_APERTURE3_BASE = 0x28b8
# 	BASE 0 31
regCP_MES_DC_APERTURE3_CNTL = 0x28ba
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE3_MASK = 0x28b9
# 	MASK 0 31
regCP_MES_DC_APERTURE4_BASE = 0x28bb
# 	BASE 0 31
regCP_MES_DC_APERTURE4_CNTL = 0x28bd
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE4_MASK = 0x28bc
# 	MASK 0 31
regCP_MES_DC_APERTURE5_BASE = 0x28be
# 	BASE 0 31
regCP_MES_DC_APERTURE5_CNTL = 0x28c0
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE5_MASK = 0x28bf
# 	MASK 0 31
regCP_MES_DC_APERTURE6_BASE = 0x28c1
# 	BASE 0 31
regCP_MES_DC_APERTURE6_CNTL = 0x28c3
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE6_MASK = 0x28c2
# 	MASK 0 31
regCP_MES_DC_APERTURE7_BASE = 0x28c4
# 	BASE 0 31
regCP_MES_DC_APERTURE7_CNTL = 0x28c6
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE7_MASK = 0x28c5
# 	MASK 0 31
regCP_MES_DC_APERTURE8_BASE = 0x28c7
# 	BASE 0 31
regCP_MES_DC_APERTURE8_CNTL = 0x28c9
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE8_MASK = 0x28c8
# 	MASK 0 31
regCP_MES_DC_APERTURE9_BASE = 0x28ca
# 	BASE 0 31
regCP_MES_DC_APERTURE9_CNTL = 0x28cc
# 	VMID 0 3
# 	BYPASS_MODE 4 4
regCP_MES_DC_APERTURE9_MASK = 0x28cb
# 	MASK 0 31
regCP_MES_DC_BASE_CNTL = 0x2836
# 	VMID 0 3
# 	CACHE_POLICY 24 25
regCP_MES_DC_BASE_HI = 0x5855
# 	DC_BASE_HI 0 15
regCP_MES_DC_BASE_LO = 0x5854
# 	DC_BASE_LO 16 31
regCP_MES_DC_OP_CNTL = 0x2837
# 	INVALIDATE_DCACHE 0 0
# 	INVALIDATE_DCACHE_COMPLETE 1 1
# 	BYPASS_ALL 2 2
regCP_MES_DM_INDEX_ADDR = 0x5c00
# 	ADDR 0 31
regCP_MES_DM_INDEX_DATA = 0x5c01
# 	DATA 0 31
regCP_MES_DOORBELL_CONTROL1 = 0x283c
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_DOORBELL_CONTROL2 = 0x283d
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_DOORBELL_CONTROL3 = 0x283e
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_DOORBELL_CONTROL4 = 0x283f
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_DOORBELL_CONTROL5 = 0x2840
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_DOORBELL_CONTROL6 = 0x2841
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_MES_GP0_HI = 0x2844
# 	M_RET_ADDR 0 31
regCP_MES_GP0_LO = 0x2843
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_MES_GP1_HI = 0x2846
# 	RD_WR_SELECT_HI 0 31
regCP_MES_GP1_LO = 0x2845
# 	RD_WR_SELECT_LO 0 31
regCP_MES_GP2_HI = 0x2848
# 	STACK_PNTR_HI 0 31
regCP_MES_GP2_LO = 0x2847
# 	STACK_PNTR_LO 0 31
regCP_MES_GP3_HI = 0x284a
# 	DATA 0 31
regCP_MES_GP3_LO = 0x2849
# 	DATA 0 31
regCP_MES_GP4_HI = 0x284c
# 	DATA 0 31
regCP_MES_GP4_LO = 0x284b
# 	DATA 0 31
regCP_MES_GP5_HI = 0x284e
# 	M_RET_ADDR 0 31
regCP_MES_GP5_LO = 0x284d
# 	PG_VIRT_HALTED 0 0
# 	DATA 1 31
regCP_MES_GP6_HI = 0x2850
# 	RD_WR_SELECT_HI 0 31
regCP_MES_GP6_LO = 0x284f
# 	RD_WR_SELECT_LO 0 31
regCP_MES_GP7_HI = 0x2852
# 	STACK_PNTR_HI 0 31
regCP_MES_GP7_LO = 0x2851
# 	STACK_PNTR_LO 0 31
regCP_MES_GP8_HI = 0x2854
# 	DATA 0 31
regCP_MES_GP8_LO = 0x2853
# 	DATA 0 31
regCP_MES_GP9_HI = 0x2856
# 	DATA 0 31
regCP_MES_GP9_LO = 0x2855
# 	DATA 0 31
regCP_MES_HEADER_DUMP = 0x280d
# 	HEADER_DUMP 0 31
regCP_MES_IC_BASE_CNTL = 0x5852
# 	VMID 0 3
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
regCP_MES_IC_BASE_HI = 0x5851
# 	IC_BASE_HI 0 15
regCP_MES_IC_BASE_LO = 0x5850
# 	IC_BASE_LO 12 31
regCP_MES_IC_OP_CNTL = 0x2820
# 	INVALIDATE_CACHE 0 0
# 	PRIME_ICACHE 4 4
# 	ICACHE_PRIMED 5 5
regCP_MES_INSTR_PNTR = 0x2813
# 	INSTR_PNTR 0 19
regCP_MES_INTERRUPT = 0x2810
# 	MES_INT 0 31
regCP_MES_INTERRUPT_DATA_16 = 0x289f
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_17 = 0x28a0
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_18 = 0x28a1
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_19 = 0x28a2
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_20 = 0x28a3
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_21 = 0x28a4
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_22 = 0x28a5
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_23 = 0x28a6
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_24 = 0x28a7
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_25 = 0x28a8
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_26 = 0x28a9
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_27 = 0x28aa
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_28 = 0x28ab
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_29 = 0x28ac
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_30 = 0x28ad
# 	DATA 0 31
regCP_MES_INTERRUPT_DATA_31 = 0x28ae
# 	DATA 0 31
regCP_MES_INTR_ROUTINE_START = 0x2801
# 	IR_START 0 31
regCP_MES_INTR_ROUTINE_START_HI = 0x2802
# 	IR_START 0 31
regCP_MES_LOCAL_APERTURE = 0x2887
# 	APERTURE 0 2
regCP_MES_LOCAL_BASE0_HI = 0x2884
# 	BASE0_HI 0 15
regCP_MES_LOCAL_BASE0_LO = 0x2883
# 	BASE0_LO 16 31
regCP_MES_LOCAL_INSTR_APERTURE = 0x288c
# 	APERTURE 0 2
regCP_MES_LOCAL_INSTR_BASE_HI = 0x2889
# 	BASE_HI 0 15
regCP_MES_LOCAL_INSTR_BASE_LO = 0x2888
# 	BASE_LO 16 31
regCP_MES_LOCAL_INSTR_MASK_HI = 0x288b
# 	MASK_HI 0 15
regCP_MES_LOCAL_INSTR_MASK_LO = 0x288a
# 	MASK_LO 16 31
regCP_MES_LOCAL_MASK0_HI = 0x2886
# 	MASK0_HI 0 15
regCP_MES_LOCAL_MASK0_LO = 0x2885
# 	MASK0_LO 16 31
regCP_MES_LOCAL_SCRATCH_APERTURE = 0x288d
# 	APERTURE 0 2
regCP_MES_LOCAL_SCRATCH_BASE_HI = 0x288f
# 	BASE_HI 0 15
regCP_MES_LOCAL_SCRATCH_BASE_LO = 0x288e
# 	BASE_LO 16 31
regCP_MES_MARCHID_HI = 0x2831
# 	MARCHID_HI 0 31
regCP_MES_MARCHID_LO = 0x2830
# 	MARCHID_LO 0 31
regCP_MES_MBADADDR_HI = 0x281d
# 	ADDR_HI 0 31
regCP_MES_MBADADDR_LO = 0x281c
# 	ADDR_LO 0 31
regCP_MES_MCAUSE_HI = 0x281b
# 	CAUSE_HI 0 31
regCP_MES_MCAUSE_LO = 0x281a
# 	CAUSE_LO 0 31
regCP_MES_MCYCLE_HI = 0x2827
# 	CYCLE_HI 0 31
regCP_MES_MCYCLE_LO = 0x2826
# 	CYCLE_LO 0 31
regCP_MES_MDBASE_HI = 0x5855
# 	BASE_HI 0 15
regCP_MES_MDBASE_LO = 0x5854
# 	BASE_LO 16 31
regCP_MES_MDBOUND_HI = 0x585e
# 	BOUND_HI 0 31
regCP_MES_MDBOUND_LO = 0x585d
# 	BOUND_LO 0 31
regCP_MES_MEPC_HI = 0x2819
# 	MEPC_HI 0 31
regCP_MES_MEPC_LO = 0x2818
# 	MEPC_LO 0 31
regCP_MES_MHARTID_HI = 0x2835
# 	MHARTID_HI 0 31
regCP_MES_MHARTID_LO = 0x2834
# 	MHARTID_LO 0 31
regCP_MES_MIBASE_HI = 0x5851
# 	IC_BASE_HI 0 15
regCP_MES_MIBASE_LO = 0x5850
# 	IC_BASE_LO 12 31
regCP_MES_MIBOUND_HI = 0x585c
# 	BOUND_HI 0 31
regCP_MES_MIBOUND_LO = 0x585b
# 	BOUND_LO 0 31
regCP_MES_MIE_HI = 0x280f
# 	MES_INT 0 31
regCP_MES_MIE_LO = 0x280e
# 	MES_INT 0 31
regCP_MES_MIMPID_HI = 0x2833
# 	MIMPID_HI 0 31
regCP_MES_MIMPID_LO = 0x2832
# 	MIMPID_LO 0 31
regCP_MES_MINSTRET_HI = 0x282b
# 	INSTRET_HI 0 31
regCP_MES_MINSTRET_LO = 0x282a
# 	INSTRET_LO 0 31
regCP_MES_MIP_HI = 0x281f
# 	MIP_HI 0 31
regCP_MES_MIP_LO = 0x281e
# 	MIP_LO 0 31
regCP_MES_MISA_HI = 0x282d
# 	MISA_HI 0 31
regCP_MES_MISA_LO = 0x282c
# 	MISA_LO 0 31
regCP_MES_MSCRATCH_HI = 0x2814
# 	DATA 0 31
regCP_MES_MSCRATCH_LO = 0x2815
# 	DATA 0 31
regCP_MES_MSTATUS_HI = 0x2817
# 	STATUS_HI 0 31
regCP_MES_MSTATUS_LO = 0x2816
# 	STATUS_LO 0 31
regCP_MES_MTIMECMP_HI = 0x2839
# 	TIME_HI 0 31
regCP_MES_MTIMECMP_LO = 0x2838
# 	TIME_LO 0 31
regCP_MES_MTIME_HI = 0x2829
# 	TIME_HI 0 31
regCP_MES_MTIME_LO = 0x2828
# 	TIME_LO 0 31
regCP_MES_MTVEC_HI = 0x2802
# 	ADDR_LO 0 31
regCP_MES_MTVEC_LO = 0x2801
# 	ADDR_LO 0 31
regCP_MES_MVENDORID_HI = 0x282f
# 	MVENDORID_HI 0 31
regCP_MES_MVENDORID_LO = 0x282e
# 	MVENDORID_LO 0 31
regCP_MES_PENDING_INTERRUPT = 0x289a
# 	PENDING_INTERRUPT 0 31
regCP_MES_PERFCOUNT_CNTL = 0x2899
# 	EVENT_SEL 0 4
regCP_MES_PIPE0_PRIORITY = 0x2809
# 	PRIORITY 0 1
regCP_MES_PIPE1_PRIORITY = 0x280a
# 	PRIORITY 0 1
regCP_MES_PIPE2_PRIORITY = 0x280b
# 	PRIORITY 0 1
regCP_MES_PIPE3_PRIORITY = 0x280c
# 	PRIORITY 0 1
regCP_MES_PIPE_PRIORITY_CNTS = 0x2808
# 	PRIORITY1_CNT 0 7
# 	PRIORITY2A_CNT 8 15
# 	PRIORITY2B_CNT 16 23
# 	PRIORITY3_CNT 24 31
regCP_MES_PRGRM_CNTR_START = 0x2800
# 	IP_START 0 31
regCP_MES_PRGRM_CNTR_START_HI = 0x289d
# 	IP_START 0 29
regCP_MES_PROCESS_QUANTUM_PIPE0 = 0x283a
# 	QUANTUM_DURATION 0 27
# 	TIMER_EXPIRED 28 28
# 	QUANTUM_SCALE 29 30
# 	QUANTUM_EN 31 31
regCP_MES_PROCESS_QUANTUM_PIPE1 = 0x283b
# 	QUANTUM_DURATION 0 27
# 	TIMER_EXPIRED 28 28
# 	QUANTUM_SCALE 29 30
# 	QUANTUM_EN 31 31
regCP_MES_SCRATCH_DATA = 0x2812
# 	SCRATCH_DATA 0 31
regCP_MES_SCRATCH_INDEX = 0x2811
# 	SCRATCH_INDEX 0 8
# 	SCRATCH_INDEX_64BIT_MODE 31 31
regCP_ME_ATOMIC_PREOP_HI = 0x205e
# 	ATOMIC_PREOP_HI 0 31
regCP_ME_ATOMIC_PREOP_LO = 0x205d
# 	ATOMIC_PREOP_LO 0 31
regCP_ME_CNTL = 0x803
# 	CE_INVALIDATE_ICACHE 4 4
# 	PFP_INVALIDATE_ICACHE 6 6
# 	ME_INVALIDATE_ICACHE 8 8
# 	PFP_PIPE0_DISABLE 12 12
# 	PFP_PIPE1_DISABLE 13 13
# 	ME_PIPE0_DISABLE 14 14
# 	ME_PIPE1_DISABLE 15 15
# 	CE_PIPE0_RESET 16 16
# 	CE_PIPE1_RESET 17 17
# 	PFP_PIPE0_RESET 18 18
# 	PFP_PIPE1_RESET 19 19
# 	ME_PIPE0_RESET 20 20
# 	ME_PIPE1_RESET 21 21
# 	CE_HALT 24 24
# 	CE_STEP 25 25
# 	PFP_HALT 26 26
# 	PFP_STEP 27 27
# 	ME_HALT 28 28
# 	ME_STEP 29 29
regCP_ME_COHER_BASE = 0x2101
# 	COHER_BASE_256B 0 31
regCP_ME_COHER_BASE_HI = 0x2102
# 	COHER_BASE_HI_256B 0 7
regCP_ME_COHER_CNTL = 0x20fe
# 	DEST_BASE_0_ENA 0 0
# 	DEST_BASE_1_ENA 1 1
# 	CB0_DEST_BASE_ENA 6 6
# 	CB1_DEST_BASE_ENA 7 7
# 	CB2_DEST_BASE_ENA 8 8
# 	CB3_DEST_BASE_ENA 9 9
# 	CB4_DEST_BASE_ENA 10 10
# 	CB5_DEST_BASE_ENA 11 11
# 	CB6_DEST_BASE_ENA 12 12
# 	CB7_DEST_BASE_ENA 13 13
# 	DB_DEST_BASE_ENA 14 14
# 	DEST_BASE_2_ENA 19 19
# 	DEST_BASE_3_ENA 21 21
regCP_ME_COHER_SIZE = 0x20ff
# 	COHER_SIZE_256B 0 31
regCP_ME_COHER_SIZE_HI = 0x2100
# 	COHER_SIZE_HI_256B 0 7
regCP_ME_COHER_STATUS = 0x2103
# 	MATCHING_GFX_CNTX 0 7
# 	STATUS 31 31
regCP_ME_F32_INTERRUPT = 0x1e13
# 	ECC_ERROR_INT 0 0
# 	TIME_STAMP_INT 1 1
# 	ME_F32_INT_2 2 2
# 	ME_F32_INT_3 3 3
regCP_ME_GDS_ATOMIC0_PREOP_HI = 0x2060
# 	GDS_ATOMIC0_PREOP_HI 0 31
regCP_ME_GDS_ATOMIC0_PREOP_LO = 0x205f
# 	GDS_ATOMIC0_PREOP_LO 0 31
regCP_ME_GDS_ATOMIC1_PREOP_HI = 0x2062
# 	GDS_ATOMIC1_PREOP_HI 0 31
regCP_ME_GDS_ATOMIC1_PREOP_LO = 0x2061
# 	GDS_ATOMIC1_PREOP_LO 0 31
regCP_ME_HEADER_DUMP = 0xf41
# 	ME_HEADER_DUMP 0 31
regCP_ME_IC_BASE_CNTL = 0x5846
# 	VMID 0 3
# 	ADDRESS_CLAMP 4 4
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
regCP_ME_IC_BASE_HI = 0x5845
# 	IC_BASE_HI 0 15
regCP_ME_IC_BASE_LO = 0x5844
# 	IC_BASE_LO 12 31
regCP_ME_IC_OP_CNTL = 0x5847
# 	INVALIDATE_CACHE 0 0
# 	INVALIDATE_CACHE_COMPLETE 1 1
# 	PRIME_ICACHE 4 4
# 	ICACHE_PRIMED 5 5
regCP_ME_INSTR_PNTR = 0xf46
# 	INSTR_PNTR 0 15
regCP_ME_INTR_ROUTINE_START = 0x1e4a
# 	IR_START 0 31
regCP_ME_INTR_ROUTINE_START_HI = 0x1e7b
# 	IR_START 0 29
regCP_ME_MC_RADDR_HI = 0x206e
# 	ME_MC_RADDR_HI 0 15
# 	SIZE 16 19
# 	CACHE_POLICY 22 23
# 	VMID 24 27
# 	PRIVILEGE 31 31
regCP_ME_MC_RADDR_LO = 0x206d
# 	ME_MC_RADDR_LO 2 31
regCP_ME_MC_WADDR_HI = 0x206a
# 	ME_MC_WADDR_HI 0 15
# 	WRITE_CONFIRM 17 17
# 	WRITE64 18 18
# 	CACHE_POLICY 22 23
# 	VMID 24 27
# 	RINGID 28 29
# 	PRIVILEGE 31 31
regCP_ME_MC_WADDR_LO = 0x2069
# 	ME_MC_WADDR_LO 2 31
regCP_ME_MC_WDATA_HI = 0x206c
# 	ME_MC_WDATA_HI 0 31
regCP_ME_MC_WDATA_LO = 0x206b
# 	ME_MC_WDATA_LO 0 31
regCP_ME_PREEMPTION = 0xf59
# 	OBSOLETE 0 0
regCP_ME_PRGRM_CNTR_START = 0x1e45
# 	IP_START 0 31
regCP_ME_PRGRM_CNTR_START_HI = 0x1e79
# 	IP_START 0 29
regCP_ME_RAM_DATA = 0x5817
# 	ME_RAM_DATA 0 31
regCP_ME_RAM_RADDR = 0x5816
# 	ME_RAM_RADDR 0 19
regCP_ME_RAM_WADDR = 0x5816
# 	ME_RAM_WADDR 0 20
regCP_ME_SDMA_CS = 0x1f50
# 	REQUEST_GRANT 0 0
# 	SDMA_ID 4 7
# 	REQUEST_POSITION 8 11
# 	SDMA_COUNT 12 13
regCP_MQD_BASE_ADDR = 0x1fa9
# 	BASE_ADDR 2 31
regCP_MQD_BASE_ADDR_HI = 0x1faa
# 	BASE_ADDR_HI 0 15
regCP_MQD_CONTROL = 0x1fcb
# 	VMID 0 3
# 	PRIV_STATE 8 8
# 	PROCESSING_MQD 12 12
# 	PROCESSING_MQD_EN 13 13
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
# 	MQD_VOLATILE 26 26
regCP_PA_CINVOC_COUNT_HI = 0x2029
# 	CINVOC_COUNT_HI 0 31
regCP_PA_CINVOC_COUNT_LO = 0x2028
# 	CINVOC_COUNT_LO 0 31
regCP_PA_CPRIM_COUNT_HI = 0x202b
# 	CPRIM_COUNT_HI 0 31
regCP_PA_CPRIM_COUNT_LO = 0x202a
# 	CPRIM_COUNT_LO 0 31
regCP_PA_MSPRIM_COUNT_HI = 0x20a5
# 	MSPRIM_COUNT_HI 0 31
regCP_PA_MSPRIM_COUNT_LO = 0x20a4
# 	MSPRIM_COUNT_LO 0 31
regCP_PERFMON_CNTL = 0x3808
# 	PERFMON_STATE 0 3
# 	SPM_PERFMON_STATE 4 7
# 	PERFMON_ENABLE_MODE 8 9
# 	PERFMON_SAMPLE_ENABLE 10 10
regCP_PERFMON_CNTX_CNTL = 0xd8
# 	PERFMON_ENABLE 31 31
regCP_PFP_ATOMIC_PREOP_HI = 0x2053
# 	ATOMIC_PREOP_HI 0 31
regCP_PFP_ATOMIC_PREOP_LO = 0x2052
# 	ATOMIC_PREOP_LO 0 31
regCP_PFP_COMPLETION_STATUS = 0x20ec
# 	STATUS 0 1
regCP_PFP_F32_INTERRUPT = 0x1e14
# 	ECC_ERROR_INT 0 0
# 	PRIV_REG_INT 1 1
# 	RESERVED_BIT_ERR_INT 2 2
# 	PFP_F32_INT_3 3 3
regCP_PFP_GDS_ATOMIC0_PREOP_HI = 0x2055
# 	GDS_ATOMIC0_PREOP_HI 0 31
regCP_PFP_GDS_ATOMIC0_PREOP_LO = 0x2054
# 	GDS_ATOMIC0_PREOP_LO 0 31
regCP_PFP_GDS_ATOMIC1_PREOP_HI = 0x2057
# 	GDS_ATOMIC1_PREOP_HI 0 31
regCP_PFP_GDS_ATOMIC1_PREOP_LO = 0x2056
# 	GDS_ATOMIC1_PREOP_LO 0 31
regCP_PFP_HEADER_DUMP = 0xf42
# 	PFP_HEADER_DUMP 0 31
regCP_PFP_IB_CONTROL = 0x208d
# 	IB_EN 0 7
regCP_PFP_IC_BASE_CNTL = 0x5842
# 	VMID 0 3
# 	ADDRESS_CLAMP 4 4
# 	EXE_DISABLE 23 23
# 	CACHE_POLICY 24 25
regCP_PFP_IC_BASE_HI = 0x5841
# 	IC_BASE_HI 0 15
regCP_PFP_IC_BASE_LO = 0x5840
# 	IC_BASE_LO 12 31
regCP_PFP_IC_OP_CNTL = 0x5843
# 	INVALIDATE_CACHE 0 0
# 	INVALIDATE_CACHE_COMPLETE 1 1
# 	PRIME_ICACHE 4 4
# 	ICACHE_PRIMED 5 5
regCP_PFP_INSTR_PNTR = 0xf45
# 	INSTR_PNTR 0 15
regCP_PFP_INTR_ROUTINE_START = 0x1e49
# 	IR_START 0 31
regCP_PFP_INTR_ROUTINE_START_HI = 0x1e7a
# 	IR_START 0 29
regCP_PFP_JT_STAT = 0x1ed3
# 	JT_LOADED 0 1
# 	WR_MASK 16 17
regCP_PFP_LOAD_CONTROL = 0x208e
# 	CONFIG_REG_EN 0 0
# 	CNTX_REG_EN 1 1
# 	UCONFIG_REG_EN 15 15
# 	SH_GFX_REG_EN 16 16
# 	SH_CS_REG_EN 24 24
# 	LOAD_ORDINAL 31 31
regCP_PFP_METADATA_BASE_ADDR = 0x20f0
# 	ADDR_LO 0 31
regCP_PFP_METADATA_BASE_ADDR_HI = 0x20f1
# 	ADDR_HI 0 15
regCP_PFP_PRGRM_CNTR_START = 0x1e44
# 	IP_START 0 31
regCP_PFP_PRGRM_CNTR_START_HI = 0x1e59
# 	IP_START 0 29
regCP_PFP_SDMA_CS = 0x1f4f
# 	REQUEST_GRANT 0 0
# 	SDMA_ID 4 7
# 	REQUEST_POSITION 8 11
# 	SDMA_COUNT 12 13
regCP_PFP_UCODE_ADDR = 0x5814
# 	UCODE_ADDR 0 19
regCP_PFP_UCODE_DATA = 0x5815
# 	UCODE_DATA 0 31
regCP_PIPEID = 0xd9
# 	PIPE_ID 0 1
regCP_PIPE_STATS_ADDR_HI = 0x2019
# 	PIPE_STATS_ADDR_HI 0 15
regCP_PIPE_STATS_ADDR_LO = 0x2018
# 	PIPE_STATS_ADDR_LO 2 31
regCP_PIPE_STATS_CONTROL = 0x203d
# 	CACHE_POLICY 25 26
regCP_PQ_STATUS = 0x1e58
# 	DOORBELL_UPDATED 0 0
# 	DOORBELL_ENABLE 1 1
# 	DOORBELL_UPDATED_EN 2 2
# 	DOORBELL_UPDATED_MODE 3 3
regCP_PQ_WPTR_POLL_CNTL = 0x1e23
# 	PERIOD 0 7
# 	DISABLE_PEND_REQ_ONE_SHOT 29 29
# 	POLL_ACTIVE 30 30
# 	EN 31 31
regCP_PQ_WPTR_POLL_CNTL1 = 0x1e24
# 	QUEUE_MASK 0 31
regCP_PRED_NOT_VISIBLE = 0x20ee
# 	NOT_VISIBLE 0 0
regCP_PRIV_VIOLATION_ADDR = 0xf9a
# 	PRIV_VIOLATION_ADDR 0 17
regCP_PROCESS_QUANTUM = 0x1df9
# 	QUANTUM_DURATION 0 27
# 	TIMER_EXPIRED 28 28
# 	QUANTUM_SCALE 29 30
# 	QUANTUM_EN 31 31
regCP_PWR_CNTL = 0x1e18
# 	GFX_CLK_HALT_ME0_PIPE0 0 0
# 	GFX_CLK_HALT_ME0_PIPE1 1 1
# 	CMP_CLK_HALT_ME1_PIPE0 8 8
# 	CMP_CLK_HALT_ME1_PIPE1 9 9
# 	CMP_CLK_HALT_ME1_PIPE2 10 10
# 	CMP_CLK_HALT_ME1_PIPE3 11 11
# 	CMP_CLK_HALT_ME2_PIPE0 16 16
# 	CMP_CLK_HALT_ME2_PIPE1 17 17
# 	CMP_CLK_HALT_ME2_PIPE2 18 18
# 	CMP_CLK_HALT_ME2_PIPE3 19 19
# 	CMP_CLK_HALT_ME3_PIPE0 20 20
# 	CMP_CLK_HALT_ME3_PIPE1 21 21
# 	CMP_CLK_HALT_ME3_PIPE2 22 22
# 	CMP_CLK_HALT_ME3_PIPE3 23 23
regCP_RB0_ACTIVE = 0x1f40
# 	ACTIVE 0 0
regCP_RB0_BASE = 0x1de0
# 	RB_BASE 0 31
regCP_RB0_BASE_HI = 0x1e51
# 	RB_BASE_HI 0 7
regCP_RB0_BUFSZ_MASK = 0x1de5
# 	DATA 0 19
regCP_RB0_CNTL = 0x1de1
# 	RB_BUFSZ 0 5
# 	TMZ_STATE 6 6
# 	TMZ_MATCH 7 7
# 	RB_BLKSZ 8 13
# 	RB_NON_PRIV 15 15
# 	MIN_AVAILSZ 20 21
# 	MIN_IB_AVAILSZ 22 23
# 	CACHE_POLICY 24 25
# 	RB_VOLATILE 26 26
# 	RB_NO_UPDATE 27 27
# 	RB_EXE 28 28
# 	KMD_QUEUE 29 29
# 	RB_RPTR_WR_ENA 31 31
regCP_RB0_RPTR = 0xf60
# 	RB_RPTR 0 19
regCP_RB0_RPTR_ADDR = 0x1de3
# 	RB_RPTR_ADDR 2 31
regCP_RB0_RPTR_ADDR_HI = 0x1de4
# 	RB_RPTR_ADDR_HI 0 15
regCP_RB0_WPTR = 0x1df4
# 	RB_WPTR 0 31
regCP_RB0_WPTR_HI = 0x1df5
# 	RB_WPTR 0 31
regCP_RB1_ACTIVE = 0x1f41
# 	ACTIVE 0 0
regCP_RB1_BASE = 0x1e00
# 	RB_BASE 0 31
regCP_RB1_BASE_HI = 0x1e52
# 	RB_BASE_HI 0 7
regCP_RB1_BUFSZ_MASK = 0x1e04
# 	DATA 0 19
regCP_RB1_CNTL = 0x1e01
# 	RB_BUFSZ 0 5
# 	TMZ_STATE 6 6
# 	TMZ_MATCH 7 7
# 	RB_BLKSZ 8 13
# 	RB_NON_PRIV 15 15
# 	MIN_AVAILSZ 20 21
# 	MIN_IB_AVAILSZ 22 23
# 	CACHE_POLICY 24 25
# 	RB_VOLATILE 26 26
# 	RB_NO_UPDATE 27 27
# 	RB_EXE 28 28
# 	KMD_QUEUE 29 29
# 	RB_RPTR_WR_ENA 31 31
regCP_RB1_RPTR = 0xf5f
# 	RB_RPTR 0 19
regCP_RB1_RPTR_ADDR = 0x1e02
# 	RB_RPTR_ADDR 2 31
regCP_RB1_RPTR_ADDR_HI = 0x1e03
# 	RB_RPTR_ADDR_HI 0 15
regCP_RB1_WPTR = 0x1df6
# 	RB_WPTR 0 31
regCP_RB1_WPTR_HI = 0x1df7
# 	RB_WPTR 0 31
regCP_RB_ACTIVE = 0x1f40
# 	ACTIVE 0 0
regCP_RB_BASE = 0x1de0
# 	RB_BASE 0 31
regCP_RB_BUFSZ_MASK = 0x1de5
# 	DATA 0 19
regCP_RB_CNTL = 0x1de1
# 	RB_BUFSZ 0 5
# 	TMZ_STATE 6 6
# 	TMZ_MATCH 7 7
# 	RB_BLKSZ 8 13
# 	RB_NON_PRIV 15 15
# 	MIN_AVAILSZ 20 21
# 	MIN_IB_AVAILSZ 22 23
# 	CACHE_POLICY 24 25
# 	RB_VOLATILE 26 26
# 	RB_NO_UPDATE 27 27
# 	RB_EXE 28 28
# 	KMD_QUEUE 29 29
# 	RB_RPTR_WR_ENA 31 31
regCP_RB_DOORBELL_CLEAR = 0x1f28
# 	MAPPED_QUEUE 0 2
# 	MAPPED_QUE_DOORBELL_EN_CLEAR 8 8
# 	MAPPED_QUE_DOORBELL_HIT_CLEAR 9 9
# 	MASTER_DOORBELL_EN_CLEAR 10 10
# 	MASTER_DOORBELL_HIT_CLEAR 11 11
# 	QUEUES_DOORBELL_EN_CLEAR 12 12
# 	QUEUES_DOORBELL_HIT_CLEAR 13 13
regCP_RB_DOORBELL_CONTROL = 0x1e8d
# 	DOORBELL_BIF_DROP 1 1
# 	DOORBELL_OFFSET 2 27
# 	DOORBELL_EN 30 30
# 	DOORBELL_HIT 31 31
regCP_RB_DOORBELL_RANGE_LOWER = 0x1dfa
# 	DOORBELL_RANGE_LOWER 2 11
regCP_RB_DOORBELL_RANGE_UPPER = 0x1dfb
# 	DOORBELL_RANGE_UPPER 2 11
regCP_RB_OFFSET = 0x2091
# 	RB_OFFSET 0 19
regCP_RB_RPTR = 0xf60
# 	RB_RPTR 0 19
regCP_RB_RPTR_ADDR = 0x1de3
# 	RB_RPTR_ADDR 2 31
regCP_RB_RPTR_ADDR_HI = 0x1de4
# 	RB_RPTR_ADDR_HI 0 15
regCP_RB_RPTR_WR = 0x1de2
# 	RB_RPTR_WR 0 19
regCP_RB_STATUS = 0x1f43
# 	DOORBELL_UPDATED 0 0
# 	DOORBELL_ENABLE 1 1
regCP_RB_VMID = 0x1df1
# 	RB0_VMID 0 3
# 	RB1_VMID 8 11
# 	RB2_VMID 16 19
regCP_RB_WPTR = 0x1df4
# 	RB_WPTR 0 31
regCP_RB_WPTR_DELAY = 0xf61
# 	PRE_WRITE_TIMER 0 27
# 	PRE_WRITE_LIMIT 28 31
regCP_RB_WPTR_HI = 0x1df5
# 	RB_WPTR 0 31
regCP_RB_WPTR_POLL_ADDR_HI = 0x1e8c
# 	RB_WPTR_POLL_ADDR_HI 0 15
regCP_RB_WPTR_POLL_ADDR_LO = 0x1e8b
# 	RB_WPTR_POLL_ADDR_LO 2 31
regCP_RB_WPTR_POLL_CNTL = 0xf62
# 	POLL_FREQUENCY 0 15
# 	IDLE_POLL_COUNT 16 31
regCP_RING0_PRIORITY = 0x1ded
# 	PRIORITY 0 1
regCP_RING1_PRIORITY = 0x1dee
# 	PRIORITY 0 1
regCP_RINGID = 0xd9
# 	RINGID 0 1
regCP_RING_PRIORITY_CNTS = 0x1dec
# 	PRIORITY1_CNT 0 7
# 	PRIORITY2A_CNT 8 15
# 	PRIORITY2B_CNT 16 23
# 	PRIORITY3_CNT 24 31
regCP_ROQ1_THRESHOLDS = 0xf75
# 	RB1_START 0 9
# 	R0_IB1_START 10 19
# 	R1_IB1_START 20 29
regCP_ROQ2_AVAIL = 0xf7c
# 	ROQ_CNT_IB2 0 11
# 	ROQ_CNT_DB 16 27
regCP_ROQ2_THRESHOLDS = 0xf76
# 	R0_IB2_START 0 9
# 	R1_IB2_START 10 19
regCP_ROQ3_THRESHOLDS = 0xf8c
# 	R0_DB_START 0 9
# 	R1_DB_START 10 19
regCP_ROQ_AVAIL = 0xf7a
# 	ROQ_CNT_RING 0 11
# 	ROQ_CNT_IB1 16 27
regCP_ROQ_DB_STAT = 0xf8d
# 	ROQ_RPTR_DB 0 11
# 	ROQ_WPTR_DB 16 27
regCP_ROQ_IB1_STAT = 0xf81
# 	ROQ_RPTR_INDIRECT1 0 11
# 	ROQ_WPTR_INDIRECT1 16 27
regCP_ROQ_IB2_STAT = 0xf82
# 	ROQ_RPTR_INDIRECT2 0 11
# 	ROQ_WPTR_INDIRECT2 16 27
regCP_ROQ_RB_STAT = 0xf80
# 	ROQ_RPTR_PRIMARY 0 11
# 	ROQ_WPTR_PRIMARY 16 27
regCP_SAMPLE_STATUS = 0x20fd
# 	Z_PASS_ACITVE 0 0
# 	STREAMOUT_ACTIVE 1 1
# 	PIPELINE_ACTIVE 2 2
# 	STIPPLE_ACTIVE 3 3
# 	VGT_BUFFERS_ACTIVE 4 4
# 	SCREEN_EXT_ACTIVE 5 5
# 	DRAW_INDIRECT_ACTIVE 6 6
# 	DISP_INDIRECT_ACTIVE 7 7
regCP_SCRATCH_DATA = 0x2090
# 	SCRATCH_DATA 0 31
regCP_SCRATCH_INDEX = 0x208f
# 	SCRATCH_INDEX 0 8
# 	SCRATCH_INDEX_64BIT_MODE 31 31
regCP_SC_PSINVOC_COUNT0_HI = 0x202d
# 	PSINVOC_COUNT0_HI 0 31
regCP_SC_PSINVOC_COUNT0_LO = 0x202c
# 	PSINVOC_COUNT0_LO 0 31
regCP_SC_PSINVOC_COUNT1_HI = 0x202f
# 	OBSOLETE 0 31
regCP_SC_PSINVOC_COUNT1_LO = 0x202e
# 	OBSOLETE 0 31
regCP_SDMA_DMA_DONE = 0x1f4e
# 	SDMA_ID 0 3
regCP_SD_CNTL = 0x1f57
# 	CPF_EN 0 0
# 	CPG_EN 1 1
# 	CPC_EN 2 2
# 	RLC_EN 3 3
# 	GE_EN 5 5
# 	UTCL1_EN 6 6
# 	EA_EN 9 9
# 	SDMA_EN 10 10
# 	SD_VMIDVEC_OVERRIDE 31 31
regCP_SEM_WAIT_TIMER = 0x206f
# 	SEM_WAIT_TIMER 0 31
regCP_SIG_SEM_ADDR_HI = 0x2071
# 	SEM_ADDR_HI 0 15
# 	SEM_USE_MAILBOX 16 16
# 	SEM_SIGNAL_TYPE 20 20
# 	SEM_CLIENT_CODE 24 25
# 	SEM_SELECT 29 31
regCP_SIG_SEM_ADDR_LO = 0x2070
# 	SEM_PRIV 0 0
# 	SEM_ADDR_LO 3 31
regCP_SOFT_RESET_CNTL = 0x1f59
# 	CMP_ONLY_SOFT_RESET 0 0
# 	GFX_ONLY_SOFT_RESET 1 1
# 	CMP_HQD_REG_RESET 2 2
# 	CMP_INTR_REG_RESET 3 3
# 	CMP_HQD_QUEUE_DOORBELL_RESET 4 4
# 	GFX_RB_DOORBELL_RESET 5 5
# 	GFX_INTR_REG_RESET 6 6
# 	GFX_HQD_REG_RESET 7 7
regCP_STALLED_STAT1 = 0xf3d
# 	RBIU_TO_DMA_NOT_RDY_TO_RCV 0 0
# 	RBIU_TO_SEM_NOT_RDY_TO_RCV_R0 2 2
# 	RBIU_TO_SEM_NOT_RDY_TO_RCV_R1 3 3
# 	RBIU_TO_MEMWR_NOT_RDY_TO_RCV_R0 4 4
# 	RBIU_TO_MEMWR_NOT_RDY_TO_RCV_R1 5 5
# 	ME_HAS_ACTIVE_CE_BUFFER_FLAG 10 10
# 	ME_HAS_ACTIVE_DE_BUFFER_FLAG 11 11
# 	ME_STALLED_ON_TC_WR_CONFIRM 12 12
# 	ME_STALLED_ON_ATOMIC_RTN_DATA 13 13
# 	ME_WAITING_ON_TC_READ_DATA 14 14
# 	ME_WAITING_ON_REG_READ_DATA 15 15
# 	RCIU_WAITING_ON_GDS_FREE 23 23
# 	RCIU_WAITING_ON_GRBM_FREE 24 24
# 	RCIU_WAITING_ON_VGT_FREE 25 25
# 	RCIU_STALLED_ON_ME_READ 26 26
# 	RCIU_STALLED_ON_DMA_READ 27 27
# 	RCIU_STALLED_ON_APPEND_READ 28 28
# 	RCIU_HALTED_BY_REG_VIOLATION 29 29
regCP_STALLED_STAT2 = 0xf3e
# 	PFP_TO_CSF_NOT_RDY_TO_RCV 0 0
# 	PFP_TO_MEQ_NOT_RDY_TO_RCV 1 1
# 	PFP_TO_RCIU_NOT_RDY_TO_RCV 2 2
# 	PFP_TO_VGT_WRITES_PENDING 4 4
# 	PFP_RCIU_READ_PENDING 5 5
# 	PFP_TO_MEQ_DDID_NOT_RDY_TO_RCV 6 6
# 	PFP_WAITING_ON_BUFFER_DATA 8 8
# 	ME_WAIT_ON_CE_COUNTER 9 9
# 	ME_WAIT_ON_AVAIL_BUFFER 10 10
# 	GFX_CNTX_NOT_AVAIL_TO_ME 11 11
# 	ME_RCIU_NOT_RDY_TO_RCV 12 12
# 	ME_TO_CONST_NOT_RDY_TO_RCV 13 13
# 	ME_WAITING_DATA_FROM_PFP 14 14
# 	ME_WAITING_ON_PARTIAL_FLUSH 15 15
# 	MEQ_TO_ME_NOT_RDY_TO_RCV 16 16
# 	STQ_TO_ME_NOT_RDY_TO_RCV 17 17
# 	ME_WAITING_DATA_FROM_STQ 18 18
# 	PFP_STALLED_ON_TC_WR_CONFIRM 19 19
# 	PFP_STALLED_ON_ATOMIC_RTN_DATA 20 20
# 	QU_STALLED_ON_EOP_DONE_PULSE 21 21
# 	QU_STALLED_ON_EOP_DONE_WR_CONFIRM 22 22
# 	STRMO_WR_OF_PRIM_DATA_PENDING 23 23
# 	PIPE_STATS_WR_DATA_PENDING 24 24
# 	APPEND_RDY_WAIT_ON_CS_DONE 25 25
# 	APPEND_RDY_WAIT_ON_PS_DONE 26 26
# 	APPEND_WAIT_ON_WR_CONFIRM 27 27
# 	APPEND_ACTIVE_PARTITION 28 28
# 	APPEND_WAITING_TO_SEND_MEMWRITE 29 29
# 	SURF_SYNC_NEEDS_IDLE_CNTXS 30 30
# 	SURF_SYNC_NEEDS_ALL_CLEAN 31 31
regCP_STALLED_STAT3 = 0xf3c
# 	CE_TO_CSF_NOT_RDY_TO_RCV 0 0
# 	CE_TO_RAM_INIT_FETCHER_NOT_RDY_TO_RCV 1 1
# 	CE_WAITING_ON_DATA_FROM_RAM_INIT_FETCHER 2 2
# 	CE_TO_RAM_INIT_NOT_RDY 3 3
# 	CE_TO_RAM_DUMP_NOT_RDY 4 4
# 	CE_TO_RAM_WRITE_NOT_RDY 5 5
# 	CE_TO_INC_FIFO_NOT_RDY_TO_RCV 6 6
# 	CE_TO_WR_FIFO_NOT_RDY_TO_RCV 7 7
# 	CE_WAITING_ON_BUFFER_DATA 10 10
# 	CE_WAITING_ON_CE_BUFFER_FLAG 11 11
# 	CE_WAITING_ON_DE_COUNTER 12 12
# 	CE_WAITING_ON_DE_COUNTER_UNDERFLOW 13 13
# 	TCIU_WAITING_ON_FREE 14 14
# 	TCIU_WAITING_ON_TAGS 15 15
# 	CE_STALLED_ON_TC_WR_CONFIRM 16 16
# 	CE_STALLED_ON_ATOMIC_RTN_DATA 17 17
# 	UTCL2IU_WAITING_ON_FREE 18 18
# 	UTCL2IU_WAITING_ON_TAGS 19 19
# 	UTCL1_WAITING_ON_TRANS 20 20
# 	GCRIU_WAITING_ON_FREE 21 21
regCP_STAT = 0xf40
# 	ROQ_DB_BUSY 5 5
# 	ROQ_CE_DB_BUSY 6 6
# 	ROQ_RING_BUSY 9 9
# 	ROQ_INDIRECT1_BUSY 10 10
# 	ROQ_INDIRECT2_BUSY 11 11
# 	ROQ_STATE_BUSY 12 12
# 	DC_BUSY 13 13
# 	UTCL2IU_BUSY 14 14
# 	PFP_BUSY 15 15
# 	MEQ_BUSY 16 16
# 	ME_BUSY 17 17
# 	QUERY_BUSY 18 18
# 	SEMAPHORE_BUSY 19 19
# 	INTERRUPT_BUSY 20 20
# 	SURFACE_SYNC_BUSY 21 21
# 	DMA_BUSY 22 22
# 	RCIU_BUSY 23 23
# 	SCRATCH_RAM_BUSY 24 24
# 	GCRIU_BUSY 25 25
# 	CE_BUSY 26 26
# 	TCIU_BUSY 27 27
# 	ROQ_CE_RING_BUSY 28 28
# 	ROQ_CE_INDIRECT1_BUSY 29 29
# 	ROQ_CE_INDIRECT2_BUSY 30 30
# 	CP_BUSY 31 31
regCP_STQ_AVAIL = 0xf7b
# 	STQ_CNT 0 8
regCP_STQ_STAT = 0xf83
# 	STQ_RPTR 0 9
regCP_STQ_THRESHOLDS = 0xf77
# 	STQ0_START 0 7
# 	STQ1_START 8 15
# 	STQ2_START 16 23
regCP_STQ_WR_STAT = 0xf84
# 	STQ_WPTR 0 9
regCP_ST_BASE_HI = 0x20d3
# 	ST_BASE_HI 0 15
regCP_ST_BASE_LO = 0x20d2
# 	ST_BASE_LO 2 31
regCP_ST_BUFSZ = 0x20d4
# 	ST_BUFSZ 0 19
regCP_ST_CMD_BUFSZ = 0x20c2
# 	ST_CMD_REQSZ 0 19
regCP_SUSPEND_CNTL = 0x1e69
# 	SUSPEND_MODE 0 0
# 	SUSPEND_ENABLE 1 1
# 	RESUME_LOCK 2 2
# 	ACE_SUSPEND_ACTIVE 3 3
regCP_SUSPEND_RESUME_REQ = 0x1e68
# 	SUSPEND_REQ 0 0
# 	RESUME_REQ 1 1
regCP_VGT_ASINVOC_COUNT_HI = 0x2033
# 	ASINVOC_COUNT_HI 0 31
regCP_VGT_ASINVOC_COUNT_LO = 0x2032
# 	ASINVOC_COUNT_LO 0 31
regCP_VGT_CSINVOC_COUNT_HI = 0x2031
# 	CSINVOC_COUNT_HI 0 31
regCP_VGT_CSINVOC_COUNT_LO = 0x2030
# 	CSINVOC_COUNT_LO 0 31
regCP_VGT_DSINVOC_COUNT_HI = 0x2027
# 	DSINVOC_COUNT_HI 0 31
regCP_VGT_DSINVOC_COUNT_LO = 0x2026
# 	DSINVOC_COUNT_LO 0 31
regCP_VGT_GSINVOC_COUNT_HI = 0x2023
# 	GSINVOC_COUNT_HI 0 31
regCP_VGT_GSINVOC_COUNT_LO = 0x2022
# 	GSINVOC_COUNT_LO 0 31
regCP_VGT_GSPRIM_COUNT_HI = 0x201f
# 	GSPRIM_COUNT_HI 0 31
regCP_VGT_GSPRIM_COUNT_LO = 0x201e
# 	GSPRIM_COUNT_LO 0 31
regCP_VGT_HSINVOC_COUNT_HI = 0x2025
# 	HSINVOC_COUNT_HI 0 31
regCP_VGT_HSINVOC_COUNT_LO = 0x2024
# 	HSINVOC_COUNT_LO 0 31
regCP_VGT_IAPRIM_COUNT_HI = 0x201d
# 	IAPRIM_COUNT_HI 0 31
regCP_VGT_IAPRIM_COUNT_LO = 0x201c
# 	IAPRIM_COUNT_LO 0 31
regCP_VGT_IAVERT_COUNT_HI = 0x201b
# 	IAVERT_COUNT_HI 0 31
regCP_VGT_IAVERT_COUNT_LO = 0x201a
# 	IAVERT_COUNT_LO 0 31
regCP_VGT_VSINVOC_COUNT_HI = 0x2021
# 	VSINVOC_COUNT_HI 0 31
regCP_VGT_VSINVOC_COUNT_LO = 0x2020
# 	VSINVOC_COUNT_LO 0 31
regCP_VIRT_STATUS = 0x1dd8
# 	VIRT_STATUS 0 31
regCP_VMID = 0xda
# 	VMID 0 3
regCP_VMID_PREEMPT = 0x1e56
# 	PREEMPT_REQUEST 0 15
# 	VIRT_COMMAND 16 19
regCP_VMID_RESET = 0x1e53
# 	RESET_REQUEST 0 15
# 	PIPE0_QUEUES 16 23
# 	PIPE1_QUEUES 24 31
regCP_VMID_STATUS = 0x1e5f
# 	PREEMPT_DE_STATUS 0 15
# 	PREEMPT_CE_STATUS 16 31
regCP_WAIT_REG_MEM_TIMEOUT = 0x2074
# 	WAIT_REG_MEM_TIMEOUT 0 31
regCP_WAIT_SEM_ADDR_HI = 0x2076
# 	SEM_ADDR_HI 0 15
# 	SEM_USE_MAILBOX 16 16
# 	SEM_SIGNAL_TYPE 20 20
# 	SEM_CLIENT_CODE 24 25
# 	SEM_SELECT 29 31
regCP_WAIT_SEM_ADDR_LO = 0x2075
# 	SEM_PRIV 0 0
# 	SEM_ADDR_LO 3 31
regDB_ALPHA_TO_MASK = 0x2dc
# 	ALPHA_TO_MASK_ENABLE 0 0
# 	ALPHA_TO_MASK_OFFSET0 8 9
# 	ALPHA_TO_MASK_OFFSET1 10 11
# 	ALPHA_TO_MASK_OFFSET2 12 13
# 	ALPHA_TO_MASK_OFFSET3 14 15
# 	OFFSET_ROUND 16 16
regDB_CGTT_CLK_CTRL_0 = 0x50a4
# 	SOFT_OVERRIDE0 0 0
# 	SOFT_OVERRIDE1 1 1
# 	SOFT_OVERRIDE2 2 2
# 	SOFT_OVERRIDE3 3 3
# 	SOFT_OVERRIDE4 4 4
# 	SOFT_OVERRIDE5 5 5
# 	SOFT_OVERRIDE6 6 6
# 	SOFT_OVERRIDE7 7 7
# 	SOFT_OVERRIDE8 8 8
# 	RESERVED 9 31
regDB_COUNT_CONTROL = 0x1
# 	PERFECT_ZPASS_COUNTS 1 1
# 	DISABLE_CONSERVATIVE_ZPASS_COUNTS 2 2
# 	ENHANCED_CONSERVATIVE_ZPASS_COUNTS 3 3
# 	SAMPLE_RATE 4 6
# 	ZPASS_ENABLE 8 11
# 	ZFAIL_ENABLE 12 15
# 	SFAIL_ENABLE 16 19
# 	DBFAIL_ENABLE 20 23
# 	SLICE_EVEN_ENABLE 24 27
# 	SLICE_ODD_ENABLE 28 31
regDB_CREDIT_LIMIT = 0x13b4
# 	DB_SC_TILE_CREDITS 0 4
# 	DB_SC_QUAD_CREDITS 5 9
# 	DB_CB_LQUAD_CREDITS 10 12
# 	DB_SC_WAVE_CREDITS 13 17
# 	DB_SC_FREE_WAVE_CREDITS 18 22
regDB_DEBUG = 0x13ac
# 	DEBUG_STENCIL_COMPRESS_DISABLE 0 0
# 	DEBUG_DEPTH_COMPRESS_DISABLE 1 1
# 	FETCH_FULL_Z_TILE 2 2
# 	FETCH_FULL_STENCIL_TILE 3 3
# 	FORCE_Z_MODE 4 5
# 	DEBUG_FORCE_DEPTH_READ 6 6
# 	DEBUG_FORCE_STENCIL_READ 7 7
# 	DEBUG_FORCE_HIZ_ENABLE 8 9
# 	DEBUG_FORCE_HIS_ENABLE0 10 11
# 	DEBUG_FORCE_HIS_ENABLE1 12 13
# 	DEBUG_FAST_Z_DISABLE 14 14
# 	DEBUG_FAST_STENCIL_DISABLE 15 15
# 	DEBUG_NOOP_CULL_DISABLE 16 16
# 	DISABLE_SUMM_SQUADS 17 17
# 	DEPTH_CACHE_FORCE_MISS 18 18
# 	DEBUG_FORCE_FULL_Z_RANGE 19 20
# 	NEVER_FREE_Z_ONLY 21 21
# 	ZPASS_COUNTS_LOOK_AT_PIPE_STAT_EVENTS 22 22
# 	DISABLE_VPORT_ZPLANE_OPTIMIZATION 23 23
# 	DECOMPRESS_AFTER_N_ZPLANES 24 27
# 	ONE_FREE_IN_FLIGHT 28 28
# 	FORCE_MISS_IF_NOT_INFLIGHT 29 29
# 	DISABLE_DEPTH_SURFACE_SYNC 30 30
# 	DISABLE_HTILE_SURFACE_SYNC 31 31
regDB_DEBUG2 = 0x13ad
# 	ALLOW_COMPZ_BYTE_MASKING 0 0
# 	DISABLE_TC_ZRANGE_L0_CACHE 1 1
# 	DISABLE_TC_MASK_L0_CACHE 2 2
# 	DTR_ROUND_ROBIN_ARB 3 3
# 	DTR_PREZ_STALLS_FOR_ETF_ROOM 4 4
# 	DISABLE_PREZL_FIFO_STALL 5 5
# 	DISABLE_PREZL_FIFO_STALL_REZ 6 6
# 	ENABLE_VIEWPORT_STALL_ON_ALL 7 7
# 	OPTIMIZE_HIZ_MATCHES_FB_DISABLE 8 8
# 	CLK_OFF_DELAY 9 13
# 	FORCE_PERF_COUNTERS_ON 14 14
# 	FULL_TILE_CACHE_EVICT_ON_HALF_FULL 15 15
# 	DISABLE_HTILE_PAIRED_PIPES 16 16
# 	DISABLE_NULL_EOT_FORWARDING 17 17
# 	DISABLE_DTT_DATA_FORWARDING 18 18
# 	DISABLE_QUAD_COHERENCY_STALL 19 19
# 	DISABLE_FULL_TILE_WAVE_BREAK 20 20
# 	ENABLE_FULL_TILE_WAVE_BREAK_FOR_ALL_TILES 21 21
# 	FORCE_ITERATE_256 24 25
# 	RESERVED1 26 26
# 	DEBUG_BUS_FLOP_EN 27 27
# 	ENABLE_PREZ_OF_REZ_SUMM 28 28
# 	DISABLE_PREZL_VIEWPORT_STALL 29 29
# 	DISABLE_SINGLE_STENCIL_QUAD_SUMM 30 30
# 	DISABLE_WRITE_STALL_ON_RDWR_CONFLICT 31 31
regDB_DEBUG3 = 0x13ae
# 	DISABLE_CLEAR_ZRANGE_CORRECTION 0 0
# 	DISABLE_RELOAD_CONTEXT_DRAW_DATA 1 1
# 	FORCE_DB_IS_GOOD 2 2
# 	DISABLE_TL_SSO_NULL_SUPPRESSION 3 3
# 	DISABLE_HIZ_ON_VPORT_CLAMP 4 4
# 	EQAA_INTERPOLATE_COMP_Z 5 5
# 	EQAA_INTERPOLATE_SRC_Z 6 6
# 	DISABLE_ZCMP_DIRTY_SUPPRESSION 8 8
# 	DISABLE_RECOMP_TO_1ZPLANE_WITHOUT_FASTOP 10 10
# 	ENABLE_INCOHERENT_EQAA_READS 11 11
# 	DISABLE_OP_DF_BYPASS 13 13
# 	DISABLE_OP_DF_WRITE_COMBINE 14 14
# 	DISABLE_OP_DF_DIRECT_FEEDBACK 15 15
# 	DISABLE_SLOCS_PER_CTXT_MATCH 16 16
# 	SLOW_PREZ_TO_A2M_OMASK_RATE 17 17
# 	DISABLE_TC_UPDATE_WRITE_COMBINE 19 19
# 	DISABLE_HZ_TC_WRITE_COMBINE 20 20
# 	ENABLE_RECOMP_ZDIRTY_SUPPRESSION_OPT 21 21
# 	ENABLE_TC_MA_ROUND_ROBIN_ARB 22 22
# 	DISABLE_RAM_READ_SUPPRESION_ON_FWD 23 23
# 	DISABLE_EQAA_A2M_PERF_OPT 24 24
# 	DISABLE_DI_DT_STALL 25 25
# 	ENABLE_DB_PROCESS_RESET 26 26
# 	DISABLE_OVERRASTERIZATION_FIX 27 27
# 	DONT_INSERT_CONTEXT_SUSPEND 28 28
# 	DELETE_CONTEXT_SUSPEND 29 29
# 	DISABLE_TS_WRITE_L0 30 30
# 	DISABLE_MULTIDTAG_FL_PANIC_REQUIREMENT 31 31
regDB_DEBUG4 = 0x13af
# 	DISABLE_QC_Z_MASK_SUMMATION 0 0
# 	DISABLE_QC_STENCIL_MASK_SUMMATION 1 1
# 	DISABLE_RESUMM_TO_SINGLE_STENCIL 2 2
# 	DISABLE_PREZ_POSTZ_DTILE_CONFLICT_STALL 3 3
# 	DISABLE_SEPARATE_OP_PIPE_CLK 4 4
# 	DISABLE_SEPARATE_SX_CLK 5 5
# 	ALWAYS_ON_RMI_CLK_EN 6 6
# 	ENABLE_DBCB_SLOW_FORMAT_COLLAPSE 7 7
# 	DISABLE_SEPARATE_DBG_CLK 8 8
# 	DISABLE_UNMAPPED_Z_INDICATOR 9 9
# 	DISABLE_UNMAPPED_S_INDICATOR 10 10
# 	DISABLE_UNMAPPED_H_INDICATOR 11 11
# 	ENABLE_A2M_DQUAD_OPTIMIZATION 12 12
# 	DISABLE_DTT_FAST_HTILENACK_LOOKUP 13 13
# 	DISABLE_RESCHECK_MEMCOHER_OPTIMIZATION 14 14
# 	DISABLE_DYNAMIC_RAM_LIGHT_SLEEP_MODE 15 15
# 	DISABLE_HIZ_TS_COLLISION_DETECT 16 16
# 	DISABLE_LAST_OF_BURST_ON_FLUSH_CHUNK0_ALL_DONE 18 18
# 	ENABLE_CZ_OVERFLOW_TESTMODE 19 19
# 	DISABLE_MCC_BURST_FIFO 21 21
# 	DISABLE_MCC_BURST_FIFO_CONFLICT 22 22
# 	WR_MEM_BURST_CTL 24 26
# 	DISABLE_WR_MEM_BURST_POOLING 27 27
# 	DISABLE_RD_MEM_BURST 28 28
# 	LATE_ACK_SCOREBOARD_MULTIPLE_SLOT 30 30
# 	LATE_ACK_PSD_EOP_OLD_METHOD 31 31
regDB_DEBUG5 = 0x13d1
# 	DISABLE_TILE_CACHE_PRELOAD 0 0
# 	ENABLE_SECONDARY_MIPS_TAILS_COMPRESSION 1 1
# 	DISABLE_CLEAR_VALUE_UPDATE_ON_TILE_CACHE_HIT 2 2
# 	DISABLE_2SRC_VRS_HARD_CONFLICT 3 3
# 	DISABLE_FLQ_MCC_DTILEID_CHECK 4 4
# 	DISABLE_NOZ_POWER_SAVINGS 5 5
# 	DISABLE_TILE_INFLIGHT_DEC_POSTZ_FIX 6 6
# 	DISABLE_MGCG_GATING_ON_SHADER_WAIT 7 7
# 	DISABLE_VRS_1X2_2XAA 8 8
# 	ENABLE_FULL_TILE_WAVE_BREAK_ON_COARSE 9 9
# 	DISABLE_HTILE_HARVESTING 10 10
# 	DISABLE_SEPARATE_TILE_CLK 11 11
# 	DISABLE_TILE_CACHE_PREFETCH 12 12
# 	DISABLE_PSL_AUTO_MODE_FIX 13 13
# 	DISABLE_FORCE_ZMASK_EXPANDED 14 14
# 	DISABLE_SEPARATE_LQO_CLK 15 15
# 	DISABLE_Z_WITHOUT_PLANES_FLQ 16 16
# 	PRESERVE_QMASK_FOR_POSTZ_OP_PIPE 17 17
# 	Z_NACK_BEHAVIOR_ONLY_WHEN_Z_IS_PRT 18 18
# 	S_NACK_BEHAVIOR_ONLY_WHEN_S_IS_PRT 19 19
# 	DISABLE_RESIDENCY_CHECK_Z 20 20
# 	DISABLE_RESIDENCY_CHECK_STENCIL 21 21
# 	DISABLE_LQO_FTCQ_DUAL_QUAD_REGION_CHECK 22 22
# 	DISABLE_EVENT_INSERTION_AFTER_ZPC_BEFORE_CONTEXT_DONE 23 23
# 	SPARE_BITS 24 31
regDB_DEBUG6 = 0x13be
# 	FORCE_DB_SC_WAVE_CONFLICT 0 0
# 	FORCE_DB_SC_WAVE_HARD_CONFLICT 1 1
# 	FORCE_DB_SC_QUAD_CONFLICT 2 2
# 	OREO_TRANSITION_EVENT_ALL 3 3
# 	OREO_TRANSITION_EVENT_ID 4 9
# 	OREO_TRANSITION_EVENT_EN 10 10
# 	DISABLE_PWS_PLUS_TCP_CM_LIVENESS_STALL 11 11
# 	DISABLE_PWS_PLUS_DTT_TAG_LIVENESS_STALL 12 12
# 	SET_DB_PERFMON_PWS_PIPE_ID 13 14
# 	FTWB_MAX_TIMEOUT_VAL 16 23
# 	DISABLE_LQO_SMT_RAM_OPT 24 24
# 	FORCE_MAX_TILES_IN_WAVE_CHECK 25 25
# 	DISABLE_OSB_DEADLOCK_FIX 26 26
# 	DISABLE_OSB_DEADLOCK_WAIT_PANIC 27 27
regDB_DEBUG7 = 0x13d0
# 	SPARE_BITS 0 31
regDB_DEPTH_BOUNDS_MAX = 0x9
# 	MAX 0 31
regDB_DEPTH_BOUNDS_MIN = 0x8
# 	MIN 0 31
regDB_DEPTH_CLEAR = 0xb
# 	DEPTH_CLEAR 0 31
regDB_DEPTH_CONTROL = 0x200
# 	STENCIL_ENABLE 0 0
# 	Z_ENABLE 1 1
# 	Z_WRITE_ENABLE 2 2
# 	DEPTH_BOUNDS_ENABLE 3 3
# 	ZFUNC 4 6
# 	BACKFACE_ENABLE 7 7
# 	STENCILFUNC 8 10
# 	STENCILFUNC_BF 20 22
# 	ENABLE_COLOR_WRITES_ON_DEPTH_FAIL 30 30
# 	DISABLE_COLOR_WRITES_ON_DEPTH_PASS 31 31
regDB_DEPTH_SIZE_XY = 0x7
# 	X_MAX 0 13
# 	Y_MAX 16 29
regDB_DEPTH_VIEW = 0x2
# 	SLICE_START 0 10
# 	SLICE_START_HI 11 12
# 	SLICE_MAX 13 23
# 	Z_READ_ONLY 24 24
# 	STENCIL_READ_ONLY 25 25
# 	MIPID 26 29
# 	SLICE_MAX_HI 30 31
regDB_EQAA = 0x201
# 	MAX_ANCHOR_SAMPLES 0 2
# 	PS_ITER_SAMPLES 4 6
# 	MASK_EXPORT_NUM_SAMPLES 8 10
# 	ALPHA_TO_MASK_NUM_SAMPLES 12 14
# 	HIGH_QUALITY_INTERSECTIONS 16 16
# 	INCOHERENT_EQAA_READS 17 17
# 	INTERPOLATE_COMP_Z 18 18
# 	INTERPOLATE_SRC_Z 19 19
# 	STATIC_ANCHOR_ASSOCIATIONS 20 20
# 	ALPHA_TO_MASK_EQAA_DISABLE 21 21
# 	OVERRASTERIZATION_AMOUNT 24 26
# 	ENABLE_POSTZ_OVERRASTERIZATION 27 27
regDB_EQUAD_STUTTER_CONTROL = 0x13b2
# 	THRESHOLD 0 7
# 	TIMEOUT 16 23
regDB_ETILE_STUTTER_CONTROL = 0x13b0
# 	THRESHOLD 0 7
# 	TIMEOUT 16 23
regDB_EXCEPTION_CONTROL = 0x13bf
# 	EARLY_Z_PANIC_DISABLE 0 0
# 	LATE_Z_PANIC_DISABLE 1 1
# 	RE_Z_PANIC_DISABLE 2 2
# 	AUTO_FLUSH_HTILE 3 3
# 	AUTO_FLUSH_QUAD 4 4
# 	FORCE_SUMMARIZE 8 11
# 	DTAG_WATERMARK 24 30
regDB_FGCG_INTERFACES_CLK_CTRL = 0x13d8
# 	DB_SC_QUAD_OVERRIDE 0 0
# 	DB_CB_EXPORT_OVERRIDE 2 2
# 	DB_RMI_RDREQ_OVERRIDE 3 3
# 	DB_RMI_WRREQ_OVERRIDE 4 4
# 	DB_SC_TILE_OVERRIDE 5 5
# 	DB_CB_RMIRET_OVERRIDE 6 6
# 	DB_SC_WAVE_OVERRIDE 7 7
# 	DB_SC_FREE_WAVE_OVERRIDE 8 8
regDB_FGCG_SRAMS_CLK_CTRL = 0x13d7
# 	OVERRIDE0 0 0
# 	OVERRIDE1 1 1
# 	OVERRIDE2 2 2
# 	OVERRIDE3 3 3
# 	OVERRIDE4 4 4
# 	OVERRIDE5 5 5
# 	OVERRIDE6 6 6
# 	OVERRIDE7 7 7
# 	OVERRIDE8 8 8
# 	OVERRIDE9 9 9
# 	OVERRIDE10 10 10
# 	OVERRIDE11 11 11
# 	OVERRIDE12 12 12
# 	OVERRIDE13 13 13
# 	OVERRIDE14 14 14
# 	OVERRIDE15 15 15
# 	OVERRIDE16 16 16
# 	OVERRIDE17 17 17
# 	OVERRIDE18 18 18
# 	OVERRIDE19 19 19
# 	OVERRIDE20 20 20
# 	OVERRIDE21 21 21
# 	OVERRIDE22 22 22
# 	OVERRIDE23 23 23
# 	OVERRIDE24 24 24
# 	OVERRIDE25 25 25
# 	OVERRIDE26 26 26
# 	OVERRIDE27 27 27
# 	OVERRIDE28 28 28
# 	OVERRIDE29 29 29
# 	OVERRIDE30 30 30
# 	OVERRIDE31 31 31
regDB_FIFO_DEPTH1 = 0x13b8
# 	MI_RDREQ_FIFO_DEPTH 0 7
# 	MI_WRREQ_FIFO_DEPTH 8 15
# 	MCC_DEPTH 16 23
# 	QC_DEPTH 24 31
regDB_FIFO_DEPTH2 = 0x13b9
# 	EQUAD_FIFO_DEPTH 0 7
# 	ETILE_OP_FIFO_DEPTH 8 15
# 	LQUAD_FIFO_DEPTH 16 24
# 	LTILE_OP_FIFO_DEPTH 25 31
regDB_FIFO_DEPTH3 = 0x13bd
# 	LTILE_PROBE_FIFO_DEPTH 0 7
# 	OSB_WAVE_TABLE_DEPTH 8 15
# 	OREO_WAVE_HIDE_DEPTH 16 23
# 	QUAD_READ_REQS 24 31
regDB_FIFO_DEPTH4 = 0x13d9
# 	OSB_SQUAD_TABLE_DEPTH 0 7
# 	OSB_TILE_TABLE_DEPTH 8 15
# 	OSB_SCORE_BOARD_DEPTH 16 23
# 	OSB_EVENT_FIFO_DEPTH 24 31
regDB_FREE_CACHELINES = 0x13b7
# 	FREE_DTILE_DEPTH 0 7
# 	FREE_PLANE_DEPTH 8 15
# 	FREE_Z_DEPTH 16 23
# 	FREE_HTILE_DEPTH 24 31
regDB_HTILE_DATA_BASE = 0x5
# 	BASE_256B 0 31
regDB_HTILE_DATA_BASE_HI = 0x1e
# 	BASE_HI 0 7
regDB_HTILE_SURFACE = 0x2af
# 	RESERVED_FIELD_1 0 0
# 	FULL_CACHE 1 1
# 	RESERVED_FIELD_2 2 2
# 	RESERVED_FIELD_3 3 3
# 	RESERVED_FIELD_4 4 9
# 	RESERVED_FIELD_5 10 15
# 	DST_OUTSIDE_ZERO_TO_ONE 16 16
# 	RESERVED_FIELD_6 17 17
# 	PIPE_ALIGNED 18 18
regDB_LAST_OF_BURST_CONFIG = 0x13ba
# 	MAXBURST 0 7
# 	TIMEOUT 8 10
# 	DBCB_LOB_SWITCH_TIMEOUT 11 15
# 	ENABLE_FG_DEFAULT_TIMEOUT 17 17
# 	DISABLE_MCC_BURST_COUNT_RESET_ON_LOB 18 18
# 	DISABLE_FLQ_LOB_EVERY_256B 19 19
# 	DISABLE_ZCACHE_FL_OP_EVEN_ARB 20 20
# 	DISABLE_MCC_BURST_FORCE_FLUSH_BEFORE_FIFO 21 21
# 	ENABLE_TIMEOUT_DKG_LOB_GEN 22 22
# 	ENABLE_TIMEOUT_LPF_LOB_GEN 23 23
# 	ENABLE_TIMEOUT_FL_BURST 25 25
# 	ENABLE_TIMEOUT_FG_LOB_FWDR 26 26
# 	BYPASS_SORT_RD_BA 28 28
# 	DISABLE_256B_COALESCE 29 29
# 	DISABLE_RD_BURST 30 30
# 	LEGACY_LOB_INSERT_EN 31 31
regDB_LQUAD_STUTTER_CONTROL = 0x13b3
# 	THRESHOLD 0 7
# 	TIMEOUT 16 23
regDB_LTILE_STUTTER_CONTROL = 0x13b1
# 	THRESHOLD 0 7
# 	TIMEOUT 16 23
regDB_MEM_ARB_WATERMARKS = 0x13bc
# 	CLIENT0_WATERMARK 0 2
# 	CLIENT1_WATERMARK 8 10
# 	CLIENT2_WATERMARK 16 18
# 	CLIENT3_WATERMARK 24 26
regDB_OCCLUSION_COUNT0_HI = 0x23c1
# 	COUNT_HI 0 30
regDB_OCCLUSION_COUNT0_LOW = 0x23c0
# 	COUNT_LOW 0 31
regDB_OCCLUSION_COUNT1_HI = 0x23c3
# 	COUNT_HI 0 30
regDB_OCCLUSION_COUNT1_LOW = 0x23c2
# 	COUNT_LOW 0 31
regDB_OCCLUSION_COUNT2_HI = 0x23c5
# 	COUNT_HI 0 30
regDB_OCCLUSION_COUNT2_LOW = 0x23c4
# 	COUNT_LOW 0 31
regDB_OCCLUSION_COUNT3_HI = 0x23c7
# 	COUNT_HI 0 30
regDB_OCCLUSION_COUNT3_LOW = 0x23c6
# 	COUNT_LOW 0 31
regDB_PERFCOUNTER0_HI = 0x3441
# 	PERFCOUNTER_HI 0 31
regDB_PERFCOUNTER0_LO = 0x3440
# 	PERFCOUNTER_LO 0 31
regDB_PERFCOUNTER0_SELECT = 0x3c40
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regDB_PERFCOUNTER0_SELECT1 = 0x3c41
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regDB_PERFCOUNTER1_HI = 0x3443
# 	PERFCOUNTER_HI 0 31
regDB_PERFCOUNTER1_LO = 0x3442
# 	PERFCOUNTER_LO 0 31
regDB_PERFCOUNTER1_SELECT = 0x3c42
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regDB_PERFCOUNTER1_SELECT1 = 0x3c43
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regDB_PERFCOUNTER2_HI = 0x3445
# 	PERFCOUNTER_HI 0 31
regDB_PERFCOUNTER2_LO = 0x3444
# 	PERFCOUNTER_LO 0 31
regDB_PERFCOUNTER2_SELECT = 0x3c44
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regDB_PERFCOUNTER3_HI = 0x3447
# 	PERFCOUNTER_HI 0 31
regDB_PERFCOUNTER3_LO = 0x3446
# 	PERFCOUNTER_LO 0 31
regDB_PERFCOUNTER3_SELECT = 0x3c46
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regDB_PRELOAD_CONTROL = 0x2b2
# 	START_X 0 7
# 	START_Y 8 15
# 	MAX_X 16 23
# 	MAX_Y 24 31
regDB_RENDER_CONTROL = 0x0
# 	DEPTH_CLEAR_ENABLE 0 0
# 	STENCIL_CLEAR_ENABLE 1 1
# 	DEPTH_COPY 2 2
# 	STENCIL_COPY 3 3
# 	RESUMMARIZE_ENABLE 4 4
# 	STENCIL_COMPRESS_DISABLE 5 5
# 	DEPTH_COMPRESS_DISABLE 6 6
# 	COPY_CENTROID 7 7
# 	COPY_SAMPLE 8 11
# 	DECOMPRESS_ENABLE 12 12
# 	PS_INVOKE_DISABLE 14 14
# 	OREO_MODE 16 17
# 	FORCE_OREO_MODE 18 18
# 	FORCE_EXPORT_ORDER 19 19
# 	MAX_ALLOWED_TILES_IN_WAVE 20 23
regDB_RENDER_OVERRIDE = 0x3
# 	FORCE_HIZ_ENABLE 0 1
# 	FORCE_HIS_ENABLE0 2 3
# 	FORCE_HIS_ENABLE1 4 5
# 	FORCE_SHADER_Z_ORDER 6 6
# 	FAST_Z_DISABLE 7 7
# 	FAST_STENCIL_DISABLE 8 8
# 	NOOP_CULL_DISABLE 9 9
# 	FORCE_COLOR_KILL 10 10
# 	FORCE_Z_READ 11 11
# 	FORCE_STENCIL_READ 12 12
# 	FORCE_FULL_Z_RANGE 13 14
# 	DISABLE_VIEWPORT_CLAMP 16 16
# 	IGNORE_SC_ZRANGE 17 17
# 	DISABLE_FULLY_COVERED 18 18
# 	FORCE_Z_LIMIT_SUMM 19 20
# 	MAX_TILES_IN_DTT 21 25
# 	DISABLE_TILE_RATE_TILES 26 26
# 	FORCE_Z_DIRTY 27 27
# 	FORCE_STENCIL_DIRTY 28 28
# 	FORCE_Z_VALID 29 29
# 	FORCE_STENCIL_VALID 30 30
# 	PRESERVE_COMPRESSION 31 31
regDB_RENDER_OVERRIDE2 = 0x4
# 	PARTIAL_SQUAD_LAUNCH_CONTROL 0 1
# 	PARTIAL_SQUAD_LAUNCH_COUNTDOWN 2 4
# 	DISABLE_ZMASK_EXPCLEAR_OPTIMIZATION 5 5
# 	DISABLE_SMEM_EXPCLEAR_OPTIMIZATION 6 6
# 	DISABLE_COLOR_ON_VALIDATION 7 7
# 	DECOMPRESS_Z_ON_FLUSH 8 8
# 	DISABLE_REG_SNOOP 9 9
# 	DEPTH_BOUNDS_HIER_DEPTH_DISABLE 10 10
# 	SEPARATE_HIZS_FUNC_ENABLE 11 11
# 	HIZ_ZFUNC 12 14
# 	HIS_SFUNC_FF 15 17
# 	HIS_SFUNC_BF 18 20
# 	PRESERVE_ZRANGE 21 21
# 	PRESERVE_SRESULTS 22 22
# 	DISABLE_FAST_PASS 23 23
# 	ALLOW_PARTIAL_RES_HIER_KILL 25 25
# 	CENTROID_COMPUTATION_MODE 27 28
# 	DISABLE_NOZ 29 29
regDB_RESERVED_REG_1 = 0x16
# 	FIELD_1 0 10
# 	FIELD_2 11 21
regDB_RESERVED_REG_2 = 0xf
# 	FIELD_1 0 3
# 	FIELD_2 4 7
# 	FIELD_3 8 12
# 	FIELD_4 13 14
# 	FIELD_5 15 16
# 	FIELD_6 17 18
# 	FIELD_7 19 20
# 	FIELD_8 28 31
regDB_RESERVED_REG_3 = 0x17
# 	FIELD_1 0 21
regDB_RING_CONTROL = 0x13bb
# 	COUNTER_CONTROL 0 1
regDB_RMI_L2_CACHE_CONTROL = 0x1f
# 	Z_WR_POLICY 0 1
# 	S_WR_POLICY 2 3
# 	HTILE_WR_POLICY 4 5
# 	ZPCPSD_WR_POLICY 6 7
# 	Z_RD_POLICY 16 17
# 	S_RD_POLICY 18 19
# 	HTILE_RD_POLICY 20 21
# 	Z_BIG_PAGE 24 24
# 	S_BIG_PAGE 25 25
# 	Z_NOALLOC 26 26
# 	S_NOALLOC 27 27
# 	HTILE_NOALLOC 28 28
# 	ZPCPSD_NOALLOC 29 29
regDB_SHADER_CONTROL = 0x203
# 	Z_EXPORT_ENABLE 0 0
# 	STENCIL_TEST_VAL_EXPORT_ENABLE 1 1
# 	STENCIL_OP_VAL_EXPORT_ENABLE 2 2
# 	Z_ORDER 4 5
# 	KILL_ENABLE 6 6
# 	COVERAGE_TO_MASK_ENABLE 7 7
# 	MASK_EXPORT_ENABLE 8 8
# 	EXEC_ON_HIER_FAIL 9 9
# 	EXEC_ON_NOOP 10 10
# 	ALPHA_TO_MASK_DISABLE 11 11
# 	DEPTH_BEFORE_SHADER 12 12
# 	CONSERVATIVE_Z_EXPORT 13 14
# 	DUAL_QUAD_DISABLE 15 15
# 	PRIMITIVE_ORDERED_PIXEL_SHADER 16 16
# 	PRE_SHADER_DEPTH_COVERAGE_ENABLE 23 23
# 	OREO_BLEND_ENABLE 24 24
# 	OVERRIDE_INTRINSIC_RATE_ENABLE 25 25
# 	OVERRIDE_INTRINSIC_RATE 26 28
regDB_SRESULTS_COMPARE_STATE0 = 0x2b0
# 	COMPAREFUNC0 0 2
# 	COMPAREVALUE0 4 11
# 	COMPAREMASK0 12 19
# 	ENABLE0 24 24
regDB_SRESULTS_COMPARE_STATE1 = 0x2b1
# 	COMPAREFUNC1 0 2
# 	COMPAREVALUE1 4 11
# 	COMPAREMASK1 12 19
# 	ENABLE1 24 24
regDB_STENCILREFMASK = 0x10c
# 	STENCILTESTVAL 0 7
# 	STENCILMASK 8 15
# 	STENCILWRITEMASK 16 23
# 	STENCILOPVAL 24 31
regDB_STENCILREFMASK_BF = 0x10d
# 	STENCILTESTVAL_BF 0 7
# 	STENCILMASK_BF 8 15
# 	STENCILWRITEMASK_BF 16 23
# 	STENCILOPVAL_BF 24 31
regDB_STENCIL_CLEAR = 0xa
# 	CLEAR 0 7
regDB_STENCIL_CONTROL = 0x10b
# 	STENCILFAIL 0 3
# 	STENCILZPASS 4 7
# 	STENCILZFAIL 8 11
# 	STENCILFAIL_BF 12 15
# 	STENCILZPASS_BF 16 19
# 	STENCILZFAIL_BF 20 23
regDB_STENCIL_INFO = 0x11
# 	FORMAT 0 0
# 	SW_MODE 4 8
# 	FAULT_BEHAVIOR 9 10
# 	ITERATE_FLUSH 11 11
# 	PARTIALLY_RESIDENT 12 12
# 	RESERVED_FIELD_1 13 15
# 	ITERATE_256 20 20
# 	ALLOW_EXPCLEAR 27 27
# 	TILE_STENCIL_DISABLE 29 29
regDB_STENCIL_READ_BASE = 0x13
# 	BASE_256B 0 31
regDB_STENCIL_READ_BASE_HI = 0x1b
# 	BASE_HI 0 7
regDB_STENCIL_WRITE_BASE = 0x15
# 	BASE_256B 0 31
regDB_STENCIL_WRITE_BASE_HI = 0x1d
# 	BASE_HI 0 7
regDB_SUBTILE_CONTROL = 0x13b6
# 	MSAA1_X 0 1
# 	MSAA1_Y 2 3
# 	MSAA2_X 4 5
# 	MSAA2_Y 6 7
# 	MSAA4_X 8 9
# 	MSAA4_Y 10 11
# 	MSAA8_X 12 13
# 	MSAA8_Y 14 15
# 	MSAA16_X 16 17
# 	MSAA16_Y 18 19
regDB_WATERMARKS = 0x13b5
# 	DEPTH_FREE 0 7
# 	DEPTH_FLUSH 8 15
# 	DEPTH_PENDING_FREE 16 23
# 	DEPTH_CACHELINE_FREE 24 31
regDB_Z_INFO = 0x10
# 	FORMAT 0 1
# 	NUM_SAMPLES 2 3
# 	SW_MODE 4 8
# 	FAULT_BEHAVIOR 9 10
# 	ITERATE_FLUSH 11 11
# 	PARTIALLY_RESIDENT 12 12
# 	RESERVED_FIELD_1 13 15
# 	MAXMIP 16 19
# 	ITERATE_256 20 20
# 	DECOMPRESS_ON_N_ZPLANES 23 26
# 	ALLOW_EXPCLEAR 27 27
# 	READ_SIZE 28 28
# 	TILE_SURFACE_ENABLE 29 29
# 	ZRANGE_PRECISION 31 31
regDB_Z_READ_BASE = 0x12
# 	BASE_256B 0 31
regDB_Z_READ_BASE_HI = 0x1a
# 	BASE_HI 0 7
regDB_Z_WRITE_BASE = 0x14
# 	BASE_256B 0 31
regDB_Z_WRITE_BASE_HI = 0x1c
# 	BASE_HI 0 7
regDIDT_EDC_CTRL = 0x1901
# 	EDC_EN 0 0
# 	EDC_SW_RST 1 1
# 	EDC_CLK_EN_OVERRIDE 2 2
# 	EDC_FORCE_STALL 3 3
# 	EDC_TRIGGER_THROTTLE_LOWBIT 4 9
# 	EDC_STALL_PATTERN_BIT_NUMS 10 13
# 	EDC_ALLOW_WRITE_PWRDELTA 14 14
# 	EDC_ALGORITHM_MODE 15 15
# 	EDC_AVGDIV 16 19
# 	EDC_THRESHOLD_RSHIFT_SEL 20 20
# 	EDC_THRESHOLD_RSHIFT_BIT_NUMS 21 23
# 	RLC_FORCE_STALL_EN 24 24
# 	RLC_STALL_LEVEL_SEL 25 25
regDIDT_EDC_DYNAMIC_THRESHOLD_RO = 0x1909
# 	EDC_DYNAMIC_THRESHOLD_RO 0 0
regDIDT_EDC_OVERFLOW = 0x190a
# 	EDC_ROLLING_POWER_DELTA_OVERFLOW 0 0
# 	EDC_THROTTLE_LEVEL_OVERFLOW_COUNTER 1 16
regDIDT_EDC_ROLLING_POWER_DELTA = 0x190b
# 	EDC_ROLLING_POWER_DELTA 0 31
regDIDT_EDC_STALL_PATTERN_1_2 = 0x1904
# 	EDC_STALL_PATTERN_1 0 14
# 	EDC_STALL_PATTERN_2 16 30
regDIDT_EDC_STALL_PATTERN_3_4 = 0x1905
# 	EDC_STALL_PATTERN_3 0 14
# 	EDC_STALL_PATTERN_4 16 30
regDIDT_EDC_STALL_PATTERN_5_6 = 0x1906
# 	EDC_STALL_PATTERN_5 0 14
# 	EDC_STALL_PATTERN_6 16 30
regDIDT_EDC_STALL_PATTERN_7 = 0x1907
# 	EDC_STALL_PATTERN_7 0 14
regDIDT_EDC_STATUS = 0x1908
# 	EDC_FSM_STATE 0 0
# 	EDC_THROTTLE_LEVEL 1 3
regDIDT_EDC_THRESHOLD = 0x1903
# 	EDC_THRESHOLD 0 31
regDIDT_EDC_THROTTLE_CTRL = 0x1902
# 	SQ_STALL_EN 0 0
# 	DB_STALL_EN 1 1
# 	TCP_STALL_EN 2 2
# 	TD_STALL_EN 3 3
# 	PATTERN_EXTEND_EN 4 4
# 	PATTERN_EXTEND_MODE 5 7
regDIDT_INDEX_AUTO_INCR_EN = 0x1900
# 	DIDT_INDEX_AUTO_INCR_EN 0 0
regDIDT_IND_DATA = 0x190d
# 	DIDT_IND_DATA 0 31
regDIDT_IND_INDEX = 0x190c
# 	DIDT_IND_INDEX 0 31
regDIDT_STALL_PATTERN_1_2 = 0x1aff
# 	DIDT_STALL_PATTERN_1 0 14
# 	DIDT_STALL_PATTERN_2 16 30
regDIDT_STALL_PATTERN_3_4 = 0x1b00
# 	DIDT_STALL_PATTERN_3 0 14
# 	DIDT_STALL_PATTERN_4 16 30
regDIDT_STALL_PATTERN_5_6 = 0x1b01
# 	DIDT_STALL_PATTERN_5 0 14
# 	DIDT_STALL_PATTERN_6 16 30
regDIDT_STALL_PATTERN_7 = 0x1b02
# 	DIDT_STALL_PATTERN_7 0 14
regDIDT_STALL_PATTERN_CTRL = 0x1afe
# 	DIDT_DROOP_CTRL_EN 0 0
# 	DIDT_DROOP_SW_RST 1 1
# 	DIDT_DROOP_CLK_EN_OVERRIDE 2 2
# 	DIDT_STALL_PATTERN_BIT_NUMS 3 6
# 	DIDT_PATTERN_EXTEND_EN 7 7
# 	DIDT_PATTERN_EXTEND_MODE 8 10
regEDC_HYSTERESIS_CNTL = 0x1af1
# 	MAX_HYSTERESIS 0 7
# 	EDC_AGGR_TIMER 8 15
# 	PATTERN_EXTEND_EN 16 16
# 	PATTERN_EXTEND_MODE 17 19
# 	EDC_AGGR_MODE 20 20
regEDC_HYSTERESIS_STAT = 0x1b0e
# 	HYSTERESIS_CNT 0 7
# 	EDC_STATUS 8 8
# 	EDC_CREDIT_INCR_OVERFLOW 9 9
# 	EDC_THRESHOLD_SEL 10 10
regEDC_PERF_COUNTER = 0x1b0b
# 	EDC_PERF_COUNTER 0 31
regEDC_STRETCH_NUM_PERF_COUNTER = 0x1b06
# 	STRETCH_NUM_PERF_COUNTER 0 31
regEDC_STRETCH_PERF_COUNTER = 0x1b04
# 	STRETCH_PERF_COUNTER 0 31
regEDC_UNSTRETCH_PERF_COUNTER = 0x1b05
# 	UNSTRETCH_PERF_COUNTER 0 31
regGB_ADDR_CONFIG = 0x13de
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regGB_ADDR_CONFIG_READ = 0x13e2
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regGB_BACKEND_MAP = 0x13df
# 	BACKEND_MAP 0 31
regGB_EDC_MODE = 0x1e1e
# 	FORCE_SEC_ON_DED 15 15
# 	COUNT_FED_OUT 16 16
# 	GATE_FUE 17 17
# 	DED_MODE 20 21
# 	PROP_FED 29 29
# 	BYPASS 31 31
regGB_GPU_ID = 0x13e0
# 	GPU_ID 0 3
regGCEA_DRAM_PAGE_BURST = 0x17aa
# 	RD_LIMIT_LO 0 7
# 	RD_LIMIT_HI 8 15
# 	WR_LIMIT_LO 16 23
# 	WR_LIMIT_HI 24 31
regGCEA_DRAM_RD_CAM_CNTL = 0x17a8
# 	DEPTH_GROUP0 0 3
# 	DEPTH_GROUP1 4 7
# 	DEPTH_GROUP2 8 11
# 	DEPTH_GROUP3 12 15
# 	REORDER_LIMIT_GROUP0 16 18
# 	REORDER_LIMIT_GROUP1 19 21
# 	REORDER_LIMIT_GROUP2 22 24
# 	REORDER_LIMIT_GROUP3 25 27
# 	REFILL_CHAIN 28 28
regGCEA_DRAM_RD_CLI2GRP_MAP0 = 0x17a0
# 	CID0_GROUP 0 1
# 	CID1_GROUP 2 3
# 	CID2_GROUP 4 5
# 	CID3_GROUP 6 7
# 	CID4_GROUP 8 9
# 	CID5_GROUP 10 11
# 	CID6_GROUP 12 13
# 	CID7_GROUP 14 15
# 	CID8_GROUP 16 17
# 	CID9_GROUP 18 19
# 	CID10_GROUP 20 21
# 	CID11_GROUP 22 23
# 	CID12_GROUP 24 25
# 	CID13_GROUP 26 27
# 	CID14_GROUP 28 29
# 	CID15_GROUP 30 31
regGCEA_DRAM_RD_CLI2GRP_MAP1 = 0x17a1
# 	CID16_GROUP 0 1
# 	CID17_GROUP 2 3
# 	CID18_GROUP 4 5
# 	CID19_GROUP 6 7
# 	CID20_GROUP 8 9
# 	CID21_GROUP 10 11
# 	CID22_GROUP 12 13
# 	CID23_GROUP 14 15
# 	CID24_GROUP 16 17
# 	CID25_GROUP 18 19
# 	CID26_GROUP 20 21
# 	CID27_GROUP 22 23
# 	CID28_GROUP 24 25
# 	CID29_GROUP 26 27
# 	CID30_GROUP 28 29
# 	CID31_GROUP 30 31
regGCEA_DRAM_RD_GRP2VC_MAP = 0x17a4
# 	GROUP0_VC 0 2
# 	GROUP1_VC 3 5
# 	GROUP2_VC 6 8
# 	GROUP3_VC 9 11
regGCEA_DRAM_RD_LAZY = 0x17a6
# 	GROUP0_DELAY 0 2
# 	GROUP1_DELAY 3 5
# 	GROUP2_DELAY 6 8
# 	GROUP3_DELAY 9 11
# 	REQ_ACCUM_THRESH 12 17
# 	REQ_ACCUM_TIMEOUT 20 26
# 	REQ_ACCUM_IDLEMAX 27 30
regGCEA_DRAM_RD_PRI_AGE = 0x17ab
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP0_AGE_COEFFICIENT 12 14
# 	GROUP1_AGE_COEFFICIENT 15 17
# 	GROUP2_AGE_COEFFICIENT 18 20
# 	GROUP3_AGE_COEFFICIENT 21 23
regGCEA_DRAM_RD_PRI_FIXED = 0x17af
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
regGCEA_DRAM_RD_PRI_QUANT_PRI1 = 0x17b3
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_RD_PRI_QUANT_PRI2 = 0x17b4
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_RD_PRI_QUANT_PRI3 = 0x17b5
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_RD_PRI_QUEUING = 0x17ad
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
regGCEA_DRAM_RD_PRI_URGENCY = 0x17b1
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP0_URGENCY_MODE 12 12
# 	GROUP1_URGENCY_MODE 13 13
# 	GROUP2_URGENCY_MODE 14 14
# 	GROUP3_URGENCY_MODE 15 15
regGCEA_DRAM_WR_CAM_CNTL = 0x17a9
# 	DEPTH_GROUP0 0 3
# 	DEPTH_GROUP1 4 7
# 	DEPTH_GROUP2 8 11
# 	DEPTH_GROUP3 12 15
# 	REORDER_LIMIT_GROUP0 16 18
# 	REORDER_LIMIT_GROUP1 19 21
# 	REORDER_LIMIT_GROUP2 22 24
# 	REORDER_LIMIT_GROUP3 25 27
# 	REFILL_CHAIN 28 28
regGCEA_DRAM_WR_CLI2GRP_MAP0 = 0x17a2
# 	CID0_GROUP 0 1
# 	CID1_GROUP 2 3
# 	CID2_GROUP 4 5
# 	CID3_GROUP 6 7
# 	CID4_GROUP 8 9
# 	CID5_GROUP 10 11
# 	CID6_GROUP 12 13
# 	CID7_GROUP 14 15
# 	CID8_GROUP 16 17
# 	CID9_GROUP 18 19
# 	CID10_GROUP 20 21
# 	CID11_GROUP 22 23
# 	CID12_GROUP 24 25
# 	CID13_GROUP 26 27
# 	CID14_GROUP 28 29
# 	CID15_GROUP 30 31
regGCEA_DRAM_WR_CLI2GRP_MAP1 = 0x17a3
# 	CID16_GROUP 0 1
# 	CID17_GROUP 2 3
# 	CID18_GROUP 4 5
# 	CID19_GROUP 6 7
# 	CID20_GROUP 8 9
# 	CID21_GROUP 10 11
# 	CID22_GROUP 12 13
# 	CID23_GROUP 14 15
# 	CID24_GROUP 16 17
# 	CID25_GROUP 18 19
# 	CID26_GROUP 20 21
# 	CID27_GROUP 22 23
# 	CID28_GROUP 24 25
# 	CID29_GROUP 26 27
# 	CID30_GROUP 28 29
# 	CID31_GROUP 30 31
regGCEA_DRAM_WR_GRP2VC_MAP = 0x17a5
# 	GROUP0_VC 0 2
# 	GROUP1_VC 3 5
# 	GROUP2_VC 6 8
# 	GROUP3_VC 9 11
regGCEA_DRAM_WR_LAZY = 0x17a7
# 	GROUP0_DELAY 0 2
# 	GROUP1_DELAY 3 5
# 	GROUP2_DELAY 6 8
# 	GROUP3_DELAY 9 11
# 	REQ_ACCUM_THRESH 12 17
# 	REQ_ACCUM_TIMEOUT 20 26
# 	REQ_ACCUM_IDLEMAX 27 30
regGCEA_DRAM_WR_PRI_AGE = 0x17ac
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP0_AGE_COEFFICIENT 12 14
# 	GROUP1_AGE_COEFFICIENT 15 17
# 	GROUP2_AGE_COEFFICIENT 18 20
# 	GROUP3_AGE_COEFFICIENT 21 23
regGCEA_DRAM_WR_PRI_FIXED = 0x17b0
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
regGCEA_DRAM_WR_PRI_QUANT_PRI1 = 0x17b6
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_WR_PRI_QUANT_PRI2 = 0x17b7
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_WR_PRI_QUANT_PRI3 = 0x17b8
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_DRAM_WR_PRI_QUEUING = 0x17ae
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
regGCEA_DRAM_WR_PRI_URGENCY = 0x17b2
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP0_URGENCY_MODE 12 12
# 	GROUP1_URGENCY_MODE 13 13
# 	GROUP2_URGENCY_MODE 14 14
# 	GROUP3_URGENCY_MODE 15 15
regGCEA_DSM_CNTL = 0x14b4
# 	DRAMRD_CMDMEM_DSM_IRRITATOR_DATA 0 1
# 	DRAMRD_CMDMEM_ENABLE_SINGLE_WRITE 2 2
# 	DRAMWR_CMDMEM_DSM_IRRITATOR_DATA 3 4
# 	DRAMWR_CMDMEM_ENABLE_SINGLE_WRITE 5 5
# 	DRAMWR_DATAMEM_DSM_IRRITATOR_DATA 6 7
# 	DRAMWR_DATAMEM_ENABLE_SINGLE_WRITE 8 8
# 	RRET_TAGMEM_DSM_IRRITATOR_DATA 9 10
# 	RRET_TAGMEM_ENABLE_SINGLE_WRITE 11 11
# 	WRET_TAGMEM_DSM_IRRITATOR_DATA 12 13
# 	WRET_TAGMEM_ENABLE_SINGLE_WRITE 14 14
# 	GMIRD_CMDMEM_DSM_IRRITATOR_DATA 15 16
# 	GMIRD_CMDMEM_ENABLE_SINGLE_WRITE 17 17
# 	GMIWR_CMDMEM_DSM_IRRITATOR_DATA 18 19
# 	GMIWR_CMDMEM_ENABLE_SINGLE_WRITE 20 20
# 	GMIWR_DATAMEM_DSM_IRRITATOR_DATA 21 22
# 	GMIWR_DATAMEM_ENABLE_SINGLE_WRITE 23 23
regGCEA_DSM_CNTL2 = 0x14b7
# 	DRAMRD_CMDMEM_ENABLE_ERROR_INJECT 0 1
# 	DRAMRD_CMDMEM_SELECT_INJECT_DELAY 2 2
# 	DRAMWR_CMDMEM_ENABLE_ERROR_INJECT 3 4
# 	DRAMWR_CMDMEM_SELECT_INJECT_DELAY 5 5
# 	DRAMWR_DATAMEM_ENABLE_ERROR_INJECT 6 7
# 	DRAMWR_DATAMEM_SELECT_INJECT_DELAY 8 8
# 	RRET_TAGMEM_ENABLE_ERROR_INJECT 9 10
# 	RRET_TAGMEM_SELECT_INJECT_DELAY 11 11
# 	WRET_TAGMEM_ENABLE_ERROR_INJECT 12 13
# 	WRET_TAGMEM_SELECT_INJECT_DELAY 14 14
# 	GMIRD_CMDMEM_ENABLE_ERROR_INJECT 15 16
# 	GMIRD_CMDMEM_SELECT_INJECT_DELAY 17 17
# 	GMIWR_CMDMEM_ENABLE_ERROR_INJECT 18 19
# 	GMIWR_CMDMEM_SELECT_INJECT_DELAY 20 20
# 	GMIWR_DATAMEM_ENABLE_ERROR_INJECT 21 22
# 	GMIWR_DATAMEM_SELECT_INJECT_DELAY 23 23
# 	INJECT_DELAY 26 31
regGCEA_DSM_CNTL2A = 0x14b8
# 	DRAMRD_PAGEMEM_ENABLE_ERROR_INJECT 0 1
# 	DRAMRD_PAGEMEM_SELECT_INJECT_DELAY 2 2
# 	DRAMWR_PAGEMEM_ENABLE_ERROR_INJECT 3 4
# 	DRAMWR_PAGEMEM_SELECT_INJECT_DELAY 5 5
# 	IORD_CMDMEM_ENABLE_ERROR_INJECT 6 7
# 	IORD_CMDMEM_SELECT_INJECT_DELAY 8 8
# 	IOWR_CMDMEM_ENABLE_ERROR_INJECT 9 10
# 	IOWR_CMDMEM_SELECT_INJECT_DELAY 11 11
# 	IOWR_DATAMEM_ENABLE_ERROR_INJECT 12 13
# 	IOWR_DATAMEM_SELECT_INJECT_DELAY 14 14
# 	GMIRD_PAGEMEM_ENABLE_ERROR_INJECT 15 16
# 	GMIRD_PAGEMEM_SELECT_INJECT_DELAY 17 17
# 	GMIWR_PAGEMEM_ENABLE_ERROR_INJECT 18 19
# 	GMIWR_PAGEMEM_SELECT_INJECT_DELAY 20 20
regGCEA_DSM_CNTL2B = 0x14b9
regGCEA_DSM_CNTLA = 0x14b5
# 	DRAMRD_PAGEMEM_DSM_IRRITATOR_DATA 0 1
# 	DRAMRD_PAGEMEM_ENABLE_SINGLE_WRITE 2 2
# 	DRAMWR_PAGEMEM_DSM_IRRITATOR_DATA 3 4
# 	DRAMWR_PAGEMEM_ENABLE_SINGLE_WRITE 5 5
# 	IORD_CMDMEM_DSM_IRRITATOR_DATA 6 7
# 	IORD_CMDMEM_ENABLE_SINGLE_WRITE 8 8
# 	IOWR_CMDMEM_DSM_IRRITATOR_DATA 9 10
# 	IOWR_CMDMEM_ENABLE_SINGLE_WRITE 11 11
# 	IOWR_DATAMEM_DSM_IRRITATOR_DATA 12 13
# 	IOWR_DATAMEM_ENABLE_SINGLE_WRITE 14 14
# 	GMIRD_PAGEMEM_DSM_IRRITATOR_DATA 15 16
# 	GMIRD_PAGEMEM_ENABLE_SINGLE_WRITE 17 17
# 	GMIWR_PAGEMEM_DSM_IRRITATOR_DATA 18 19
# 	GMIWR_PAGEMEM_ENABLE_SINGLE_WRITE 20 20
regGCEA_DSM_CNTLB = 0x14b6
regGCEA_EDC_CNT = 0x14b2
# 	DRAMRD_CMDMEM_SEC_COUNT 0 1
# 	DRAMRD_CMDMEM_DED_COUNT 2 3
# 	DRAMWR_CMDMEM_SEC_COUNT 4 5
# 	DRAMWR_CMDMEM_DED_COUNT 6 7
# 	DRAMWR_DATAMEM_SEC_COUNT 8 9
# 	DRAMWR_DATAMEM_DED_COUNT 10 11
# 	RRET_TAGMEM_SEC_COUNT 12 13
# 	RRET_TAGMEM_DED_COUNT 14 15
# 	WRET_TAGMEM_SEC_COUNT 16 17
# 	WRET_TAGMEM_DED_COUNT 18 19
# 	IOWR_DATAMEM_SEC_COUNT 20 21
# 	IOWR_DATAMEM_DED_COUNT 22 23
# 	DRAMRD_PAGEMEM_SED_COUNT 24 25
# 	DRAMWR_PAGEMEM_SED_COUNT 26 27
# 	IORD_CMDMEM_SED_COUNT 28 29
# 	IOWR_CMDMEM_SED_COUNT 30 31
regGCEA_EDC_CNT2 = 0x14b3
# 	GMIRD_CMDMEM_SEC_COUNT 0 1
# 	GMIRD_CMDMEM_DED_COUNT 2 3
# 	GMIWR_CMDMEM_SEC_COUNT 4 5
# 	GMIWR_CMDMEM_DED_COUNT 6 7
# 	GMIWR_DATAMEM_SEC_COUNT 8 9
# 	GMIWR_DATAMEM_DED_COUNT 10 11
# 	GMIRD_PAGEMEM_SED_COUNT 12 13
# 	GMIWR_PAGEMEM_SED_COUNT 14 15
# 	MAM_D0MEM_SED_COUNT 16 17
# 	MAM_D1MEM_SED_COUNT 18 19
# 	MAM_D2MEM_SED_COUNT 20 21
# 	MAM_D3MEM_SED_COUNT 22 23
# 	MAM_D0MEM_DED_COUNT 24 25
# 	MAM_D1MEM_DED_COUNT 26 27
# 	MAM_D2MEM_DED_COUNT 28 29
# 	MAM_D3MEM_DED_COUNT 30 31
regGCEA_EDC_CNT3 = 0x151a
# 	DRAMRD_PAGEMEM_DED_COUNT 0 1
# 	DRAMWR_PAGEMEM_DED_COUNT 2 3
# 	IORD_CMDMEM_DED_COUNT 4 5
# 	IOWR_CMDMEM_DED_COUNT 6 7
# 	GMIRD_PAGEMEM_DED_COUNT 8 9
# 	GMIWR_PAGEMEM_DED_COUNT 10 11
# 	MAM_A0MEM_SEC_COUNT 12 13
# 	MAM_A0MEM_DED_COUNT 14 15
# 	MAM_A1MEM_SEC_COUNT 16 17
# 	MAM_A1MEM_DED_COUNT 18 19
# 	MAM_A2MEM_SEC_COUNT 20 21
# 	MAM_A2MEM_DED_COUNT 22 23
# 	MAM_A3MEM_SEC_COUNT 24 25
# 	MAM_A3MEM_DED_COUNT 26 27
# 	MAM_AFMEM_SEC_COUNT 28 29
# 	MAM_AFMEM_DED_COUNT 30 31
regGCEA_ERR_STATUS = 0x14be
# 	SDP_RDRSP_STATUS 0 3
# 	SDP_WRRSP_STATUS 4 7
# 	SDP_RDRSP_DATASTATUS 8 9
# 	SDP_RDRSP_DATAPARITY_ERROR 10 10
# 	CLEAR_ERROR_STATUS 11 11
# 	BUSY_ON_ERROR 12 12
# 	FUE_FLAG 13 13
# 	IGNORE_RDRSP_FED 14 14
# 	INTERRUPT_ON_FATAL 15 15
# 	INTERRUPT_IGNORE_CLI_FATAL 16 16
# 	LEVEL_INTERRUPT 17 17
regGCEA_GL2C_XBR_CREDITS = 0x14ba
# 	DRAM_RD_LIMIT 0 5
# 	DRAM_RD_RESERVE 6 7
# 	IO_RD_LIMIT 8 13
# 	IO_RD_RESERVE 14 15
# 	DRAM_WR_LIMIT 16 21
# 	DRAM_WR_RESERVE 22 23
# 	IO_WR_LIMIT 24 29
# 	IO_WR_RESERVE 30 31
regGCEA_GL2C_XBR_MAXBURST = 0x14bb
# 	DRAM_RD 0 3
# 	IO_RD 4 7
# 	DRAM_WR 8 11
# 	IO_WR 12 15
# 	DRAM_RD_COMB_FLUSH_TIMER 16 18
# 	DRAM_RD_COMB_SAME64B_ONLY 19 19
# 	DRAM_WR_COMB_FLUSH_TIMER 20 22
# 	DRAM_WR_COMB_SAME64B_ONLY 23 23
regGCEA_ICG_CTRL = 0x50c4
# 	SOFT_OVERRIDE_RETURN 0 0
# 	SOFT_OVERRIDE_READ 1 1
# 	SOFT_OVERRIDE_WRITE 2 2
# 	SOFT_OVERRIDE_REGISTER 3 3
# 	SOFT_OVERRIDE_PERFMON 4 4
regGCEA_IO_GROUP_BURST = 0x1883
# 	RD_LIMIT_LO 0 7
# 	RD_LIMIT_HI 8 15
# 	WR_LIMIT_LO 16 23
# 	WR_LIMIT_HI 24 31
regGCEA_IO_RD_CLI2GRP_MAP0 = 0x187d
# 	CID0_GROUP 0 1
# 	CID1_GROUP 2 3
# 	CID2_GROUP 4 5
# 	CID3_GROUP 6 7
# 	CID4_GROUP 8 9
# 	CID5_GROUP 10 11
# 	CID6_GROUP 12 13
# 	CID7_GROUP 14 15
# 	CID8_GROUP 16 17
# 	CID9_GROUP 18 19
# 	CID10_GROUP 20 21
# 	CID11_GROUP 22 23
# 	CID12_GROUP 24 25
# 	CID13_GROUP 26 27
# 	CID14_GROUP 28 29
# 	CID15_GROUP 30 31
regGCEA_IO_RD_CLI2GRP_MAP1 = 0x187e
# 	CID16_GROUP 0 1
# 	CID17_GROUP 2 3
# 	CID18_GROUP 4 5
# 	CID19_GROUP 6 7
# 	CID20_GROUP 8 9
# 	CID21_GROUP 10 11
# 	CID22_GROUP 12 13
# 	CID23_GROUP 14 15
# 	CID24_GROUP 16 17
# 	CID25_GROUP 18 19
# 	CID26_GROUP 20 21
# 	CID27_GROUP 22 23
# 	CID28_GROUP 24 25
# 	CID29_GROUP 26 27
# 	CID30_GROUP 28 29
# 	CID31_GROUP 30 31
regGCEA_IO_RD_COMBINE_FLUSH = 0x1881
# 	GROUP0_TIMER 0 3
# 	GROUP1_TIMER 4 7
# 	GROUP2_TIMER 8 11
# 	GROUP3_TIMER 12 15
# 	COMB_MODE 16 17
regGCEA_IO_RD_PRI_AGE = 0x1884
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP0_AGE_COEFFICIENT 12 14
# 	GROUP1_AGE_COEFFICIENT 15 17
# 	GROUP2_AGE_COEFFICIENT 18 20
# 	GROUP3_AGE_COEFFICIENT 21 23
regGCEA_IO_RD_PRI_FIXED = 0x1888
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
regGCEA_IO_RD_PRI_QUANT_PRI1 = 0x188e
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_RD_PRI_QUANT_PRI2 = 0x188f
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_RD_PRI_QUANT_PRI3 = 0x1890
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_RD_PRI_QUEUING = 0x1886
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
regGCEA_IO_RD_PRI_URGENCY = 0x188a
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP0_URGENCY_MODE 12 12
# 	GROUP1_URGENCY_MODE 13 13
# 	GROUP2_URGENCY_MODE 14 14
# 	GROUP3_URGENCY_MODE 15 15
regGCEA_IO_RD_PRI_URGENCY_MASKING = 0x188c
# 	CID0_MASK 0 0
# 	CID1_MASK 1 1
# 	CID2_MASK 2 2
# 	CID3_MASK 3 3
# 	CID4_MASK 4 4
# 	CID5_MASK 5 5
# 	CID6_MASK 6 6
# 	CID7_MASK 7 7
# 	CID8_MASK 8 8
# 	CID9_MASK 9 9
# 	CID10_MASK 10 10
# 	CID11_MASK 11 11
# 	CID12_MASK 12 12
# 	CID13_MASK 13 13
# 	CID14_MASK 14 14
# 	CID15_MASK 15 15
# 	CID16_MASK 16 16
# 	CID17_MASK 17 17
# 	CID18_MASK 18 18
# 	CID19_MASK 19 19
# 	CID20_MASK 20 20
# 	CID21_MASK 21 21
# 	CID22_MASK 22 22
# 	CID23_MASK 23 23
# 	CID24_MASK 24 24
# 	CID25_MASK 25 25
# 	CID26_MASK 26 26
# 	CID27_MASK 27 27
# 	CID28_MASK 28 28
# 	CID29_MASK 29 29
# 	CID30_MASK 30 30
# 	CID31_MASK 31 31
regGCEA_IO_WR_CLI2GRP_MAP0 = 0x187f
# 	CID0_GROUP 0 1
# 	CID1_GROUP 2 3
# 	CID2_GROUP 4 5
# 	CID3_GROUP 6 7
# 	CID4_GROUP 8 9
# 	CID5_GROUP 10 11
# 	CID6_GROUP 12 13
# 	CID7_GROUP 14 15
# 	CID8_GROUP 16 17
# 	CID9_GROUP 18 19
# 	CID10_GROUP 20 21
# 	CID11_GROUP 22 23
# 	CID12_GROUP 24 25
# 	CID13_GROUP 26 27
# 	CID14_GROUP 28 29
# 	CID15_GROUP 30 31
regGCEA_IO_WR_CLI2GRP_MAP1 = 0x1880
# 	CID16_GROUP 0 1
# 	CID17_GROUP 2 3
# 	CID18_GROUP 4 5
# 	CID19_GROUP 6 7
# 	CID20_GROUP 8 9
# 	CID21_GROUP 10 11
# 	CID22_GROUP 12 13
# 	CID23_GROUP 14 15
# 	CID24_GROUP 16 17
# 	CID25_GROUP 18 19
# 	CID26_GROUP 20 21
# 	CID27_GROUP 22 23
# 	CID28_GROUP 24 25
# 	CID29_GROUP 26 27
# 	CID30_GROUP 28 29
# 	CID31_GROUP 30 31
regGCEA_IO_WR_COMBINE_FLUSH = 0x1882
# 	GROUP0_TIMER 0 3
# 	GROUP1_TIMER 4 7
# 	GROUP2_TIMER 8 11
# 	GROUP3_TIMER 12 15
# 	COMB_MODE 16 17
regGCEA_IO_WR_PRI_AGE = 0x1885
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP0_AGE_COEFFICIENT 12 14
# 	GROUP1_AGE_COEFFICIENT 15 17
# 	GROUP2_AGE_COEFFICIENT 18 20
# 	GROUP3_AGE_COEFFICIENT 21 23
regGCEA_IO_WR_PRI_FIXED = 0x1889
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
regGCEA_IO_WR_PRI_QUANT_PRI1 = 0x1891
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_WR_PRI_QUANT_PRI2 = 0x1892
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_WR_PRI_QUANT_PRI3 = 0x1893
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGCEA_IO_WR_PRI_QUEUING = 0x1887
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
regGCEA_IO_WR_PRI_URGENCY = 0x188b
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP0_URGENCY_MODE 12 12
# 	GROUP1_URGENCY_MODE 13 13
# 	GROUP2_URGENCY_MODE 14 14
# 	GROUP3_URGENCY_MODE 15 15
regGCEA_IO_WR_PRI_URGENCY_MASKING = 0x188d
# 	CID0_MASK 0 0
# 	CID1_MASK 1 1
# 	CID2_MASK 2 2
# 	CID3_MASK 3 3
# 	CID4_MASK 4 4
# 	CID5_MASK 5 5
# 	CID6_MASK 6 6
# 	CID7_MASK 7 7
# 	CID8_MASK 8 8
# 	CID9_MASK 9 9
# 	CID10_MASK 10 10
# 	CID11_MASK 11 11
# 	CID12_MASK 12 12
# 	CID13_MASK 13 13
# 	CID14_MASK 14 14
# 	CID15_MASK 15 15
# 	CID16_MASK 16 16
# 	CID17_MASK 17 17
# 	CID18_MASK 18 18
# 	CID19_MASK 19 19
# 	CID20_MASK 20 20
# 	CID21_MASK 21 21
# 	CID22_MASK 22 22
# 	CID23_MASK 23 23
# 	CID24_MASK 24 24
# 	CID25_MASK 25 25
# 	CID26_MASK 26 26
# 	CID27_MASK 27 27
# 	CID28_MASK 28 28
# 	CID29_MASK 29 29
# 	CID30_MASK 30 30
# 	CID31_MASK 31 31
regGCEA_LATENCY_SAMPLING = 0x14a3
# 	SAMPLER0_DRAM 0 0
# 	SAMPLER1_DRAM 1 1
# 	SAMPLER0_GMI 2 2
# 	SAMPLER1_GMI 3 3
# 	SAMPLER0_IO 4 4
# 	SAMPLER1_IO 5 5
# 	SAMPLER0_READ 6 6
# 	SAMPLER1_READ 7 7
# 	SAMPLER0_WRITE 8 8
# 	SAMPLER1_WRITE 9 9
# 	SAMPLER0_ATOMIC_RET 10 10
# 	SAMPLER1_ATOMIC_RET 11 11
# 	SAMPLER0_ATOMIC_NORET 12 12
# 	SAMPLER1_ATOMIC_NORET 13 13
# 	SAMPLER0_VC 14 21
# 	SAMPLER1_VC 22 29
regGCEA_MAM_CTRL = 0x14ab
# 	MAM_DISABLE 0 0
# 	DBIT_COALESCE_DISABLE 1 1
# 	ARAM_COALESCE_DISABLE 2 2
# 	ARAM_FLUSH_SNOOP_EN 3 3
# 	SDMA_UPDT_ARAM 4 4
# 	ARAM_FLUSH_NOALLOC 5 5
# 	FLUSH_TRACKER 6 6
# 	CLEAR_TRACKER 7 7
# 	SDP_PRIORITY 8 11
# 	FORCE_FLUSH_UPDT_TRACKER 12 12
# 	FORCE_FLUSH_GEN_INTERRUPT 13 13
# 	TIMER_FLUSH_UPDT_TRACKER 14 14
# 	TIMER_FLUSH_GEN_INTERRUPT 15 15
# 	RESERVED_FIELD 16 22
# 	ARAM_NUM_RB_ENTRIES 23 27
# 	ARAM_RB_ADDR_HI 28 31
regGCEA_MAM_CTRL2 = 0x14a9
# 	ARAM_FLUSH_DISABLE 0 0
# 	DBIT_PF_CLR_ONLY 1 1
# 	DBIT_PF_RD_ONLY 2 2
# 	DBIT_TRACK_SEGMENT 3 5
# 	ARAM_TRACK_SEGMENT 6 8
# 	ARAM_FB_TRACK_SIZE 9 14
# 	ARAM_RB_ENTRY_SIZE 15 17
# 	ARAM_OVERRIDE_EA_STRAP 18 18
# 	ABIT_FLUSH_SPACE_OVERRIDE_ENABLE 19 19
# 	ABIT_FLUSH_SPACE_OVERRIDE_VALUE 20 20
# 	ARAM_REMOVE_TRACKER 21 21
# 	FORCE_DBIT_QUERY_DIRTY_ENABLE 22 22
# 	FORCE_DBIT_QUERY_DIRTY_VALUE 23 23
# 	RESERVED_FIELD 24 31
regGCEA_MISC = 0x14a2
# 	RELATIVE_PRI_IN_DRAM_RD_ARB 0 0
# 	RELATIVE_PRI_IN_DRAM_WR_ARB 1 1
# 	RELATIVE_PRI_IN_GMI_RD_ARB 2 2
# 	RELATIVE_PRI_IN_GMI_WR_ARB 3 3
# 	RELATIVE_PRI_IN_IO_RD_ARB 4 4
# 	RELATIVE_PRI_IN_IO_WR_ARB 5 5
# 	EARLYWRRET_ENABLE_VC0 6 6
# 	EARLYWRRET_ENABLE_VC1 7 7
# 	EARLYWRRET_ENABLE_VC2 8 8
# 	EARLYWRRET_ENABLE_VC3 9 9
# 	EARLYWRRET_ENABLE_VC4 10 10
# 	EARLYWRRET_ENABLE_VC5 11 11
# 	EARLYWRRET_ENABLE_VC6 12 12
# 	EARLYWRRET_ENABLE_VC7 13 13
# 	EARLY_SDP_ORIGDATA 14 14
# 	LINKMGR_DYNAMIC_MODE 15 16
# 	LINKMGR_HALT_THRESHOLD 17 18
# 	LINKMGR_RECONNECT_DELAY 19 20
# 	LINKMGR_IDLE_THRESHOLD 21 25
# 	FAVOUR_MIDCHAIN_CS_IN_DRAM_ARB 26 26
# 	FAVOUR_MIDCHAIN_CS_IN_GMI_ARB 27 27
# 	FAVOUR_LAST_CS_IN_DRAM_ARB 28 28
# 	FAVOUR_LAST_CS_IN_GMI_ARB 29 29
# 	SWITCH_CS_ON_W2R_IN_DRAM_ARB 30 30
# 	SWITCH_CS_ON_W2R_IN_GMI_ARB 31 31
regGCEA_MISC2 = 0x14bf
# 	CSGROUP_SWAP_IN_DRAM_ARB 0 0
# 	CSGROUP_SWAP_IN_GMI_ARB 1 1
# 	CSGRP_BURST_LIMIT_DATA_DRAM 2 6
# 	CSGRP_BURST_LIMIT_DATA_GMI 7 11
# 	IO_RDWR_PRIORITY_ENABLE 12 12
# 	BLOCK_REQUESTS 13 13
# 	REQUESTS_BLOCKED 14 14
# 	FGCLKEN_OVERRIDE 15 15
# 	LINKMGR_CRBUSY_MASK 16 16
regGCEA_PERFCOUNTER0_CFG = 0x3a03
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCEA_PERFCOUNTER1_CFG = 0x3a04
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCEA_PERFCOUNTER2_HI = 0x3261
# 	PERFCOUNTER_HI 0 31
regGCEA_PERFCOUNTER2_LO = 0x3260
# 	PERFCOUNTER_LO 0 31
regGCEA_PERFCOUNTER2_MODE = 0x3a02
# 	COMPARE_MODE0 0 1
# 	COMPARE_MODE1 2 3
# 	COMPARE_MODE2 4 5
# 	COMPARE_MODE3 6 7
# 	COMPARE_VALUE0 8 11
# 	COMPARE_VALUE1 12 15
# 	COMPARE_VALUE2 16 19
# 	COMPARE_VALUE3 20 23
regGCEA_PERFCOUNTER2_SELECT = 0x3a00
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGCEA_PERFCOUNTER2_SELECT1 = 0x3a01
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGCEA_PERFCOUNTER_HI = 0x3263
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regGCEA_PERFCOUNTER_LO = 0x3262
# 	COUNTER_LO 0 31
regGCEA_PERFCOUNTER_RSLT_CNTL = 0x3a05
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regGCEA_PROBE_CNTL = 0x14bc
# 	REQ2RSP_DELAY 0 4
# 	PRB_FILTER_DISABLE 5 5
regGCEA_PROBE_MAP = 0x14bd
# 	CHADDR0_TO_RIGHTGL2C 0 0
# 	CHADDR1_TO_RIGHTGL2C 1 1
# 	CHADDR2_TO_RIGHTGL2C 2 2
# 	CHADDR3_TO_RIGHTGL2C 3 3
# 	CHADDR4_TO_RIGHTGL2C 4 4
# 	CHADDR5_TO_RIGHTGL2C 5 5
# 	CHADDR6_TO_RIGHTGL2C 6 6
# 	CHADDR7_TO_RIGHTGL2C 7 7
# 	CHADDR8_TO_RIGHTGL2C 8 8
# 	CHADDR9_TO_RIGHTGL2C 9 9
# 	CHADDR10_TO_RIGHTGL2C 10 10
# 	CHADDR11_TO_RIGHTGL2C 11 11
# 	CHADDR12_TO_RIGHTGL2C 12 12
# 	CHADDR13_TO_RIGHTGL2C 13 13
# 	CHADDR14_TO_RIGHTGL2C 14 14
# 	CHADDR15_TO_RIGHTGL2C 15 15
# 	INTLV_SIZE 16 17
regGCEA_RRET_MEM_RESERVE = 0x1518
# 	VC0 0 3
# 	VC1 4 7
# 	VC2 8 11
# 	VC3 12 15
# 	VC4 16 19
# 	VC5 20 23
# 	VC6 24 27
# 	VC7 28 31
regGCEA_SDP_ARB_FINAL = 0x1896
# 	DRAM_BURST_LIMIT 0 4
# 	GMI_BURST_LIMIT 5 9
# 	IO_BURST_LIMIT 10 14
# 	BURST_LIMIT_MULTIPLIER 15 16
# 	RDONLY_VC0 17 17
# 	RDONLY_VC1 18 18
# 	RDONLY_VC2 19 19
# 	RDONLY_VC3 20 20
# 	RDONLY_VC4 21 21
# 	RDONLY_VC5 22 22
# 	RDONLY_VC6 23 23
# 	RDONLY_VC7 24 24
# 	ERREVENT_ON_ERROR 25 25
# 	HALTREQ_ON_ERROR 26 26
# 	GMI_BURST_STRETCH 27 27
# 	DRAM_RD_THROTTLE 28 28
# 	DRAM_WR_THROTTLE 29 29
# 	GMI_RD_THROTTLE 30 30
# 	GMI_WR_THROTTLE 31 31
regGCEA_SDP_CREDITS = 0x189a
# 	TAG_LIMIT 0 7
# 	WR_RESP_CREDITS 8 14
# 	RD_RESP_CREDITS 16 22
# 	PRB_REQ_CREDITS 24 29
regGCEA_SDP_ENABLE = 0x151e
# 	ENABLE 0 0
# 	EARLY_CREDIT_REQUEST 1 1
regGCEA_SDP_IO_PRIORITY = 0x1899
# 	RD_GROUP0_PRIORITY 0 3
# 	RD_GROUP1_PRIORITY 4 7
# 	RD_GROUP2_PRIORITY 8 11
# 	RD_GROUP3_PRIORITY 12 15
# 	WR_GROUP0_PRIORITY 16 19
# 	WR_GROUP1_PRIORITY 20 23
# 	WR_GROUP2_PRIORITY 24 27
# 	WR_GROUP3_PRIORITY 28 31
regGCEA_SDP_TAG_RESERVE0 = 0x189b
# 	VC0 0 7
# 	VC1 8 15
# 	VC2 16 23
# 	VC3 24 31
regGCEA_SDP_TAG_RESERVE1 = 0x189c
# 	VC4 0 7
# 	VC5 8 15
# 	VC6 16 23
# 	VC7 24 31
regGCEA_SDP_VCC_RESERVE0 = 0x189d
# 	VC0_CREDITS 0 5
# 	VC1_CREDITS 6 11
# 	VC2_CREDITS 12 17
# 	VC3_CREDITS 18 23
# 	VC4_CREDITS 24 29
regGCEA_SDP_VCC_RESERVE1 = 0x189e
# 	VC5_CREDITS 0 5
# 	VC6_CREDITS 6 11
# 	VC7_CREDITS 12 17
# 	DISTRIBUTE_POOL 31 31
regGCMC_MEM_POWER_LS = 0x15ac
# 	LS_SETUP 0 5
# 	LS_HOLD 6 11
regGCMC_VM_AGP_BASE = 0x167c
# 	AGP_BASE 0 23
regGCMC_VM_AGP_BOT = 0x167b
# 	AGP_BOT 0 23
regGCMC_VM_AGP_TOP = 0x167a
# 	AGP_TOP 0 23
regGCMC_VM_APT_CNTL = 0x15b1
# 	FORCE_MTYPE_UC 0 0
# 	DIRECT_SYSTEM_EN 1 1
# 	FRAG_APT_INTXN_MODE 2 3
# 	CHECK_IS_LOCAL 4 4
# 	CAP_FRAG_SIZE_2M 5 5
# 	LOCAL_SYSMEM_APERTURE_CNTL 6 7
regGCMC_VM_CACHEABLE_DRAM_ADDRESS_END = 0x15ae
# 	ADDRESS 0 19
regGCMC_VM_CACHEABLE_DRAM_ADDRESS_START = 0x15ad
# 	ADDRESS 0 19
regGCMC_VM_FB_LOCATION_BASE = 0x1678
# 	FB_BASE 0 23
regGCMC_VM_FB_LOCATION_TOP = 0x1679
# 	FB_TOP 0 23
regGCMC_VM_FB_NOALLOC_CNTL = 0x15b8
# 	LOCAL_FB_NOALLOC_NOPTE 0 0
# 	REMOTE_FB_NOALLOC_NOPTE 1 1
# 	FB_NOALLOC_WALKER_FETCH 2 2
# 	ROUTER_ATCL2_NOALLOC 3 3
# 	ROUTER_GPA_MODE2_NOALLOC 4 4
# 	ROUTER_GPA_MODE3_NOALLOC 5 5
regGCMC_VM_FB_OFFSET = 0x15a7
# 	FB_OFFSET 0 23
regGCMC_VM_FB_SIZE_OFFSET_VF0 = 0x5a80
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF1 = 0x5a81
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF10 = 0x5a8a
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF11 = 0x5a8b
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF12 = 0x5a8c
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF13 = 0x5a8d
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF14 = 0x5a8e
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF15 = 0x5a8f
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF2 = 0x5a82
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF3 = 0x5a83
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF4 = 0x5a84
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF5 = 0x5a85
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF6 = 0x5a86
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF7 = 0x5a87
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF8 = 0x5a88
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_FB_SIZE_OFFSET_VF9 = 0x5a89
# 	VF_FB_SIZE 0 15
# 	VF_FB_OFFSET 16 31
regGCMC_VM_L2_PERFCOUNTER0_CFG = 0x3d30
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER1_CFG = 0x3d31
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER2_CFG = 0x3d32
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER3_CFG = 0x3d33
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER4_CFG = 0x3d34
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER5_CFG = 0x3d35
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER6_CFG = 0x3d36
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER7_CFG = 0x3d37
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCMC_VM_L2_PERFCOUNTER_HI = 0x34e5
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regGCMC_VM_L2_PERFCOUNTER_LO = 0x34e4
# 	COUNTER_LO 0 31
regGCMC_VM_L2_PERFCOUNTER_RSLT_CNTL = 0x3d38
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regGCMC_VM_LOCAL_FB_ADDRESS_END = 0x15b3
# 	ADDRESS 0 19
regGCMC_VM_LOCAL_FB_ADDRESS_LOCK_CNTL = 0x15b4
# 	LOCK 0 0
regGCMC_VM_LOCAL_FB_ADDRESS_START = 0x15b2
# 	ADDRESS 0 19
regGCMC_VM_LOCAL_SYSMEM_ADDRESS_END = 0x15b0
# 	ADDRESS 0 19
regGCMC_VM_LOCAL_SYSMEM_ADDRESS_START = 0x15af
# 	ADDRESS 0 19
regGCMC_VM_MARC_BASE_HI_0 = 0x5e58
# 	MARC_BASE_HI_0 0 19
regGCMC_VM_MARC_BASE_HI_1 = 0x5e59
# 	MARC_BASE_HI_1 0 19
regGCMC_VM_MARC_BASE_HI_10 = 0x5e62
# 	MARC_BASE_HI_10 0 19
regGCMC_VM_MARC_BASE_HI_11 = 0x5e63
# 	MARC_BASE_HI_11 0 19
regGCMC_VM_MARC_BASE_HI_12 = 0x5e64
# 	MARC_BASE_HI_12 0 19
regGCMC_VM_MARC_BASE_HI_13 = 0x5e65
# 	MARC_BASE_HI_13 0 19
regGCMC_VM_MARC_BASE_HI_14 = 0x5e66
# 	MARC_BASE_HI_14 0 19
regGCMC_VM_MARC_BASE_HI_15 = 0x5e67
# 	MARC_BASE_HI_15 0 19
regGCMC_VM_MARC_BASE_HI_2 = 0x5e5a
# 	MARC_BASE_HI_2 0 19
regGCMC_VM_MARC_BASE_HI_3 = 0x5e5b
# 	MARC_BASE_HI_3 0 19
regGCMC_VM_MARC_BASE_HI_4 = 0x5e5c
# 	MARC_BASE_HI_4 0 19
regGCMC_VM_MARC_BASE_HI_5 = 0x5e5d
# 	MARC_BASE_HI_5 0 19
regGCMC_VM_MARC_BASE_HI_6 = 0x5e5e
# 	MARC_BASE_HI_6 0 19
regGCMC_VM_MARC_BASE_HI_7 = 0x5e5f
# 	MARC_BASE_HI_7 0 19
regGCMC_VM_MARC_BASE_HI_8 = 0x5e60
# 	MARC_BASE_HI_8 0 19
regGCMC_VM_MARC_BASE_HI_9 = 0x5e61
# 	MARC_BASE_HI_9 0 19
regGCMC_VM_MARC_BASE_LO_0 = 0x5e48
# 	MARC_BASE_LO_0 12 31
regGCMC_VM_MARC_BASE_LO_1 = 0x5e49
# 	MARC_BASE_LO_1 12 31
regGCMC_VM_MARC_BASE_LO_10 = 0x5e52
# 	MARC_BASE_LO_10 12 31
regGCMC_VM_MARC_BASE_LO_11 = 0x5e53
# 	MARC_BASE_LO_11 12 31
regGCMC_VM_MARC_BASE_LO_12 = 0x5e54
# 	MARC_BASE_LO_12 12 31
regGCMC_VM_MARC_BASE_LO_13 = 0x5e55
# 	MARC_BASE_LO_13 12 31
regGCMC_VM_MARC_BASE_LO_14 = 0x5e56
# 	MARC_BASE_LO_14 12 31
regGCMC_VM_MARC_BASE_LO_15 = 0x5e57
# 	MARC_BASE_LO_15 12 31
regGCMC_VM_MARC_BASE_LO_2 = 0x5e4a
# 	MARC_BASE_LO_2 12 31
regGCMC_VM_MARC_BASE_LO_3 = 0x5e4b
# 	MARC_BASE_LO_3 12 31
regGCMC_VM_MARC_BASE_LO_4 = 0x5e4c
# 	MARC_BASE_LO_4 12 31
regGCMC_VM_MARC_BASE_LO_5 = 0x5e4d
# 	MARC_BASE_LO_5 12 31
regGCMC_VM_MARC_BASE_LO_6 = 0x5e4e
# 	MARC_BASE_LO_6 12 31
regGCMC_VM_MARC_BASE_LO_7 = 0x5e4f
# 	MARC_BASE_LO_7 12 31
regGCMC_VM_MARC_BASE_LO_8 = 0x5e50
# 	MARC_BASE_LO_8 12 31
regGCMC_VM_MARC_BASE_LO_9 = 0x5e51
# 	MARC_BASE_LO_9 12 31
regGCMC_VM_MARC_LEN_HI_0 = 0x5e98
# 	MARC_LEN_HI_0 0 19
regGCMC_VM_MARC_LEN_HI_1 = 0x5e99
# 	MARC_LEN_HI_1 0 19
regGCMC_VM_MARC_LEN_HI_10 = 0x5ea2
# 	MARC_LEN_HI_10 0 19
regGCMC_VM_MARC_LEN_HI_11 = 0x5ea3
# 	MARC_LEN_HI_11 0 19
regGCMC_VM_MARC_LEN_HI_12 = 0x5ea4
# 	MARC_LEN_HI_12 0 19
regGCMC_VM_MARC_LEN_HI_13 = 0x5ea5
# 	MARC_LEN_HI_13 0 19
regGCMC_VM_MARC_LEN_HI_14 = 0x5ea6
# 	MARC_LEN_HI_14 0 19
regGCMC_VM_MARC_LEN_HI_15 = 0x5ea7
# 	MARC_LEN_HI_15 0 19
regGCMC_VM_MARC_LEN_HI_2 = 0x5e9a
# 	MARC_LEN_HI_2 0 19
regGCMC_VM_MARC_LEN_HI_3 = 0x5e9b
# 	MARC_LEN_HI_3 0 19
regGCMC_VM_MARC_LEN_HI_4 = 0x5e9c
# 	MARC_LEN_HI_4 0 19
regGCMC_VM_MARC_LEN_HI_5 = 0x5e9d
# 	MARC_LEN_HI_5 0 19
regGCMC_VM_MARC_LEN_HI_6 = 0x5e9e
# 	MARC_LEN_HI_6 0 19
regGCMC_VM_MARC_LEN_HI_7 = 0x5e9f
# 	MARC_LEN_HI_7 0 19
regGCMC_VM_MARC_LEN_HI_8 = 0x5ea0
# 	MARC_LEN_HI_8 0 19
regGCMC_VM_MARC_LEN_HI_9 = 0x5ea1
# 	MARC_LEN_HI_9 0 19
regGCMC_VM_MARC_LEN_LO_0 = 0x5e88
# 	MARC_LEN_LO_0 12 31
regGCMC_VM_MARC_LEN_LO_1 = 0x5e89
# 	MARC_LEN_LO_1 12 31
regGCMC_VM_MARC_LEN_LO_10 = 0x5e92
# 	MARC_LEN_LO_10 12 31
regGCMC_VM_MARC_LEN_LO_11 = 0x5e93
# 	MARC_LEN_LO_11 12 31
regGCMC_VM_MARC_LEN_LO_12 = 0x5e94
# 	MARC_LEN_LO_12 12 31
regGCMC_VM_MARC_LEN_LO_13 = 0x5e95
# 	MARC_LEN_LO_13 12 31
regGCMC_VM_MARC_LEN_LO_14 = 0x5e96
# 	MARC_LEN_LO_14 12 31
regGCMC_VM_MARC_LEN_LO_15 = 0x5e97
# 	MARC_LEN_LO_15 12 31
regGCMC_VM_MARC_LEN_LO_2 = 0x5e8a
# 	MARC_LEN_LO_2 12 31
regGCMC_VM_MARC_LEN_LO_3 = 0x5e8b
# 	MARC_LEN_LO_3 12 31
regGCMC_VM_MARC_LEN_LO_4 = 0x5e8c
# 	MARC_LEN_LO_4 12 31
regGCMC_VM_MARC_LEN_LO_5 = 0x5e8d
# 	MARC_LEN_LO_5 12 31
regGCMC_VM_MARC_LEN_LO_6 = 0x5e8e
# 	MARC_LEN_LO_6 12 31
regGCMC_VM_MARC_LEN_LO_7 = 0x5e8f
# 	MARC_LEN_LO_7 12 31
regGCMC_VM_MARC_LEN_LO_8 = 0x5e90
# 	MARC_LEN_LO_8 12 31
regGCMC_VM_MARC_LEN_LO_9 = 0x5e91
# 	MARC_LEN_LO_9 12 31
regGCMC_VM_MARC_PFVF_MAPPING_0 = 0x5ea8
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_1 = 0x5ea9
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_10 = 0x5eb2
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_11 = 0x5eb3
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_12 = 0x5eb4
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_13 = 0x5eb5
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_14 = 0x5eb6
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_15 = 0x5eb7
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_2 = 0x5eaa
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_3 = 0x5eab
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_4 = 0x5eac
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_5 = 0x5ead
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_6 = 0x5eae
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_7 = 0x5eaf
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_8 = 0x5eb0
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_PFVF_MAPPING_9 = 0x5eb1
# 	ENABLE_VFS 0 15
# 	ENABLE_PF 16 16
regGCMC_VM_MARC_RELOC_HI_0 = 0x5e78
# 	MARC_RELOC_HI_0 0 19
regGCMC_VM_MARC_RELOC_HI_1 = 0x5e79
# 	MARC_RELOC_HI_1 0 19
regGCMC_VM_MARC_RELOC_HI_10 = 0x5e82
# 	MARC_RELOC_HI_10 0 19
regGCMC_VM_MARC_RELOC_HI_11 = 0x5e83
# 	MARC_RELOC_HI_11 0 19
regGCMC_VM_MARC_RELOC_HI_12 = 0x5e84
# 	MARC_RELOC_HI_12 0 19
regGCMC_VM_MARC_RELOC_HI_13 = 0x5e85
# 	MARC_RELOC_HI_13 0 19
regGCMC_VM_MARC_RELOC_HI_14 = 0x5e86
# 	MARC_RELOC_HI_14 0 19
regGCMC_VM_MARC_RELOC_HI_15 = 0x5e87
# 	MARC_RELOC_HI_15 0 19
regGCMC_VM_MARC_RELOC_HI_2 = 0x5e7a
# 	MARC_RELOC_HI_2 0 19
regGCMC_VM_MARC_RELOC_HI_3 = 0x5e7b
# 	MARC_RELOC_HI_3 0 19
regGCMC_VM_MARC_RELOC_HI_4 = 0x5e7c
# 	MARC_RELOC_HI_4 0 19
regGCMC_VM_MARC_RELOC_HI_5 = 0x5e7d
# 	MARC_RELOC_HI_5 0 19
regGCMC_VM_MARC_RELOC_HI_6 = 0x5e7e
# 	MARC_RELOC_HI_6 0 19
regGCMC_VM_MARC_RELOC_HI_7 = 0x5e7f
# 	MARC_RELOC_HI_7 0 19
regGCMC_VM_MARC_RELOC_HI_8 = 0x5e80
# 	MARC_RELOC_HI_8 0 19
regGCMC_VM_MARC_RELOC_HI_9 = 0x5e81
# 	MARC_RELOC_HI_9 0 19
regGCMC_VM_MARC_RELOC_LO_0 = 0x5e68
# 	MARC_ENABLE_0 0 0
# 	MARC_READONLY_0 1 1
# 	MARC_RELOC_LO_0 12 31
regGCMC_VM_MARC_RELOC_LO_1 = 0x5e69
# 	MARC_ENABLE_1 0 0
# 	MARC_READONLY_1 1 1
# 	MARC_RELOC_LO_1 12 31
regGCMC_VM_MARC_RELOC_LO_10 = 0x5e72
# 	MARC_ENABLE_10 0 0
# 	MARC_READONLY_10 1 1
# 	MARC_RELOC_LO_10 12 31
regGCMC_VM_MARC_RELOC_LO_11 = 0x5e73
# 	MARC_ENABLE_11 0 0
# 	MARC_READONLY_11 1 1
# 	MARC_RELOC_LO_11 12 31
regGCMC_VM_MARC_RELOC_LO_12 = 0x5e74
# 	MARC_ENABLE_12 0 0
# 	MARC_READONLY_12 1 1
# 	MARC_RELOC_LO_12 12 31
regGCMC_VM_MARC_RELOC_LO_13 = 0x5e75
# 	MARC_ENABLE_13 0 0
# 	MARC_READONLY_13 1 1
# 	MARC_RELOC_LO_13 12 31
regGCMC_VM_MARC_RELOC_LO_14 = 0x5e76
# 	MARC_ENABLE_14 0 0
# 	MARC_READONLY_14 1 1
# 	MARC_RELOC_LO_14 12 31
regGCMC_VM_MARC_RELOC_LO_15 = 0x5e77
# 	MARC_ENABLE_15 0 0
# 	MARC_READONLY_15 1 1
# 	MARC_RELOC_LO_15 12 31
regGCMC_VM_MARC_RELOC_LO_2 = 0x5e6a
# 	MARC_ENABLE_2 0 0
# 	MARC_READONLY_2 1 1
# 	MARC_RELOC_LO_2 12 31
regGCMC_VM_MARC_RELOC_LO_3 = 0x5e6b
# 	MARC_ENABLE_3 0 0
# 	MARC_READONLY_3 1 1
# 	MARC_RELOC_LO_3 12 31
regGCMC_VM_MARC_RELOC_LO_4 = 0x5e6c
# 	MARC_ENABLE_4 0 0
# 	MARC_READONLY_4 1 1
# 	MARC_RELOC_LO_4 12 31
regGCMC_VM_MARC_RELOC_LO_5 = 0x5e6d
# 	MARC_ENABLE_5 0 0
# 	MARC_READONLY_5 1 1
# 	MARC_RELOC_LO_5 12 31
regGCMC_VM_MARC_RELOC_LO_6 = 0x5e6e
# 	MARC_ENABLE_6 0 0
# 	MARC_READONLY_6 1 1
# 	MARC_RELOC_LO_6 12 31
regGCMC_VM_MARC_RELOC_LO_7 = 0x5e6f
# 	MARC_ENABLE_7 0 0
# 	MARC_READONLY_7 1 1
# 	MARC_RELOC_LO_7 12 31
regGCMC_VM_MARC_RELOC_LO_8 = 0x5e70
# 	MARC_ENABLE_8 0 0
# 	MARC_READONLY_8 1 1
# 	MARC_RELOC_LO_8 12 31
regGCMC_VM_MARC_RELOC_LO_9 = 0x5e71
# 	MARC_ENABLE_9 0 0
# 	MARC_READONLY_9 1 1
# 	MARC_RELOC_LO_9 12 31
regGCMC_VM_MX_L1_TLB_CNTL = 0x167f
# 	ENABLE_L1_TLB 0 0
# 	SYSTEM_ACCESS_MODE 3 4
# 	SYSTEM_APERTURE_UNMAPPED_ACCESS 5 5
# 	ENABLE_ADVANCED_DRIVER_MODEL 6 6
# 	ECO_BITS 7 10
# 	MTYPE 11 13
regGCMC_VM_NB_LOWER_TOP_OF_DRAM2 = 0x15a5
# 	ENABLE 0 0
# 	LOWER_TOM2 23 31
regGCMC_VM_NB_TOP_OF_DRAM_SLOT1 = 0x15a4
# 	TOP_OF_DRAM 23 31
regGCMC_VM_NB_UPPER_TOP_OF_DRAM2 = 0x15a6
# 	UPPER_TOM2 0 11
regGCMC_VM_STEERING = 0x15aa
# 	DEFAULT_STEERING 0 1
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x15a8
# 	PHYSICAL_PAGE_NUMBER_LSB 0 31
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x15a9
# 	PHYSICAL_PAGE_NUMBER_MSB 0 3
regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x167e
# 	LOGICAL_ADDR 0 29
regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x167d
# 	LOGICAL_ADDR 0 29
regGCRD_CREDIT_SAFE = 0x198a
# 	GCRD_CHAIN_CREDIT_SAFE_REG 0 2
# 	GCRD_TARGET_CREDIT_SAFE_REG 4 6
regGCRD_SA0_TARGETS_DISABLE = 0x1987
# 	GCRD_SA0_TARGETS_DISABLE 0 18
regGCRD_SA1_TARGETS_DISABLE = 0x1989
# 	GCRD_SA1_TARGETS_DISABLE 0 18
regGCR_CMD_STATUS = 0x1992
# 	GCR_CONTROL 0 18
# 	GCR_SRC 19 21
# 	GCR_TLB_SHOOTDOWN 23 23
# 	GCR_TLB_SHOOTDOWN_VMID 24 27
# 	UTCL2_NACK_STATUS 28 29
# 	GCR_SEQ_OP_ERROR 30 30
# 	UTCL2_NACK_ERROR 31 31
regGCR_GENERAL_CNTL = 0x1990
# 	FORCE_4K_L2_RESP 0 0
# 	REDUCE_HALF_MAIN_WQ 1 1
# 	REDUCE_HALF_PHY_WQ 2 2
# 	FORCE_INV_ALL 3 3
# 	HI_PRIORITY_CNTL 4 5
# 	HI_PRIORITY_DISABLE 6 6
# 	BIG_PAGE_FILTER_DISABLE 7 7
# 	PERF_CNTR_ENABLE 8 8
# 	FORCE_SINGLE_WQ 9 9
# 	UTCL2_REQ_PERM 10 12
# 	TARGET_MGCG_CLKEN_DIS 13 13
# 	MIXED_RANGE_MODE_DIS 14 14
# 	ENABLE_16K_UTCL2_REQ 15 15
# 	DISABLE_FGCG 16 16
# 	CLIENT_ID 20 28
regGCR_PERFCOUNTER0_HI = 0x3521
# 	PERFCOUNTER_HI 0 31
regGCR_PERFCOUNTER0_LO = 0x3520
# 	PERFCOUNTER_LO 0 31
regGCR_PERFCOUNTER0_SELECT = 0x3d60
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGCR_PERFCOUNTER0_SELECT1 = 0x3d61
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGCR_PERFCOUNTER1_HI = 0x3523
# 	PERFCOUNTER_HI 0 31
regGCR_PERFCOUNTER1_LO = 0x3522
# 	PERFCOUNTER_LO 0 31
regGCR_PERFCOUNTER1_SELECT = 0x3d62
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGCR_PIO_CNTL = 0x1580
# 	GCR_DATA_INDEX 0 1
# 	GCR_REG_DONE 2 2
# 	GCR_REG_RESET 3 3
# 	GCR_PIO_RSP_TAG 16 23
# 	GCR_PIO_RSP_DONE 30 30
# 	GCR_READY 31 31
regGCR_PIO_DATA = 0x1581
# 	GCR_DATA 0 31
regGCR_SPARE = 0x1993
# 	SPARE_BIT_1 1 1
# 	SPARE_BIT_2 2 2
# 	SPARE_BIT_3 3 3
# 	SPARE_BIT_4 4 4
# 	SPARE_BIT_5 5 5
# 	SPARE_BIT_6 6 6
# 	SPARE_BIT_7 7 7
# 	UTCL2_REQ_CREDIT 8 15
# 	GCRD_GL2A_REQ_CREDIT 16 19
# 	GCRD_SE_REQ_CREDIT 20 23
# 	SPARE_BIT_31_24 24 31
regGCUTCL2_CGTT_BUSY_CTRL = 0x15b7
# 	READ_DELAY 0 4
# 	ALWAYS_BUSY 5 5
regGCUTCL2_CREDIT_SAFETY_GROUP_CLIENTS_INVREQ_CDC = 0x15eb
# 	CREDITS 0 9
# 	UPDATE 10 10
regGCUTCL2_CREDIT_SAFETY_GROUP_CLIENTS_INVREQ_NOCDC = 0x15ec
# 	CREDITS 0 9
# 	UPDATE 10 10
regGCUTCL2_CREDIT_SAFETY_GROUP_RET_CDC = 0x15ea
# 	CREDITS 0 9
# 	UPDATE 10 10
regGCUTCL2_GROUP_RET_FAULT_STATUS = 0x15bb
# 	FAULT_GROUPS 0 31
regGCUTCL2_HARVEST_BYPASS_GROUPS = 0x15b9
# 	BYPASS_GROUPS 0 31
regGCUTCL2_ICG_CTRL = 0x15b5
# 	OFF_HYSTERESIS 0 3
# 	DYNAMIC_CLOCK_OVERRIDE 4 4
# 	STATIC_CLOCK_OVERRIDE 5 5
# 	AON_CLOCK_OVERRIDE 6 6
# 	PERFMON_CLOCK_OVERRIDE 7 7
regGCUTCL2_PERFCOUNTER0_CFG = 0x3d39
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCUTCL2_PERFCOUNTER1_CFG = 0x3d3a
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCUTCL2_PERFCOUNTER2_CFG = 0x3d3b
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCUTCL2_PERFCOUNTER3_CFG = 0x3d3c
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGCUTCL2_PERFCOUNTER_HI = 0x34e7
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regGCUTCL2_PERFCOUNTER_LO = 0x34e6
# 	COUNTER_LO 0 31
regGCUTCL2_PERFCOUNTER_RSLT_CNTL = 0x3d3d
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regGCUTCL2_TRANSLATION_BYPASS_BY_VMID = 0x5e41
# 	TRANS_BYPASS_VMIDS 0 15
# 	GPA_MODE_VMIDS 16 31
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_CNTL = 0x5e44
# 	ENABLE 0 0
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_HI = 0x15e6
# 	ADDR 0 3
# 	VMID 4 7
# 	VFID 8 11
# 	VF 12 12
# 	GPA 13 14
# 	RD_PERM 15 15
# 	WR_PERM 16 16
# 	EX_PERM 17 17
# 	CLIENT_ID 18 26
# 	REQ 30 30
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_LO = 0x15e5
# 	ADDR 0 31
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI = 0x15e8
# 	ADDR 0 3
# 	PERMS 4 6
# 	FRAGMENT_SIZE 7 12
# 	SNOOP 13 13
# 	SPA 14 14
# 	IO 15 15
# 	PTE_TMZ 16 16
# 	NO_PTE 17 17
# 	MTYPE 18 20
# 	MEMLOG 21 21
# 	NACK 22 23
# 	LLC_NOALLOC 24 24
# 	ACK 31 31
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_LO = 0x15e7
# 	ADDR 0 31
regGCUTC_TRANSLATION_FAULT_CNTL0 = 0x5eb8
# 	DEFAULT_PHYSICAL_PAGE_ADDRESS_LSB 0 31
regGCUTC_TRANSLATION_FAULT_CNTL1 = 0x5eb9
# 	DEFAULT_PHYSICAL_PAGE_ADDRESS_MSB 0 3
# 	DEFAULT_IO 4 4
# 	DEFAULT_SPA 5 5
# 	DEFAULT_SNOOP 6 6
regGCVML2_CREDIT_SAFETY_IH_FAULT_INTERRUPT = 0x15ed
# 	CREDITS 0 9
# 	UPDATE 10 10
regGCVML2_PERFCOUNTER2_0_HI = 0x34e2
# 	PERFCOUNTER_HI 0 31
regGCVML2_PERFCOUNTER2_0_LO = 0x34e0
# 	PERFCOUNTER_LO 0 31
regGCVML2_PERFCOUNTER2_0_MODE = 0x3d24
# 	COMPARE_MODE0 0 1
# 	COMPARE_MODE1 2 3
# 	COMPARE_MODE2 4 5
# 	COMPARE_MODE3 6 7
# 	COMPARE_VALUE0 8 11
# 	COMPARE_VALUE1 12 15
# 	COMPARE_VALUE2 16 19
# 	COMPARE_VALUE3 20 23
regGCVML2_PERFCOUNTER2_0_SELECT = 0x3d20
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGCVML2_PERFCOUNTER2_0_SELECT1 = 0x3d22
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGCVML2_PERFCOUNTER2_1_HI = 0x34e3
# 	PERFCOUNTER_HI 0 31
regGCVML2_PERFCOUNTER2_1_LO = 0x34e1
# 	PERFCOUNTER_LO 0 31
regGCVML2_PERFCOUNTER2_1_MODE = 0x3d25
# 	COMPARE_MODE0 0 1
# 	COMPARE_MODE1 2 3
# 	COMPARE_MODE2 4 5
# 	COMPARE_MODE3 6 7
# 	COMPARE_VALUE0 8 11
# 	COMPARE_VALUE1 12 15
# 	COMPARE_VALUE2 16 19
# 	COMPARE_VALUE3 20 23
regGCVML2_PERFCOUNTER2_1_SELECT = 0x3d21
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGCVML2_PERFCOUNTER2_1_SELECT1 = 0x3d23
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGCVML2_WALKER_CREDIT_SAFETY_FETCH_RDREQ = 0x15ee
# 	CREDITS 0 9
# 	UPDATE 10 10
regGCVML2_WALKER_MACRO_THROTTLE_FETCH_LIMIT = 0x15dd
# 	LIMIT 1 15
regGCVML2_WALKER_MACRO_THROTTLE_TIME = 0x15dc
# 	TIME 0 23
regGCVML2_WALKER_MICRO_THROTTLE_FETCH_LIMIT = 0x15df
# 	LIMIT 1 15
regGCVML2_WALKER_MICRO_THROTTLE_TIME = 0x15de
# 	TIME 0 23
regGCVM_CONTEXT0_CNTL = 0x1688
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f4
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f3
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x1734
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x1733
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x1714
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x1713
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT10_CNTL = 0x1692
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT10_PAGE_TABLE_BASE_ADDR_HI32 = 0x1708
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT10_PAGE_TABLE_BASE_ADDR_LO32 = 0x1707
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT10_PAGE_TABLE_END_ADDR_HI32 = 0x1748
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT10_PAGE_TABLE_END_ADDR_LO32 = 0x1747
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT10_PAGE_TABLE_START_ADDR_HI32 = 0x1728
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT10_PAGE_TABLE_START_ADDR_LO32 = 0x1727
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT11_CNTL = 0x1693
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT11_PAGE_TABLE_BASE_ADDR_HI32 = 0x170a
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT11_PAGE_TABLE_BASE_ADDR_LO32 = 0x1709
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT11_PAGE_TABLE_END_ADDR_HI32 = 0x174a
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT11_PAGE_TABLE_END_ADDR_LO32 = 0x1749
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT11_PAGE_TABLE_START_ADDR_HI32 = 0x172a
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT11_PAGE_TABLE_START_ADDR_LO32 = 0x1729
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT12_CNTL = 0x1694
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT12_PAGE_TABLE_BASE_ADDR_HI32 = 0x170c
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT12_PAGE_TABLE_BASE_ADDR_LO32 = 0x170b
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT12_PAGE_TABLE_END_ADDR_HI32 = 0x174c
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT12_PAGE_TABLE_END_ADDR_LO32 = 0x174b
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT12_PAGE_TABLE_START_ADDR_HI32 = 0x172c
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT12_PAGE_TABLE_START_ADDR_LO32 = 0x172b
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT13_CNTL = 0x1695
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_HI32 = 0x170e
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_LO32 = 0x170d
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_HI32 = 0x174e
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32 = 0x174d
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_HI32 = 0x172e
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_LO32 = 0x172d
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT14_CNTL = 0x1696
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT14_PAGE_TABLE_BASE_ADDR_HI32 = 0x1710
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT14_PAGE_TABLE_BASE_ADDR_LO32 = 0x170f
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT14_PAGE_TABLE_END_ADDR_HI32 = 0x1750
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT14_PAGE_TABLE_END_ADDR_LO32 = 0x174f
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT14_PAGE_TABLE_START_ADDR_HI32 = 0x1730
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT14_PAGE_TABLE_START_ADDR_LO32 = 0x172f
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT15_CNTL = 0x1697
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT15_PAGE_TABLE_BASE_ADDR_HI32 = 0x1712
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT15_PAGE_TABLE_BASE_ADDR_LO32 = 0x1711
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT15_PAGE_TABLE_END_ADDR_HI32 = 0x1752
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT15_PAGE_TABLE_END_ADDR_LO32 = 0x1751
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT15_PAGE_TABLE_START_ADDR_HI32 = 0x1732
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT15_PAGE_TABLE_START_ADDR_LO32 = 0x1731
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT1_CNTL = 0x1689
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f6
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f5
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 = 0x1736
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x1735
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 = 0x1716
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x1715
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT2_CNTL = 0x168a
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT2_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f8
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT2_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f7
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT2_PAGE_TABLE_END_ADDR_HI32 = 0x1738
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT2_PAGE_TABLE_END_ADDR_LO32 = 0x1737
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT2_PAGE_TABLE_START_ADDR_HI32 = 0x1718
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT2_PAGE_TABLE_START_ADDR_LO32 = 0x1717
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT3_CNTL = 0x168b
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT3_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fa
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT3_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f9
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT3_PAGE_TABLE_END_ADDR_HI32 = 0x173a
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT3_PAGE_TABLE_END_ADDR_LO32 = 0x1739
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT3_PAGE_TABLE_START_ADDR_HI32 = 0x171a
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT3_PAGE_TABLE_START_ADDR_LO32 = 0x1719
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT4_CNTL = 0x168c
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT4_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fc
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT4_PAGE_TABLE_BASE_ADDR_LO32 = 0x16fb
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT4_PAGE_TABLE_END_ADDR_HI32 = 0x173c
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT4_PAGE_TABLE_END_ADDR_LO32 = 0x173b
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT4_PAGE_TABLE_START_ADDR_HI32 = 0x171c
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT4_PAGE_TABLE_START_ADDR_LO32 = 0x171b
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT5_CNTL = 0x168d
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT5_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fe
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT5_PAGE_TABLE_BASE_ADDR_LO32 = 0x16fd
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT5_PAGE_TABLE_END_ADDR_HI32 = 0x173e
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT5_PAGE_TABLE_END_ADDR_LO32 = 0x173d
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT5_PAGE_TABLE_START_ADDR_HI32 = 0x171e
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT5_PAGE_TABLE_START_ADDR_LO32 = 0x171d
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT6_CNTL = 0x168e
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT6_PAGE_TABLE_BASE_ADDR_HI32 = 0x1700
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT6_PAGE_TABLE_BASE_ADDR_LO32 = 0x16ff
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT6_PAGE_TABLE_END_ADDR_HI32 = 0x1740
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT6_PAGE_TABLE_END_ADDR_LO32 = 0x173f
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT6_PAGE_TABLE_START_ADDR_HI32 = 0x1720
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT6_PAGE_TABLE_START_ADDR_LO32 = 0x171f
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT7_CNTL = 0x168f
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT7_PAGE_TABLE_BASE_ADDR_HI32 = 0x1702
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT7_PAGE_TABLE_BASE_ADDR_LO32 = 0x1701
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT7_PAGE_TABLE_END_ADDR_HI32 = 0x1742
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT7_PAGE_TABLE_END_ADDR_LO32 = 0x1741
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT7_PAGE_TABLE_START_ADDR_HI32 = 0x1722
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT7_PAGE_TABLE_START_ADDR_LO32 = 0x1721
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT8_CNTL = 0x1690
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_HI32 = 0x1704
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_LO32 = 0x1703
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT8_PAGE_TABLE_END_ADDR_HI32 = 0x1744
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT8_PAGE_TABLE_END_ADDR_LO32 = 0x1743
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT8_PAGE_TABLE_START_ADDR_HI32 = 0x1724
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT8_PAGE_TABLE_START_ADDR_LO32 = 0x1723
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT9_CNTL = 0x1691
# 	ENABLE_CONTEXT 0 0
# 	PAGE_TABLE_DEPTH 1 2
# 	PAGE_TABLE_BLOCK_SIZE 3 6
# 	RETRY_PERMISSION_OR_INVALID_PAGE_FAULT 7 7
# 	RETRY_OTHER_FAULT 8 8
# 	RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT 9 9
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT 11 11
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT 13 13
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 14 14
# 	VALID_PROTECTION_FAULT_ENABLE_INTERRUPT 15 15
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 16 16
# 	READ_PROTECTION_FAULT_ENABLE_INTERRUPT 17 17
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 18 18
# 	WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT 19 19
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 20 20
# 	EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT 21 21
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 22 22
regGCVM_CONTEXT9_PAGE_TABLE_BASE_ADDR_HI32 = 0x1706
# 	PAGE_DIRECTORY_ENTRY_HI32 0 31
regGCVM_CONTEXT9_PAGE_TABLE_BASE_ADDR_LO32 = 0x1705
# 	PAGE_DIRECTORY_ENTRY_LO32 0 31
regGCVM_CONTEXT9_PAGE_TABLE_END_ADDR_HI32 = 0x1746
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT9_PAGE_TABLE_END_ADDR_LO32 = 0x1745
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXT9_PAGE_TABLE_START_ADDR_HI32 = 0x1726
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_CONTEXT9_PAGE_TABLE_START_ADDR_LO32 = 0x1725
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_CONTEXTS_DISABLE = 0x1698
# 	DISABLE_CONTEXT_0 0 0
# 	DISABLE_CONTEXT_1 1 1
# 	DISABLE_CONTEXT_2 2 2
# 	DISABLE_CONTEXT_3 3 3
# 	DISABLE_CONTEXT_4 4 4
# 	DISABLE_CONTEXT_5 5 5
# 	DISABLE_CONTEXT_6 6 6
# 	DISABLE_CONTEXT_7 7 7
# 	DISABLE_CONTEXT_8 8 8
# 	DISABLE_CONTEXT_9 9 9
# 	DISABLE_CONTEXT_10 10 10
# 	DISABLE_CONTEXT_11 11 11
# 	DISABLE_CONTEXT_12 12 12
# 	DISABLE_CONTEXT_13 13 13
# 	DISABLE_CONTEXT_14 14 14
# 	DISABLE_CONTEXT_15 15 15
regGCVM_DUMMY_PAGE_FAULT_ADDR_HI32 = 0x15c2
# 	DUMMY_PAGE_ADDR_HI4 0 3
regGCVM_DUMMY_PAGE_FAULT_ADDR_LO32 = 0x15c1
# 	DUMMY_PAGE_ADDR_LO32 0 31
regGCVM_DUMMY_PAGE_FAULT_CNTL = 0x15c0
# 	DUMMY_PAGE_FAULT_ENABLE 0 0
# 	DUMMY_PAGE_ADDRESS_LOGICAL 1 1
# 	DUMMY_PAGE_COMPARE_MSBS 2 7
regGCVM_INVALIDATE_CNTL = 0x15c3
# 	PRI_REG_ALTERNATING 0 7
# 	MAX_REG_OUTSTANDING 8 15
regGCVM_INVALIDATE_ENG0_ACK = 0x16bd
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x16d0
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x16cf
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG0_REQ = 0x16ab
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG0_SEM = 0x1699
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG10_ACK = 0x16c7
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG10_ADDR_RANGE_HI32 = 0x16e4
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG10_ADDR_RANGE_LO32 = 0x16e3
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG10_REQ = 0x16b5
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG10_SEM = 0x16a3
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG11_ACK = 0x16c8
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG11_ADDR_RANGE_HI32 = 0x16e6
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG11_ADDR_RANGE_LO32 = 0x16e5
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG11_REQ = 0x16b6
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG11_SEM = 0x16a4
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG12_ACK = 0x16c9
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG12_ADDR_RANGE_HI32 = 0x16e8
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG12_ADDR_RANGE_LO32 = 0x16e7
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG12_REQ = 0x16b7
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG12_SEM = 0x16a5
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG13_ACK = 0x16ca
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG13_ADDR_RANGE_HI32 = 0x16ea
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG13_ADDR_RANGE_LO32 = 0x16e9
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG13_REQ = 0x16b8
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG13_SEM = 0x16a6
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG14_ACK = 0x16cb
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG14_ADDR_RANGE_HI32 = 0x16ec
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG14_ADDR_RANGE_LO32 = 0x16eb
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG14_REQ = 0x16b9
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG14_SEM = 0x16a7
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG15_ACK = 0x16cc
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG15_ADDR_RANGE_HI32 = 0x16ee
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG15_ADDR_RANGE_LO32 = 0x16ed
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG15_REQ = 0x16ba
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG15_SEM = 0x16a8
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG16_ACK = 0x16cd
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG16_ADDR_RANGE_HI32 = 0x16f0
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG16_ADDR_RANGE_LO32 = 0x16ef
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG16_REQ = 0x16bb
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG16_SEM = 0x16a9
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG17_ACK = 0x16ce
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG17_ADDR_RANGE_HI32 = 0x16f2
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG17_ADDR_RANGE_LO32 = 0x16f1
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG17_REQ = 0x16bc
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG17_SEM = 0x16aa
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG1_ACK = 0x16be
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG1_ADDR_RANGE_HI32 = 0x16d2
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG1_ADDR_RANGE_LO32 = 0x16d1
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG1_REQ = 0x16ac
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG1_SEM = 0x169a
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG2_ACK = 0x16bf
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG2_ADDR_RANGE_HI32 = 0x16d4
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG2_ADDR_RANGE_LO32 = 0x16d3
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG2_REQ = 0x16ad
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG2_SEM = 0x169b
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG3_ACK = 0x16c0
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG3_ADDR_RANGE_HI32 = 0x16d6
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG3_ADDR_RANGE_LO32 = 0x16d5
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG3_REQ = 0x16ae
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG3_SEM = 0x169c
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG4_ACK = 0x16c1
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG4_ADDR_RANGE_HI32 = 0x16d8
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG4_ADDR_RANGE_LO32 = 0x16d7
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG4_REQ = 0x16af
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG4_SEM = 0x169d
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG5_ACK = 0x16c2
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG5_ADDR_RANGE_HI32 = 0x16da
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG5_ADDR_RANGE_LO32 = 0x16d9
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG5_REQ = 0x16b0
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG5_SEM = 0x169e
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG6_ACK = 0x16c3
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG6_ADDR_RANGE_HI32 = 0x16dc
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG6_ADDR_RANGE_LO32 = 0x16db
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG6_REQ = 0x16b1
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG6_SEM = 0x169f
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG7_ACK = 0x16c4
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG7_ADDR_RANGE_HI32 = 0x16de
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG7_ADDR_RANGE_LO32 = 0x16dd
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG7_REQ = 0x16b2
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG7_SEM = 0x16a0
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG8_ACK = 0x16c5
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG8_ADDR_RANGE_HI32 = 0x16e0
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG8_ADDR_RANGE_LO32 = 0x16df
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG8_REQ = 0x16b3
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG8_SEM = 0x16a1
# 	SEMAPHORE 0 0
regGCVM_INVALIDATE_ENG9_ACK = 0x16c6
# 	PER_VMID_INVALIDATE_ACK 0 15
# 	SEMAPHORE 16 16
regGCVM_INVALIDATE_ENG9_ADDR_RANGE_HI32 = 0x16e2
# 	LOGI_PAGE_ADDR_RANGE_HI5 0 4
regGCVM_INVALIDATE_ENG9_ADDR_RANGE_LO32 = 0x16e1
# 	S_BIT 0 0
# 	LOGI_PAGE_ADDR_RANGE_LO31 1 31
regGCVM_INVALIDATE_ENG9_REQ = 0x16b4
# 	PER_VMID_INVALIDATE_REQ 0 15
# 	FLUSH_TYPE 16 18
# 	INVALIDATE_L2_PTES 19 19
# 	INVALIDATE_L2_PDE0 20 20
# 	INVALIDATE_L2_PDE1 21 21
# 	INVALIDATE_L2_PDE2 22 22
# 	INVALIDATE_L1_PTES 23 23
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 24 24
# 	LOG_REQUEST 25 25
# 	INVALIDATE_4K_PAGES_ONLY 26 26
regGCVM_INVALIDATE_ENG9_SEM = 0x16a2
# 	SEMAPHORE 0 0
regGCVM_L2_BANK_SELECT_MASKS = 0x15e9
# 	MASK0 0 3
# 	MASK1 4 7
# 	MASK2 8 11
# 	MASK3 12 15
regGCVM_L2_BANK_SELECT_RESERVED_CID = 0x15d6
# 	RESERVED_READ_CLIENT_ID 0 8
# 	RESERVED_WRITE_CLIENT_ID 10 18
# 	ENABLE 20 20
# 	RESERVED_CACHE_INVALIDATION_MODE 24 24
# 	RESERVED_CACHE_PRIVATE_INVALIDATION 25 25
# 	RESERVED_CACHE_FRAGMENT_SIZE 26 30
regGCVM_L2_BANK_SELECT_RESERVED_CID2 = 0x15d7
# 	RESERVED_READ_CLIENT_ID 0 8
# 	RESERVED_WRITE_CLIENT_ID 10 18
# 	ENABLE 20 20
# 	RESERVED_CACHE_INVALIDATION_MODE 24 24
# 	RESERVED_CACHE_PRIVATE_INVALIDATION 25 25
# 	RESERVED_CACHE_FRAGMENT_SIZE 26 30
regGCVM_L2_CACHE_PARITY_CNTL = 0x15d8
# 	ENABLE_PARITY_CHECKS_IN_4K_PTE_CACHES 0 0
# 	ENABLE_PARITY_CHECKS_IN_BIGK_PTE_CACHES 1 1
# 	ENABLE_PARITY_CHECKS_IN_PDE_CACHES 2 2
# 	FORCE_PARITY_MISMATCH_IN_4K_PTE_CACHE 3 3
# 	FORCE_PARITY_MISMATCH_IN_BIGK_PTE_CACHE 4 4
# 	FORCE_PARITY_MISMATCH_IN_PDE_CACHE 5 5
# 	FORCE_CACHE_BANK 6 8
# 	FORCE_CACHE_NUMBER 9 11
# 	FORCE_CACHE_ASSOC 12 15
regGCVM_L2_CGTT_BUSY_CTRL = 0x15e0
# 	READ_DELAY 0 4
# 	ALWAYS_BUSY 5 5
regGCVM_L2_CNTL = 0x15bc
# 	ENABLE_L2_CACHE 0 0
# 	ENABLE_L2_FRAGMENT_PROCESSING 1 1
# 	L2_CACHE_PTE_ENDIAN_SWAP_MODE 2 3
# 	L2_CACHE_PDE_ENDIAN_SWAP_MODE 4 5
# 	L2_PDE0_CACHE_TAG_GENERATION_MODE 8 8
# 	ENABLE_L2_PTE_CACHE_LRU_UPDATE_BY_WRITE 9 9
# 	ENABLE_L2_PDE0_CACHE_LRU_UPDATE_BY_WRITE 10 10
# 	ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM_MEMORY 11 11
# 	L2_PDE0_CACHE_SPLIT_MODE 12 14
# 	EFFECTIVE_L2_QUEUE_SIZE 15 17
# 	PDE_FAULT_CLASSIFICATION 18 18
# 	CONTEXT1_IDENTITY_ACCESS_MODE 19 20
# 	IDENTITY_MODE_FRAGMENT_SIZE 21 25
# 	L2_PTE_CACHE_ADDR_MODE 26 27
regGCVM_L2_CNTL2 = 0x15bd
# 	INVALIDATE_ALL_L1_TLBS 0 0
# 	INVALIDATE_L2_CACHE 1 1
# 	DISABLE_INVALIDATE_PER_DOMAIN 21 21
# 	DISABLE_BIGK_CACHE_OPTIMIZATION 22 22
# 	L2_PTE_CACHE_VMID_MODE 23 25
# 	INVALIDATE_CACHE_MODE 26 27
# 	PDE_CACHE_EFFECTIVE_SIZE 28 30
regGCVM_L2_CNTL3 = 0x15be
# 	BANK_SELECT 0 5
# 	L2_CACHE_UPDATE_MODE 6 7
# 	L2_CACHE_UPDATE_WILDCARD_REFERENCE_VALUE 8 12
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 15 19
# 	L2_CACHE_BIGK_ASSOCIATIVITY 20 20
# 	L2_CACHE_4K_EFFECTIVE_SIZE 21 23
# 	L2_CACHE_BIGK_EFFECTIVE_SIZE 24 27
# 	L2_CACHE_4K_FORCE_MISS 28 28
# 	L2_CACHE_BIGK_FORCE_MISS 29 29
# 	PDE_CACHE_FORCE_MISS 30 30
# 	L2_CACHE_4K_ASSOCIATIVITY 31 31
regGCVM_L2_CNTL4 = 0x15d4
# 	L2_CACHE_4K_PARTITION_COUNT 0 5
# 	VMC_TAP_PDE_REQUEST_PHYSICAL 6 6
# 	VMC_TAP_PTE_REQUEST_PHYSICAL 7 7
# 	MM_NONRT_IFIFO_ACTIVE_TRANSACTION_LIMIT 8 17
# 	MM_SOFTRT_IFIFO_ACTIVE_TRANSACTION_LIMIT 18 27
# 	BPM_CGCGLS_OVERRIDE 28 28
# 	GC_CH_FGCG_OFF 29 29
# 	VFIFO_HEAD_OF_QUEUE 30 30
# 	VFIFO_VISIBLE_BANK_SILOS 31 31
regGCVM_L2_CNTL5 = 0x15da
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	WALKER_PRIORITY_CLIENT_ID 5 13
# 	WALKER_FETCH_PDE_NOALLOC_ENABLE 14 14
# 	WALKER_FETCH_PDE_MTYPE_ENABLE 15 15
# 	UTCL2_ATC_REQ_FGCG_OFF 16 16
regGCVM_L2_CONTEXT0_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1754
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT10_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175e
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT11_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175f
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT12_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1760
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT13_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1761
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT14_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1762
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT15_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1763
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 = 0x15d1
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 = 0x15d0
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32 = 0x15cf
# 	LOGICAL_PAGE_NUMBER_HI4 0 3
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32 = 0x15ce
# 	LOGICAL_PAGE_NUMBER_LO32 0 31
regGCVM_L2_CONTEXT1_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1755
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT2_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1756
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT3_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1757
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT4_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1758
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT5_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1759
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT6_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175a
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT7_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175b
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT8_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175c
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT9_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175d
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32 = 0x15d3
# 	PHYSICAL_PAGE_OFFSET_HI4 0 3
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32 = 0x15d2
# 	PHYSICAL_PAGE_OFFSET_LO32 0 31
regGCVM_L2_GCR_CNTL = 0x15db
# 	GCR_ENABLE 0 0
# 	GCR_CLIENT_ID 1 9
regGCVM_L2_ICG_CTRL = 0x15d9
# 	OFF_HYSTERESIS 0 3
# 	DYNAMIC_CLOCK_OVERRIDE 4 4
# 	STATIC_CLOCK_OVERRIDE 5 5
# 	AON_CLOCK_OVERRIDE 6 6
# 	PERFMON_CLOCK_OVERRIDE 7 7
regGCVM_L2_MM_GROUP_RT_CLASSES = 0x15d5
# 	GROUP_0_RT_CLASS 0 0
# 	GROUP_1_RT_CLASS 1 1
# 	GROUP_2_RT_CLASS 2 2
# 	GROUP_3_RT_CLASS 3 3
# 	GROUP_4_RT_CLASS 4 4
# 	GROUP_5_RT_CLASS 5 5
# 	GROUP_6_RT_CLASS 6 6
# 	GROUP_7_RT_CLASS 7 7
# 	GROUP_8_RT_CLASS 8 8
# 	GROUP_9_RT_CLASS 9 9
# 	GROUP_10_RT_CLASS 10 10
# 	GROUP_11_RT_CLASS 11 11
# 	GROUP_12_RT_CLASS 12 12
# 	GROUP_13_RT_CLASS 13 13
# 	GROUP_14_RT_CLASS 14 14
# 	GROUP_15_RT_CLASS 15 15
# 	GROUP_16_RT_CLASS 16 16
# 	GROUP_17_RT_CLASS 17 17
# 	GROUP_18_RT_CLASS 18 18
# 	GROUP_19_RT_CLASS 19 19
# 	GROUP_20_RT_CLASS 20 20
# 	GROUP_21_RT_CLASS 21 21
# 	GROUP_22_RT_CLASS 22 22
# 	GROUP_23_RT_CLASS 23 23
# 	GROUP_24_RT_CLASS 24 24
# 	GROUP_25_RT_CLASS 25 25
# 	GROUP_26_RT_CLASS 26 26
# 	GROUP_27_RT_CLASS 27 27
# 	GROUP_28_RT_CLASS 28 28
# 	GROUP_29_RT_CLASS 29 29
# 	GROUP_30_RT_CLASS 30 30
# 	GROUP_31_RT_CLASS 31 31
regGCVM_L2_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1753
# 	L2_CACHE_SMALLK_FRAGMENT_SIZE 0 4
# 	L2_CACHE_BIGK_FRAGMENT_SIZE 5 9
# 	BANK_SELECT 10 15
regGCVM_L2_PROTECTION_FAULT_ADDR_HI32 = 0x15ca
# 	LOGICAL_PAGE_ADDR_HI4 0 3
regGCVM_L2_PROTECTION_FAULT_ADDR_LO32 = 0x15c9
# 	LOGICAL_PAGE_ADDR_LO32 0 31
regGCVM_L2_PROTECTION_FAULT_CNTL = 0x15c4
# 	CLEAR_PROTECTION_FAULT_STATUS_ADDR 0 0
# 	ALLOW_SUBSEQUENT_PROTECTION_FAULT_STATUS_ADDR_UPDATES 1 1
# 	RANGE_PROTECTION_FAULT_ENABLE_DEFAULT 2 2
# 	PDE0_PROTECTION_FAULT_ENABLE_DEFAULT 3 3
# 	PDE1_PROTECTION_FAULT_ENABLE_DEFAULT 4 4
# 	PDE2_PROTECTION_FAULT_ENABLE_DEFAULT 5 5
# 	TRANSLATE_FURTHER_PROTECTION_FAULT_ENABLE_DEFAULT 6 6
# 	NACK_PROTECTION_FAULT_ENABLE_DEFAULT 7 7
# 	DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT 8 8
# 	VALID_PROTECTION_FAULT_ENABLE_DEFAULT 9 9
# 	READ_PROTECTION_FAULT_ENABLE_DEFAULT 10 10
# 	WRITE_PROTECTION_FAULT_ENABLE_DEFAULT 11 11
# 	EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT 12 12
# 	CLIENT_ID_NO_RETRY_FAULT_INTERRUPT 13 28
# 	OTHER_CLIENT_ID_NO_RETRY_FAULT_INTERRUPT 29 29
# 	CRASH_ON_NO_RETRY_FAULT 30 30
# 	CRASH_ON_RETRY_FAULT 31 31
regGCVM_L2_PROTECTION_FAULT_CNTL2 = 0x15c5
# 	CLIENT_ID_PRT_FAULT_INTERRUPT 0 15
# 	OTHER_CLIENT_ID_PRT_FAULT_INTERRUPT 16 16
# 	ACTIVE_PAGE_MIGRATION_PTE 17 17
# 	ACTIVE_PAGE_MIGRATION_PTE_READ_RETRY 18 18
# 	ENABLE_RETRY_FAULT_INTERRUPT 19 19
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x15cc
# 	PHYSICAL_PAGE_ADDR_HI4 0 3
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x15cb
# 	PHYSICAL_PAGE_ADDR_LO32 0 31
regGCVM_L2_PROTECTION_FAULT_MM_CNTL3 = 0x15c6
# 	VML1_READ_CLIENT_ID_NO_RETRY_FAULT_INTERRUPT 0 31
regGCVM_L2_PROTECTION_FAULT_MM_CNTL4 = 0x15c7
# 	VML1_WRITE_CLIENT_ID_NO_RETRY_FAULT_INTERRUPT 0 31
regGCVM_L2_PROTECTION_FAULT_STATUS = 0x15c8
# 	MORE_FAULTS 0 0
# 	WALKER_ERROR 1 3
# 	PERMISSION_FAULTS 4 7
# 	MAPPING_ERROR 8 8
# 	CID 9 17
# 	RW 18 18
# 	ATOMIC 19 19
# 	VMID 20 23
# 	VF 24 24
# 	VFID 25 28
# 	PRT 29 29
regGCVM_L2_PTE_CACHE_DUMP_CNTL = 0x15e1
# 	ENABLE 0 0
# 	READY 1 1
# 	BANK 4 7
# 	CACHE 8 11
# 	ASSOC 12 15
# 	INDEX 16 31
regGCVM_L2_PTE_CACHE_DUMP_READ = 0x15e2
# 	DATA 0 31
regGCVM_L2_STATUS = 0x15bf
# 	L2_BUSY 0 0
# 	CONTEXT_DOMAIN_BUSY 1 16
# 	FOUND_4K_PTE_CACHE_PARITY_ERRORS 17 17
# 	FOUND_BIGK_PTE_CACHE_PARITY_ERRORS 18 18
# 	FOUND_PDE0_CACHE_PARITY_ERRORS 19 19
# 	FOUND_PDE1_CACHE_PARITY_ERRORS 20 20
# 	FOUND_PDE2_CACHE_PARITY_ERRORS 21 21
regGC_CAC_AGGR_GFXCLK_CYCLE = 0x1ae4
# 	GC_AGGR_GFXCLK_CYCLE 0 31
regGC_CAC_AGGR_LOWER = 0x1ad2
# 	GC_AGGR_31_0 0 31
regGC_CAC_AGGR_UPPER = 0x1ad3
# 	GC_AGGR_63_32 0 31
regGC_CAC_CTRL_1 = 0x1ad0
# 	CAC_WINDOW 0 7
# 	TDP_WINDOW 8 31
regGC_CAC_CTRL_2 = 0x1ad1
# 	CAC_ENABLE 0 0
# 	GC_LCAC_ENABLE 1 1
# 	GC_CAC_INDEX_AUTO_INCR_EN 2 2
# 	TOGGLE_EN 3 3
# 	INTR_EN 4 4
# 	CAC_COUNTER_SNAP_SEL 5 5
# 	SE_AGGR_ACC_EN 6 13
# 	GC_AGGR_ACC_EN 14 14
regGC_CAC_IND_DATA = 0x1b59
# 	GC_CAC_IND_DATA 0 31
regGC_CAC_IND_INDEX = 0x1b58
# 	GC_CAC_IND_ADDR 0 31
regGC_CAC_WEIGHT_CHC_0 = 0x1b3c
# 	WEIGHT_CHC_SIG0 0 15
# 	WEIGHT_CHC_SIG1 16 31
regGC_CAC_WEIGHT_CHC_1 = 0x1b3d
# 	WEIGHT_CHC_SIG2 0 15
regGC_CAC_WEIGHT_CP_0 = 0x1b10
# 	WEIGHT_CP_SIG0 0 15
# 	WEIGHT_CP_SIG1 16 31
regGC_CAC_WEIGHT_CP_1 = 0x1b11
# 	WEIGHT_CP_SIG2 0 15
regGC_CAC_WEIGHT_EA_0 = 0x1b12
# 	WEIGHT_EA_SIG0 0 15
# 	WEIGHT_EA_SIG1 16 31
regGC_CAC_WEIGHT_EA_1 = 0x1b13
# 	WEIGHT_EA_SIG2 0 15
# 	WEIGHT_EA_SIG3 16 31
regGC_CAC_WEIGHT_EA_2 = 0x1b14
# 	WEIGHT_EA_SIG4 0 15
# 	WEIGHT_EA_SIG5 16 31
regGC_CAC_WEIGHT_GDS_0 = 0x1b20
# 	WEIGHT_GDS_SIG0 0 15
# 	WEIGHT_GDS_SIG1 16 31
regGC_CAC_WEIGHT_GDS_1 = 0x1b21
# 	WEIGHT_GDS_SIG2 0 15
# 	WEIGHT_GDS_SIG3 16 31
regGC_CAC_WEIGHT_GDS_2 = 0x1b22
# 	WEIGHT_GDS_SIG4 0 15
regGC_CAC_WEIGHT_GE_0 = 0x1b23
# 	WEIGHT_GE_SIG0 0 15
# 	WEIGHT_GE_SIG1 16 31
regGC_CAC_WEIGHT_GE_1 = 0x1b24
# 	WEIGHT_GE_SIG2 0 15
# 	WEIGHT_GE_SIG3 16 31
regGC_CAC_WEIGHT_GE_2 = 0x1b25
# 	WEIGHT_GE_SIG4 0 15
# 	WEIGHT_GE_SIG5 16 31
regGC_CAC_WEIGHT_GE_3 = 0x1b26
# 	WEIGHT_GE_SIG6 0 15
# 	WEIGHT_GE_SIG7 16 31
regGC_CAC_WEIGHT_GE_4 = 0x1b27
# 	WEIGHT_GE_SIG8 0 15
# 	WEIGHT_GE_SIG9 16 31
regGC_CAC_WEIGHT_GE_5 = 0x1b28
# 	WEIGHT_GE_SIG10 0 15
# 	WEIGHT_GE_SIG11 16 31
regGC_CAC_WEIGHT_GE_6 = 0x1b29
# 	WEIGHT_GE_SIG12 0 15
regGC_CAC_WEIGHT_GL2C_0 = 0x1b2f
# 	WEIGHT_GL2C_SIG0 0 15
# 	WEIGHT_GL2C_SIG1 16 31
regGC_CAC_WEIGHT_GL2C_1 = 0x1b30
# 	WEIGHT_GL2C_SIG2 0 15
# 	WEIGHT_GL2C_SIG3 16 31
regGC_CAC_WEIGHT_GL2C_2 = 0x1b31
# 	WEIGHT_GL2C_SIG4 0 15
regGC_CAC_WEIGHT_GRBM_0 = 0x1b44
# 	WEIGHT_GRBM_SIG0 0 15
# 	WEIGHT_GRBM_SIG1 16 31
regGC_CAC_WEIGHT_GUS_0 = 0x1b3e
# 	WEIGHT_GUS_SIG0 0 15
# 	WEIGHT_GUS_SIG1 16 31
regGC_CAC_WEIGHT_GUS_1 = 0x1b3f
# 	WEIGHT_GUS_SIG2 0 15
regGC_CAC_WEIGHT_PH_0 = 0x1b32
# 	WEIGHT_PH_SIG0 0 15
# 	WEIGHT_PH_SIG1 16 31
regGC_CAC_WEIGHT_PH_1 = 0x1b33
# 	WEIGHT_PH_SIG2 0 15
# 	WEIGHT_PH_SIG3 16 31
regGC_CAC_WEIGHT_PH_2 = 0x1b34
# 	WEIGHT_PH_SIG4 0 15
# 	WEIGHT_PH_SIG5 16 31
regGC_CAC_WEIGHT_PH_3 = 0x1b35
# 	WEIGHT_PH_SIG6 0 15
# 	WEIGHT_PH_SIG7 16 31
regGC_CAC_WEIGHT_PMM_0 = 0x1b2e
# 	WEIGHT_PMM_SIG0 0 15
regGC_CAC_WEIGHT_RLC_0 = 0x1b40
# 	WEIGHT_RLC_SIG0 0 15
regGC_CAC_WEIGHT_SDMA_0 = 0x1b36
# 	WEIGHT_SDMA_SIG0 0 15
# 	WEIGHT_SDMA_SIG1 16 31
regGC_CAC_WEIGHT_SDMA_1 = 0x1b37
# 	WEIGHT_SDMA_SIG2 0 15
# 	WEIGHT_SDMA_SIG3 16 31
regGC_CAC_WEIGHT_SDMA_2 = 0x1b38
# 	WEIGHT_SDMA_SIG4 0 15
# 	WEIGHT_SDMA_SIG5 16 31
regGC_CAC_WEIGHT_SDMA_3 = 0x1b39
# 	WEIGHT_SDMA_SIG6 0 15
# 	WEIGHT_SDMA_SIG7 16 31
regGC_CAC_WEIGHT_SDMA_4 = 0x1b3a
# 	WEIGHT_SDMA_SIG8 0 15
# 	WEIGHT_SDMA_SIG9 16 31
regGC_CAC_WEIGHT_SDMA_5 = 0x1b3b
# 	WEIGHT_SDMA_SIG10 0 15
# 	WEIGHT_SDMA_SIG11 16 31
regGC_CAC_WEIGHT_UTCL2_ROUTER_0 = 0x1b15
# 	WEIGHT_UTCL2_ROUTER_SIG0 0 15
# 	WEIGHT_UTCL2_ROUTER_SIG1 16 31
regGC_CAC_WEIGHT_UTCL2_ROUTER_1 = 0x1b16
# 	WEIGHT_UTCL2_ROUTER_SIG2 0 15
# 	WEIGHT_UTCL2_ROUTER_SIG3 16 31
regGC_CAC_WEIGHT_UTCL2_ROUTER_2 = 0x1b17
# 	WEIGHT_UTCL2_ROUTER_SIG4 0 15
# 	WEIGHT_UTCL2_ROUTER_SIG5 16 31
regGC_CAC_WEIGHT_UTCL2_ROUTER_3 = 0x1b18
# 	WEIGHT_UTCL2_ROUTER_SIG6 0 15
# 	WEIGHT_UTCL2_ROUTER_SIG7 16 31
regGC_CAC_WEIGHT_UTCL2_ROUTER_4 = 0x1b19
# 	WEIGHT_UTCL2_ROUTER_SIG8 0 15
# 	WEIGHT_UTCL2_ROUTER_SIG9 16 31
regGC_CAC_WEIGHT_UTCL2_VML2_0 = 0x1b1a
# 	WEIGHT_UTCL2_VML2_SIG0 0 15
# 	WEIGHT_UTCL2_VML2_SIG1 16 31
regGC_CAC_WEIGHT_UTCL2_VML2_1 = 0x1b1b
# 	WEIGHT_UTCL2_VML2_SIG2 0 15
# 	WEIGHT_UTCL2_VML2_SIG3 16 31
regGC_CAC_WEIGHT_UTCL2_VML2_2 = 0x1b1c
# 	WEIGHT_UTCL2_VML2_SIG4 0 15
regGC_CAC_WEIGHT_UTCL2_WALKER_0 = 0x1b1d
# 	WEIGHT_UTCL2_WALKER_SIG0 0 15
# 	WEIGHT_UTCL2_WALKER_SIG1 16 31
regGC_CAC_WEIGHT_UTCL2_WALKER_1 = 0x1b1e
# 	WEIGHT_UTCL2_WALKER_SIG2 0 15
# 	WEIGHT_UTCL2_WALKER_SIG3 16 31
regGC_CAC_WEIGHT_UTCL2_WALKER_2 = 0x1b1f
# 	WEIGHT_UTCL2_WALKER_SIG4 0 15
regGC_EDC_CLK_MONITOR_CTRL = 0x1b56
# 	EDC_CLK_MONITOR_EN 0 0
# 	EDC_CLK_MONITOR_INTERVAL 1 4
# 	EDC_CLK_MONITOR_THRESHOLD 5 16
regGC_EDC_CTRL = 0x1aed
# 	EDC_EN 0 0
# 	EDC_SW_RST 1 1
# 	EDC_CLK_EN_OVERRIDE 2 2
# 	EDC_FORCE_STALL 3 3
# 	EDC_TRIGGER_THROTTLE_LOWBIT 4 9
# 	EDC_ALLOW_WRITE_PWRDELTA 10 10
# 	EDC_THROTTLE_PATTERN_BIT_NUMS 11 14
# 	EDC_LEVEL_SEL 15 15
# 	EDC_ALGORITHM_MODE 16 16
# 	EDC_AVGDIV 17 20
# 	PSM_THROTTLE_SRC_SEL 21 23
# 	THROTTLE_SRC0_MASK 24 24
# 	THROTTLE_SRC1_MASK 25 25
# 	THROTTLE_SRC2_MASK 26 26
# 	THROTTLE_SRC3_MASK 27 27
# 	EDC_CREDIT_SHIFT_BIT_NUMS 28 31
regGC_EDC_OVERFLOW = 0x1b08
# 	EDC_ROLLING_POWER_DELTA_OVERFLOW 0 0
# 	EDC_THROTTLE_LEVEL_OVERFLOW_COUNTER 1 16
regGC_EDC_ROLLING_POWER_DELTA = 0x1b09
# 	EDC_ROLLING_POWER_DELTA 0 31
regGC_EDC_STATUS = 0x1b07
# 	EDC_THROTTLE_LEVEL 0 2
# 	GPIO_IN_0 3 3
# 	GPIO_IN_1 4 4
regGC_EDC_STRETCH_CTRL = 0x1aef
# 	EDC_STRETCH_EN 0 0
# 	EDC_STRETCH_DELAY 1 9
# 	EDC_UNSTRETCH_DELAY 10 18
regGC_EDC_STRETCH_THRESHOLD = 0x1af0
# 	EDC_STRETCH_THRESHOLD 0 31
regGC_EDC_THRESHOLD = 0x1aee
# 	EDC_THRESHOLD 0 31
regGC_IH_COOKIE_0_PTR = 0x5a07
# 	ADDR 0 19
regGC_THROTTLE_CTRL = 0x1af2
# 	THROTTLE_CTRL_SW_RST 0 0
# 	GC_EDC_STALL_EN 1 1
# 	PWRBRK_STALL_EN 2 2
# 	PWRBRK_POLARITY_CNTL 3 3
# 	PCC_STALL_EN 4 4
# 	PATTERN_MODE 5 5
# 	GC_EDC_ONLY_MODE 6 6
# 	GC_EDC_OVERRIDE 7 7
# 	PCC_OVERRIDE 8 8
# 	PWRBRK_OVERRIDE 9 9
# 	GC_EDC_PERF_COUNTER_EN 10 10
# 	PCC_PERF_COUNTER_EN 11 11
# 	PWRBRK_PERF_COUNTER_EN 12 12
# 	RELEASE_STEP_INTERVAL 13 22
# 	FIXED_PATTERN_PERF_COUNTER_EN 23 23
# 	FIXED_PATTERN_LOG_INDEX 24 28
# 	LUT_HW_UPDATE 29 29
# 	THROTTLE_CTRL_CLK_EN_OVERRIDE 30 30
# 	PCC_POLARITY_CNTL 31 31
regGC_THROTTLE_CTRL1 = 0x1af3
# 	PCC_FP_PROGRAM_STEP_EN 0 0
# 	PCC_PROGRAM_MIN_STEP 1 4
# 	PCC_PROGRAM_MAX_STEP 5 9
# 	PCC_PROGRAM_UPWARDS_STEP_SIZE 10 12
# 	PWRBRK_FP_PROGRAM_STEP_EN 13 13
# 	PWRBRK_PROGRAM_MIN_STEP 14 17
# 	PWRBRK_PROGRAM_MAX_STEP 18 22
# 	PWRBRK_PROGRAM_UPWARDS_STEP_SIZE 23 25
# 	FIXED_PATTERN_SELECT 26 27
# 	GC_EDC_STRETCH_PERF_COUNTER_EN 30 30
# 	GC_EDC_UNSTRETCH_PERF_COUNTER_EN 31 31
regGC_THROTTLE_STATUS = 0x1b0a
# 	FSM_STATE 0 3
# 	PATTERN_INDEX 4 8
regGC_USER_PRIM_CONFIG = 0x5b91
# 	INACTIVE_PA 4 19
regGC_USER_RB_BACKEND_DISABLE = 0x5b94
# 	BACKEND_DISABLE 4 31
regGC_USER_RB_REDUNDANCY = 0x5b93
# 	FAILED_RB0 8 11
# 	EN_REDUNDANCY0 12 12
# 	FAILED_RB1 16 19
# 	EN_REDUNDANCY1 20 20
regGC_USER_RMI_REDUNDANCY = 0x5b95
# 	REPAIR_EN_IN_0 1 1
# 	REPAIR_EN_IN_1 2 2
# 	REPAIR_RMI_OVERRIDE 3 3
# 	REPAIR_ID_SWAP 4 4
regGC_USER_SA_UNIT_DISABLE = 0x5b92
# 	SA_DISABLE 8 23
regGC_USER_SHADER_ARRAY_CONFIG = 0x5b90
# 	INACTIVE_WGPS 16 31
regGC_USER_SHADER_RATE_CONFIG = 0x5b97
# 	DPFP_RATE 1 2
regGDS_ATOM_BASE = 0x240c
# 	BASE 0 11
# 	UNUSED 12 31
regGDS_ATOM_CNTL = 0x240a
# 	AINC 0 5
# 	UNUSED1 6 7
# 	DMODE 8 9
# 	UNUSED2 10 31
regGDS_ATOM_COMPLETE = 0x240b
# 	COMPLETE 0 0
# 	UNUSED 1 31
regGDS_ATOM_DST = 0x2410
# 	DST 0 31
regGDS_ATOM_OFFSET0 = 0x240e
# 	OFFSET0 0 7
# 	UNUSED 8 31
regGDS_ATOM_OFFSET1 = 0x240f
# 	OFFSET1 0 7
# 	UNUSED 8 31
regGDS_ATOM_OP = 0x2411
# 	OP 0 7
# 	UNUSED 8 31
regGDS_ATOM_READ0 = 0x2416
# 	DATA 0 31
regGDS_ATOM_READ0_U = 0x2417
# 	DATA 0 31
regGDS_ATOM_READ1 = 0x2418
# 	DATA 0 31
regGDS_ATOM_READ1_U = 0x2419
# 	DATA 0 31
regGDS_ATOM_SIZE = 0x240d
# 	SIZE 0 12
# 	UNUSED 13 31
regGDS_ATOM_SRC0 = 0x2412
# 	DATA 0 31
regGDS_ATOM_SRC0_U = 0x2413
# 	DATA 0 31
regGDS_ATOM_SRC1 = 0x2414
# 	DATA 0 31
regGDS_ATOM_SRC1_U = 0x2415
# 	DATA 0 31
regGDS_CNTL_STATUS = 0x1361
# 	GDS_BUSY 0 0
# 	GRBM_WBUF_BUSY 1 1
# 	ORD_APP_BUSY 2 2
# 	DS_WR_CLAMP 3 3
# 	DS_RD_CLAMP 4 4
# 	GRBM_RBUF_BUSY 5 5
# 	DS_BUSY 6 6
# 	GWS_BUSY 7 7
# 	ORD_FIFO_BUSY 8 8
# 	CREDIT_BUSY0 9 9
# 	CREDIT_BUSY1 10 10
# 	CREDIT_BUSY2 11 11
# 	CREDIT_BUSY3 12 12
# 	CREDIT_BUSY4 13 13
# 	CREDIT_BUSY5 14 14
# 	CREDIT_BUSY6 15 15
# 	CREDIT_BUSY7 16 16
# 	UNUSED 17 31
regGDS_COMPUTE_MAX_WAVE_ID = 0x20e8
# 	MAX_WAVE_ID 0 11
# 	UNUSED 12 31
regGDS_CONFIG = 0x1360
# 	UNUSED 1 31
regGDS_CS_CTXSW_CNT0 = 0x20ee
# 	UPDN 0 15
# 	PTR 16 31
regGDS_CS_CTXSW_CNT1 = 0x20ef
# 	UPDN 0 15
# 	PTR 16 31
regGDS_CS_CTXSW_CNT2 = 0x20f0
# 	UPDN 0 15
# 	PTR 16 31
regGDS_CS_CTXSW_CNT3 = 0x20f1
# 	UPDN 0 15
# 	PTR 16 31
regGDS_CS_CTXSW_STATUS = 0x20ed
# 	R 0 0
# 	W 1 1
# 	UNUSED 2 31
regGDS_DSM_CNTL = 0x136a
# 	SEL_DSM_GDS_MEM_IRRITATOR_DATA_0 0 0
# 	SEL_DSM_GDS_MEM_IRRITATOR_DATA_1 1 1
# 	GDS_MEM_ENABLE_SINGLE_WRITE 2 2
# 	SEL_DSM_GDS_INPUT_QUEUE_IRRITATOR_DATA_0 3 3
# 	SEL_DSM_GDS_INPUT_QUEUE_IRRITATOR_DATA_1 4 4
# 	GDS_INPUT_QUEUE_ENABLE_SINGLE_WRITE 5 5
# 	SEL_DSM_GDS_PHY_CMD_RAM_IRRITATOR_DATA_0 6 6
# 	SEL_DSM_GDS_PHY_CMD_RAM_IRRITATOR_DATA_1 7 7
# 	GDS_PHY_CMD_RAM_ENABLE_SINGLE_WRITE 8 8
# 	SEL_DSM_GDS_PHY_DATA_RAM_IRRITATOR_DATA_0 9 9
# 	SEL_DSM_GDS_PHY_DATA_RAM_IRRITATOR_DATA_1 10 10
# 	GDS_PHY_DATA_RAM_ENABLE_SINGLE_WRITE 11 11
# 	SEL_DSM_GDS_PIPE_MEM_IRRITATOR_DATA_0 12 12
# 	SEL_DSM_GDS_PIPE_MEM_IRRITATOR_DATA_1 13 13
# 	GDS_PIPE_MEM_ENABLE_SINGLE_WRITE 14 14
# 	UNUSED 15 31
regGDS_DSM_CNTL2 = 0x136d
# 	GDS_MEM_ENABLE_ERROR_INJECT 0 1
# 	GDS_MEM_SELECT_INJECT_DELAY 2 2
# 	GDS_INPUT_QUEUE_ENABLE_ERROR_INJECT 3 4
# 	GDS_INPUT_QUEUE_SELECT_INJECT_DELAY 5 5
# 	GDS_PHY_CMD_RAM_ENABLE_ERROR_INJECT 6 7
# 	GDS_PHY_CMD_RAM_SELECT_INJECT_DELAY 8 8
# 	GDS_PHY_DATA_RAM_ENABLE_ERROR_INJECT 9 10
# 	GDS_PHY_DATA_RAM_SELECT_INJECT_DELAY 11 11
# 	GDS_PIPE_MEM_ENABLE_ERROR_INJECT 12 13
# 	GDS_PIPE_MEM_SELECT_INJECT_DELAY 14 14
# 	UNUSED 15 25
# 	GDS_INJECT_DELAY 26 31
regGDS_EDC_CNT = 0x1365
# 	GDS_MEM_DED 0 1
# 	GDS_INPUT_QUEUE_SED 2 3
# 	GDS_MEM_SEC 4 5
# 	UNUSED 6 31
regGDS_EDC_GRBM_CNT = 0x1366
# 	DED 0 1
# 	SEC 2 3
# 	UNUSED 4 31
regGDS_EDC_OA_DED = 0x1367
# 	ME0_GFXHP3D_PIX_DED 0 0
# 	ME0_GFXHP3D_VTX_DED 1 1
# 	ME0_CS_DED 2 2
# 	ME0_GFXHP3D_GS_DED 3 3
# 	ME1_PIPE0_DED 4 4
# 	ME1_PIPE1_DED 5 5
# 	ME1_PIPE2_DED 6 6
# 	ME1_PIPE3_DED 7 7
# 	ME2_PIPE0_DED 8 8
# 	ME2_PIPE1_DED 9 9
# 	ME2_PIPE2_DED 10 10
# 	ME2_PIPE3_DED 11 11
# 	ME0_PIPE1_CS_DED 12 12
# 	UNUSED1 13 31
regGDS_EDC_OA_PHY_CNT = 0x136b
# 	ME0_CS_PIPE_MEM_SEC 0 1
# 	ME0_CS_PIPE_MEM_DED 2 3
# 	PHY_CMD_RAM_MEM_SEC 4 5
# 	PHY_CMD_RAM_MEM_DED 6 7
# 	PHY_DATA_RAM_MEM_SED 8 9
# 	UNUSED1 10 31
regGDS_EDC_OA_PIPE_CNT = 0x136c
# 	ME1_PIPE0_PIPE_MEM_SEC 0 1
# 	ME1_PIPE0_PIPE_MEM_DED 2 3
# 	ME1_PIPE1_PIPE_MEM_SEC 4 5
# 	ME1_PIPE1_PIPE_MEM_DED 6 7
# 	ME1_PIPE2_PIPE_MEM_SEC 8 9
# 	ME1_PIPE2_PIPE_MEM_DED 10 11
# 	ME1_PIPE3_PIPE_MEM_SEC 12 13
# 	ME1_PIPE3_PIPE_MEM_DED 14 15
# 	UNUSED 16 31
regGDS_ENHANCE = 0x1362
# 	MISC 0 15
# 	AUTO_INC_INDEX 16 16
# 	CGPG_RESTORE 17 17
# 	UNUSED 18 31
regGDS_ENHANCE2 = 0x19b0
# 	DISABLE_MEMORY_VIOLATION_REPORT 0 0
# 	GDS_INTERFACES_FGCG_OVERRIDE 1 1
# 	DISABLE_PIPE_MEMORY_RD_OPT 2 2
# 	UNUSED 3 31
regGDS_GFX_CTXSW_STATUS = 0x20f2
# 	R 0 0
# 	W 1 1
# 	UNUSED 2 31
regGDS_GS_0 = 0x2426
# 	DATA 0 31
regGDS_GS_1 = 0x2427
# 	DATA 0 31
regGDS_GS_2 = 0x2428
# 	DATA 0 31
regGDS_GS_3 = 0x2429
# 	DATA 0 31
regGDS_GS_CTXSW_CNT0 = 0x2117
# 	UPDN 0 15
# 	PTR 16 31
regGDS_GS_CTXSW_CNT1 = 0x2118
# 	UPDN 0 15
# 	PTR 16 31
regGDS_GS_CTXSW_CNT2 = 0x2119
# 	UPDN 0 15
# 	PTR 16 31
regGDS_GS_CTXSW_CNT3 = 0x211a
# 	UPDN 0 15
# 	PTR 16 31
regGDS_GWS_RESET0 = 0x20e4
# 	RESOURCE0_RESET 0 0
# 	RESOURCE1_RESET 1 1
# 	RESOURCE2_RESET 2 2
# 	RESOURCE3_RESET 3 3
# 	RESOURCE4_RESET 4 4
# 	RESOURCE5_RESET 5 5
# 	RESOURCE6_RESET 6 6
# 	RESOURCE7_RESET 7 7
# 	RESOURCE8_RESET 8 8
# 	RESOURCE9_RESET 9 9
# 	RESOURCE10_RESET 10 10
# 	RESOURCE11_RESET 11 11
# 	RESOURCE12_RESET 12 12
# 	RESOURCE13_RESET 13 13
# 	RESOURCE14_RESET 14 14
# 	RESOURCE15_RESET 15 15
# 	RESOURCE16_RESET 16 16
# 	RESOURCE17_RESET 17 17
# 	RESOURCE18_RESET 18 18
# 	RESOURCE19_RESET 19 19
# 	RESOURCE20_RESET 20 20
# 	RESOURCE21_RESET 21 21
# 	RESOURCE22_RESET 22 22
# 	RESOURCE23_RESET 23 23
# 	RESOURCE24_RESET 24 24
# 	RESOURCE25_RESET 25 25
# 	RESOURCE26_RESET 26 26
# 	RESOURCE27_RESET 27 27
# 	RESOURCE28_RESET 28 28
# 	RESOURCE29_RESET 29 29
# 	RESOURCE30_RESET 30 30
# 	RESOURCE31_RESET 31 31
regGDS_GWS_RESET1 = 0x20e5
# 	RESOURCE32_RESET 0 0
# 	RESOURCE33_RESET 1 1
# 	RESOURCE34_RESET 2 2
# 	RESOURCE35_RESET 3 3
# 	RESOURCE36_RESET 4 4
# 	RESOURCE37_RESET 5 5
# 	RESOURCE38_RESET 6 6
# 	RESOURCE39_RESET 7 7
# 	RESOURCE40_RESET 8 8
# 	RESOURCE41_RESET 9 9
# 	RESOURCE42_RESET 10 10
# 	RESOURCE43_RESET 11 11
# 	RESOURCE44_RESET 12 12
# 	RESOURCE45_RESET 13 13
# 	RESOURCE46_RESET 14 14
# 	RESOURCE47_RESET 15 15
# 	RESOURCE48_RESET 16 16
# 	RESOURCE49_RESET 17 17
# 	RESOURCE50_RESET 18 18
# 	RESOURCE51_RESET 19 19
# 	RESOURCE52_RESET 20 20
# 	RESOURCE53_RESET 21 21
# 	RESOURCE54_RESET 22 22
# 	RESOURCE55_RESET 23 23
# 	RESOURCE56_RESET 24 24
# 	RESOURCE57_RESET 25 25
# 	RESOURCE58_RESET 26 26
# 	RESOURCE59_RESET 27 27
# 	RESOURCE60_RESET 28 28
# 	RESOURCE61_RESET 29 29
# 	RESOURCE62_RESET 30 30
# 	RESOURCE63_RESET 31 31
regGDS_GWS_RESOURCE = 0x241b
# 	FLAG 0 0
# 	COUNTER 1 12
# 	TYPE 13 13
# 	DED 14 14
# 	RELEASE_ALL 15 15
# 	HEAD_QUEUE 16 28
# 	HEAD_VALID 29 29
# 	HEAD_FLAG 30 30
# 	HALTED 31 31
regGDS_GWS_RESOURCE_CNT = 0x241c
# 	RESOURCE_CNT 0 15
# 	UNUSED 16 31
regGDS_GWS_RESOURCE_CNTL = 0x241a
# 	INDEX 0 5
# 	UNUSED 6 31
regGDS_GWS_RESOURCE_RESET = 0x20e6
# 	RESET 0 0
# 	RESOURCE_ID 8 15
# 	UNUSED 16 31
regGDS_GWS_VMID0 = 0x20c0
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID1 = 0x20c1
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID10 = 0x20ca
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID11 = 0x20cb
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID12 = 0x20cc
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID13 = 0x20cd
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID14 = 0x20ce
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID15 = 0x20cf
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID2 = 0x20c2
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID3 = 0x20c3
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID4 = 0x20c4
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID5 = 0x20c5
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID6 = 0x20c6
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID7 = 0x20c7
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID8 = 0x20c8
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_GWS_VMID9 = 0x20c9
# 	BASE 0 5
# 	UNUSED1 6 15
# 	SIZE 16 22
# 	UNUSED2 23 31
regGDS_MEMORY_CLEAN = 0x211f
# 	START 0 0
# 	FINISH 1 1
# 	UNUSED 2 31
regGDS_OA_ADDRESS = 0x241f
# 	DS_ADDRESS 0 15
# 	CRAWLER_TYPE 16 19
# 	CRAWLER 20 23
# 	UNUSED 24 29
# 	NO_ALLOC 30 30
# 	ENABLE 31 31
regGDS_OA_CGPG_RESTORE = 0x19b1
# 	VMID 0 7
# 	MEID 8 11
# 	PIPEID 12 15
# 	QUEUEID 16 19
# 	UNUSED 20 31
regGDS_OA_CNTL = 0x241d
# 	INDEX 0 3
# 	UNUSED 4 31
regGDS_OA_COUNTER = 0x241e
# 	SPACE_AVAILABLE 0 31
regGDS_OA_INCDEC = 0x2420
# 	VALUE 0 30
# 	INCDEC 31 31
regGDS_OA_RESET = 0x20ea
# 	RESET 0 0
# 	PIPE_ID 8 15
# 	UNUSED 16 31
regGDS_OA_RESET_MASK = 0x20e9
# 	ME0_GFXHP3D_PIX_RESET 0 0
# 	ME0_GFXHP3D_VTX_RESET 1 1
# 	ME0_CS_RESET 2 2
# 	ME0_GFXHP3D_GS_RESET 3 3
# 	ME1_PIPE0_RESET 4 4
# 	ME1_PIPE1_RESET 5 5
# 	ME1_PIPE2_RESET 6 6
# 	ME1_PIPE3_RESET 7 7
# 	ME2_PIPE0_RESET 8 8
# 	ME2_PIPE1_RESET 9 9
# 	ME2_PIPE2_RESET 10 10
# 	ME2_PIPE3_RESET 11 11
# 	ME0_PIPE1_CS_RESET 12 12
# 	UNUSED1 13 31
regGDS_OA_RING_SIZE = 0x2421
# 	RING_SIZE 0 31
regGDS_OA_VMID0 = 0x20d0
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID1 = 0x20d1
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID10 = 0x20da
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID11 = 0x20db
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID12 = 0x20dc
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID13 = 0x20dd
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID14 = 0x20de
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID15 = 0x20df
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID2 = 0x20d2
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID3 = 0x20d3
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID4 = 0x20d4
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID5 = 0x20d5
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID6 = 0x20d6
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID7 = 0x20d7
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID8 = 0x20d8
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_OA_VMID9 = 0x20d9
# 	MASK 0 15
# 	UNUSED 16 31
regGDS_PERFCOUNTER0_HI = 0x3281
# 	PERFCOUNTER_HI 0 31
regGDS_PERFCOUNTER0_LO = 0x3280
# 	PERFCOUNTER_LO 0 31
regGDS_PERFCOUNTER0_SELECT = 0x3a80
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGDS_PERFCOUNTER0_SELECT1 = 0x3a84
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGDS_PERFCOUNTER1_HI = 0x3283
# 	PERFCOUNTER_HI 0 31
regGDS_PERFCOUNTER1_LO = 0x3282
# 	PERFCOUNTER_LO 0 31
regGDS_PERFCOUNTER1_SELECT = 0x3a81
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGDS_PERFCOUNTER1_SELECT1 = 0x3a85
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGDS_PERFCOUNTER2_HI = 0x3285
# 	PERFCOUNTER_HI 0 31
regGDS_PERFCOUNTER2_LO = 0x3284
# 	PERFCOUNTER_LO 0 31
regGDS_PERFCOUNTER2_SELECT = 0x3a82
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGDS_PERFCOUNTER2_SELECT1 = 0x3a86
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGDS_PERFCOUNTER3_HI = 0x3287
# 	PERFCOUNTER_HI 0 31
regGDS_PERFCOUNTER3_LO = 0x3286
# 	PERFCOUNTER_LO 0 31
regGDS_PERFCOUNTER3_SELECT = 0x3a83
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGDS_PERFCOUNTER3_SELECT1 = 0x3a87
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGDS_PROTECTION_FAULT = 0x1363
# 	WRITE_DIS 0 0
# 	FAULT_DETECTED 1 1
# 	GRBM 2 2
# 	SE_ID 3 5
# 	SA_ID 6 6
# 	WGP_ID 7 10
# 	SIMD_ID 11 12
# 	WAVE_ID 13 17
# 	ADDRESS 18 31
regGDS_PS_CTXSW_CNT0 = 0x20f7
# 	UPDN 0 15
# 	PTR 16 31
regGDS_PS_CTXSW_CNT1 = 0x20f8
# 	UPDN 0 15
# 	PTR 16 31
regGDS_PS_CTXSW_CNT2 = 0x20f9
# 	UPDN 0 15
# 	PTR 16 31
regGDS_PS_CTXSW_CNT3 = 0x20fa
# 	UPDN 0 15
# 	PTR 16 31
regGDS_PS_CTXSW_IDX = 0x20fb
# 	PACKER_ID 0 5
# 	UNUSED 6 31
regGDS_RD_ADDR = 0x2400
# 	READ_ADDR 0 31
regGDS_RD_BURST_ADDR = 0x2402
# 	BURST_ADDR 0 31
regGDS_RD_BURST_COUNT = 0x2403
# 	BURST_COUNT 0 31
regGDS_RD_BURST_DATA = 0x2404
# 	BURST_DATA 0 31
regGDS_RD_DATA = 0x2401
# 	READ_DATA 0 31
regGDS_STRMOUT_DWORDS_WRITTEN_0 = 0x2422
# 	DATA 0 31
regGDS_STRMOUT_DWORDS_WRITTEN_1 = 0x2423
# 	DATA 0 31
regGDS_STRMOUT_DWORDS_WRITTEN_2 = 0x2424
# 	DATA 0 31
regGDS_STRMOUT_DWORDS_WRITTEN_3 = 0x2425
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_0_HI = 0x242b
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_0_LO = 0x242a
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_1_HI = 0x242f
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_1_LO = 0x242e
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_2_HI = 0x2433
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_2_LO = 0x2432
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_3_HI = 0x2437
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_NEEDED_3_LO = 0x2436
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_0_HI = 0x242d
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_0_LO = 0x242c
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_1_HI = 0x2431
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_1_LO = 0x2430
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_2_HI = 0x2435
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_2_LO = 0x2434
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_3_HI = 0x2439
# 	DATA 0 31
regGDS_STRMOUT_PRIMS_WRITTEN_3_LO = 0x2438
# 	DATA 0 31
regGDS_VMID0_BASE = 0x20a0
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID0_SIZE = 0x20a1
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID10_BASE = 0x20b4
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID10_SIZE = 0x20b5
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID11_BASE = 0x20b6
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID11_SIZE = 0x20b7
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID12_BASE = 0x20b8
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID12_SIZE = 0x20b9
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID13_BASE = 0x20ba
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID13_SIZE = 0x20bb
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID14_BASE = 0x20bc
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID14_SIZE = 0x20bd
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID15_BASE = 0x20be
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID15_SIZE = 0x20bf
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID1_BASE = 0x20a2
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID1_SIZE = 0x20a3
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID2_BASE = 0x20a4
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID2_SIZE = 0x20a5
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID3_BASE = 0x20a6
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID3_SIZE = 0x20a7
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID4_BASE = 0x20a8
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID4_SIZE = 0x20a9
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID5_BASE = 0x20aa
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID5_SIZE = 0x20ab
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID6_BASE = 0x20ac
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID6_SIZE = 0x20ad
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID7_BASE = 0x20ae
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID7_SIZE = 0x20af
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID8_BASE = 0x20b0
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID8_SIZE = 0x20b1
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VMID9_BASE = 0x20b2
# 	BASE 0 15
# 	UNUSED 16 31
regGDS_VMID9_SIZE = 0x20b3
# 	SIZE 0 16
# 	UNUSED 17 31
regGDS_VM_PROTECTION_FAULT = 0x1364
# 	WRITE_DIS 0 0
# 	FAULT_DETECTED 1 1
# 	GWS 2 2
# 	OA 3 3
# 	GRBM 4 4
# 	TMZ 5 5
# 	UNUSED1 6 7
# 	VMID 8 11
# 	UNUSED2 12 15
# 	ADDRESS 16 31
regGDS_WRITE_COMPLETE = 0x2409
# 	WRITE_COMPLETE 0 31
regGDS_WR_ADDR = 0x2405
# 	WRITE_ADDR 0 31
regGDS_WR_BURST_ADDR = 0x2407
# 	WRITE_ADDR 0 31
regGDS_WR_BURST_DATA = 0x2408
# 	WRITE_DATA 0 31
regGDS_WR_DATA = 0x2406
# 	WRITE_DATA 0 31
regGE1_PERFCOUNTER0_HI = 0x30a5
# 	PERFCOUNTER_HI 0 31
regGE1_PERFCOUNTER0_LO = 0x30a4
# 	PERFCOUNTER_LO 0 31
regGE1_PERFCOUNTER0_SELECT = 0x38a4
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE1_PERFCOUNTER0_SELECT1 = 0x38a5
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE1_PERFCOUNTER1_HI = 0x30a7
# 	PERFCOUNTER_HI 0 31
regGE1_PERFCOUNTER1_LO = 0x30a6
# 	PERFCOUNTER_LO 0 31
regGE1_PERFCOUNTER1_SELECT = 0x38a6
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE1_PERFCOUNTER1_SELECT1 = 0x38a7
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE1_PERFCOUNTER2_HI = 0x30a9
# 	PERFCOUNTER_HI 0 31
regGE1_PERFCOUNTER2_LO = 0x30a8
# 	PERFCOUNTER_LO 0 31
regGE1_PERFCOUNTER2_SELECT = 0x38a8
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE1_PERFCOUNTER2_SELECT1 = 0x38a9
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE1_PERFCOUNTER3_HI = 0x30ab
# 	PERFCOUNTER_HI 0 31
regGE1_PERFCOUNTER3_LO = 0x30aa
# 	PERFCOUNTER_LO 0 31
regGE1_PERFCOUNTER3_SELECT = 0x38aa
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE1_PERFCOUNTER3_SELECT1 = 0x38ab
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_DIST_PERFCOUNTER0_HI = 0x30ad
# 	PERFCOUNTER_HI 0 31
regGE2_DIST_PERFCOUNTER0_LO = 0x30ac
# 	PERFCOUNTER_LO 0 31
regGE2_DIST_PERFCOUNTER0_SELECT = 0x38ac
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_DIST_PERFCOUNTER0_SELECT1 = 0x38ad
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_DIST_PERFCOUNTER1_HI = 0x30af
# 	PERFCOUNTER_HI 0 31
regGE2_DIST_PERFCOUNTER1_LO = 0x30ae
# 	PERFCOUNTER_LO 0 31
regGE2_DIST_PERFCOUNTER1_SELECT = 0x38ae
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_DIST_PERFCOUNTER1_SELECT1 = 0x38af
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_DIST_PERFCOUNTER2_HI = 0x30b1
# 	PERFCOUNTER_HI 0 31
regGE2_DIST_PERFCOUNTER2_LO = 0x30b0
# 	PERFCOUNTER_LO 0 31
regGE2_DIST_PERFCOUNTER2_SELECT = 0x38b0
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_DIST_PERFCOUNTER2_SELECT1 = 0x38b1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_DIST_PERFCOUNTER3_HI = 0x30b3
# 	PERFCOUNTER_HI 0 31
regGE2_DIST_PERFCOUNTER3_LO = 0x30b2
# 	PERFCOUNTER_LO 0 31
regGE2_DIST_PERFCOUNTER3_SELECT = 0x38b2
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_DIST_PERFCOUNTER3_SELECT1 = 0x38b3
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_SE_CNTL_STATUS = 0x1011
# 	TE_BUSY 0 0
# 	NGG_BUSY 1 1
# 	HS_BUSY 2 2
regGE2_SE_PERFCOUNTER0_HI = 0x30b5
# 	PERFCOUNTER_HI 0 31
regGE2_SE_PERFCOUNTER0_LO = 0x30b4
# 	PERFCOUNTER_LO 0 31
regGE2_SE_PERFCOUNTER0_SELECT = 0x38b4
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_SE_PERFCOUNTER0_SELECT1 = 0x38b5
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_SE_PERFCOUNTER1_HI = 0x30b7
# 	PERFCOUNTER_HI 0 31
regGE2_SE_PERFCOUNTER1_LO = 0x30b6
# 	PERFCOUNTER_LO 0 31
regGE2_SE_PERFCOUNTER1_SELECT = 0x38b6
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_SE_PERFCOUNTER1_SELECT1 = 0x38b7
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_SE_PERFCOUNTER2_HI = 0x30b9
# 	PERFCOUNTER_HI 0 31
regGE2_SE_PERFCOUNTER2_LO = 0x30b8
# 	PERFCOUNTER_LO 0 31
regGE2_SE_PERFCOUNTER2_SELECT = 0x38b8
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_SE_PERFCOUNTER2_SELECT1 = 0x38b9
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE2_SE_PERFCOUNTER3_HI = 0x30bb
# 	PERFCOUNTER_HI 0 31
regGE2_SE_PERFCOUNTER3_LO = 0x30ba
# 	PERFCOUNTER_LO 0 31
regGE2_SE_PERFCOUNTER3_SELECT = 0x38ba
# 	PERF_SEL0 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE0 28 31
regGE2_SE_PERFCOUNTER3_SELECT1 = 0x38bb
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGE_CNTL = 0x225b
# 	PRIMS_PER_SUBGRP 0 8
# 	VERTS_PER_SUBGRP 9 17
# 	BREAK_SUBGRP_AT_EOI 18 18
# 	PACKET_TO_ONE_PA 19 19
# 	BREAK_PRIMGRP_AT_EOI 20 20
# 	PRIM_GRP_SIZE 21 29
# 	GCR_DISABLE 30 30
# 	DIS_PG_SIZE_ADJUST_FOR_STRIP 31 31
regGE_GS_FAST_LAUNCH_WG_DIM = 0x2264
# 	GS_FL_DIM_X 0 15
# 	GS_FL_DIM_Y 16 31
regGE_GS_FAST_LAUNCH_WG_DIM_1 = 0x2265
# 	GS_FL_DIM_Z 0 15
regGE_INDX_OFFSET = 0x224a
# 	INDX_OFFSET 0 31
regGE_MAX_OUTPUT_PER_SUBGROUP = 0x1ff
# 	MAX_VERTS_PER_SUBGROUP 0 9
regGE_MAX_VTX_INDX = 0x2259
# 	MAX_INDX 0 31
regGE_MIN_VTX_INDX = 0x2249
# 	MIN_INDX 0 31
regGE_MULTI_PRIM_IB_RESET_EN = 0x224b
# 	RESET_EN 0 0
# 	MATCH_ALL_BITS 1 1
# 	DISABLE_FOR_AUTO_INDEX 2 2
regGE_NGG_SUBGRP_CNTL = 0x2d3
# 	PRIM_AMP_FACTOR 0 8
# 	THDS_PER_SUBGRP 9 17
regGE_PA_IF_SAFE_REG = 0x1019
# 	GE_PA_CSB 0 9
# 	GE_PA_PAYLOAD 10 19
regGE_PC_ALLOC = 0x2260
# 	OVERSUB_EN 0 0
# 	NUM_PC_LINES 1 10
regGE_PRIV_CONTROL = 0x1004
# 	RESERVED 0 0
# 	CLAMP_PRIMGRP_SIZE 1 9
# 	RESET_ON_PIPELINE_CHANGE 10 10
# 	FGCG_OVERRIDE 15 15
# 	CLAMP_HS_OFFCHIP_PER_SE_OVERRIDE 16 16
# 	DISABLE_ACCUM_AGM 17 17
regGE_RATE_CNTL_1 = 0xff4
# 	ADD_X_CLKS_LS_VERT 0 3
# 	AFTER_Y_TRANS_LS_VERT 4 7
# 	ADD_X_CLKS_HS_VERT 8 11
# 	AFTER_Y_TRANS_HS_VERT 12 15
# 	ADD_X_CLKS_ES_VERT 16 19
# 	AFTER_Y_TRANS_ES_VERT 20 23
# 	ADD_X_CLKS_GS_PRIM 24 27
# 	AFTER_Y_TRANS_GS_PRIM 28 31
regGE_RATE_CNTL_2 = 0xff5
# 	ADD_X_CLKS_VS_VERT 0 3
# 	AFTER_Y_TRANS_VS_VERT 4 7
# 	ADD_X_CLKS_PA_PRIM 8 11
# 	AFTER_Y_TRANS_PA_PRIM 12 15
# 	ADD_X_CLKS_MERGED_HS_GS 16 19
# 	ADD_X_CLKS_MERGED_LS_ES 20 23
# 	MERGED_HS_GS_MODE 24 24
# 	MERGED_LS_ES_MODE 25 25
# 	ENABLE_RATE_CNTL 26 26
# 	SWAP_PRIORITY 27 27
regGE_SPI_IF_SAFE_REG = 0x1018
# 	GE_SPI_LS_ES_DATA 0 5
# 	GE_SPI_HS_GS_DATA 6 11
# 	GE_SPI_GRP 12 17
regGE_STATUS = 0x1005
# 	PERFCOUNTER_STATUS 0 0
# 	THREAD_TRACE_STATUS 1 1
regGE_STEREO_CNTL = 0x225f
# 	RT_SLICE 0 2
# 	VIEWPORT 3 6
# 	EN_STEREO 8 8
regGE_USER_VGPR1 = 0x225c
# 	DATA 0 31
regGE_USER_VGPR2 = 0x225d
# 	DATA 0 31
regGE_USER_VGPR3 = 0x225e
# 	DATA 0 31
regGE_USER_VGPR_EN = 0x2262
# 	EN_USER_VGPR1 0 0
# 	EN_USER_VGPR2 1 1
# 	EN_USER_VGPR3 2 2
regGFX_COPY_STATE = 0x1f4
# 	SRC_STATE_ID 0 2
regGFX_ICG_GL2C_CTRL = 0x50fc
# 	REG_OVERRIDE 0 0
# 	PERFMON_OVERRIDE 1 1
# 	IB_OVERRIDE 2 2
# 	TAG_OVERRIDE 3 3
# 	CM_CORE_OVERRIDE 4 4
# 	CORE_OVERRIDE 5 5
# 	CACHE_RAM_OVERRIDE 6 6
# 	GCR_OVERRIDE 7 7
# 	EXECUTE_OVERRIDE 8 8
# 	RETURN_BUFFER_OVERRIDE 9 9
# 	LATENCY_FIFO_OVERRIDE 10 10
# 	OUTPUT_FIFOS_OVERRIDE 11 11
# 	MC_WRITE_OVERRIDE 12 12
# 	EXECUTE_DECOMP_OVERRIDE 13 13
# 	EXECUTE_WRITE_OVERRIDE 14 14
# 	TAG_FLOPSET_GROUP0_OVERRIDE 15 15
# 	TAG_FLOPSET_GROUP1_OVERRIDE 16 16
# 	TAG_FLOPSET_GROUP2_OVERRIDE 17 17
# 	TAG_FLOPSET_GROUP3_OVERRIDE 18 18
# 	CM_RVF_OVERRIDE 20 20
# 	CM_SDR_OVERRIDE 21 21
# 	CM_RPF_OVERRIDE 22 22
# 	CM_STS_OVERRIDE 23 23
# 	CM_READ_OVERRIDE 24 24
# 	CM_MERGE_OVERRIDE 25 25
# 	CM_COMP_OVERRIDE 26 26
# 	CM_DCC_OVERRIDE 27 27
# 	CM_WRITE_OVERRIDE 28 28
# 	CM_NOOP_OVERRIDE 29 29
# 	MDC_TAG_OVERRIDE 30 30
# 	MDC_DATA_OVERRIDE 31 31
regGFX_ICG_GL2C_CTRL1 = 0x50fd
# 	OUTPUT_FIFOS_INTERNAL_CLIENT0_OVERRIDE 0 0
# 	OUTPUT_FIFOS_INTERNAL_CLIENT1_OVERRIDE 1 1
# 	OUTPUT_FIFOS_INTERNAL_CLIENT2_OVERRIDE 2 2
# 	OUTPUT_FIFOS_INTERNAL_CLIENT3_OVERRIDE 3 3
# 	OUTPUT_FIFOS_INTERNAL_CLIENT4_OVERRIDE 4 4
# 	OUTPUT_FIFOS_INTERNAL_CLIENT5_OVERRIDE 5 5
# 	OUTPUT_FIFOS_INTERNAL_CLIENT6_OVERRIDE 6 6
# 	OUTPUT_FIFOS_INTERNAL_CLIENT7_OVERRIDE 7 7
# 	OUTPUT_FIFOS_INTERNAL_CLIENT8_OVERRIDE 8 8
# 	OUTPUT_FIFOS_INTERNAL_CLIENT9_OVERRIDE 9 9
# 	OUTPUT_FIFOS_INTERNAL_CLIENT10_OVERRIDE 10 10
# 	OUTPUT_FIFOS_INTERNAL_CLIENT11_OVERRIDE 11 11
# 	OUTPUT_FIFOS_INTERNAL_CLIENT12_OVERRIDE 12 12
# 	OUTPUT_FIFOS_INTERNAL_CLIENT13_OVERRIDE 13 13
# 	OUTPUT_FIFOS_INTERNAL_CLIENT14_OVERRIDE 14 14
# 	OUTPUT_FIFOS_INTERNAL_CLIENT15_OVERRIDE 15 15
# 	OUTPUT_FIFOS_INTERNAL_CLIENT16_OVERRIDE 16 16
# 	OUTPUT_FIFOS_INTERNAL_CLIENT17_OVERRIDE 17 17
# 	TAG_PROBE_OVERRIDE 24 24
# 	DCC_UPPER_OVERRIDE 25 25
# 	DCC_LOWER_OVERRIDE 26 26
# 	ZD_UPPER_OVERRIDE 27 27
# 	ZD_LOWER_OVERRIDE 28 28
regGFX_IMU_AEB_OVERRIDE = 0x40bd
# 	AEB_OVERRIDE_CTRL 0 0
# 	AEB_RESET_VALUE 1 1
# 	AEB_VALID_VALUE 2 2
regGFX_IMU_C2PMSG_0 = 0x4000
# 	DATA 0 31
regGFX_IMU_C2PMSG_1 = 0x4001
# 	DATA 0 31
regGFX_IMU_C2PMSG_10 = 0x400a
# 	DATA 0 31
regGFX_IMU_C2PMSG_11 = 0x400b
# 	DATA 0 31
regGFX_IMU_C2PMSG_12 = 0x400c
# 	DATA 0 31
regGFX_IMU_C2PMSG_13 = 0x400d
# 	DATA 0 31
regGFX_IMU_C2PMSG_14 = 0x400e
# 	DATA 0 31
regGFX_IMU_C2PMSG_15 = 0x400f
# 	DATA 0 31
regGFX_IMU_C2PMSG_16 = 0x4010
# 	DATA 0 31
regGFX_IMU_C2PMSG_17 = 0x4011
# 	DATA 0 31
regGFX_IMU_C2PMSG_18 = 0x4012
# 	DATA 0 31
regGFX_IMU_C2PMSG_19 = 0x4013
# 	DATA 0 31
regGFX_IMU_C2PMSG_2 = 0x4002
# 	DATA 0 31
regGFX_IMU_C2PMSG_20 = 0x4014
# 	DATA 0 31
regGFX_IMU_C2PMSG_21 = 0x4015
# 	DATA 0 31
regGFX_IMU_C2PMSG_22 = 0x4016
# 	DATA 0 31
regGFX_IMU_C2PMSG_23 = 0x4017
# 	DATA 0 31
regGFX_IMU_C2PMSG_24 = 0x4018
# 	DATA 0 31
regGFX_IMU_C2PMSG_25 = 0x4019
# 	DATA 0 31
regGFX_IMU_C2PMSG_26 = 0x401a
# 	DATA 0 31
regGFX_IMU_C2PMSG_27 = 0x401b
# 	DATA 0 31
regGFX_IMU_C2PMSG_28 = 0x401c
# 	DATA 0 31
regGFX_IMU_C2PMSG_29 = 0x401d
# 	DATA 0 31
regGFX_IMU_C2PMSG_3 = 0x4003
# 	DATA 0 31
regGFX_IMU_C2PMSG_30 = 0x401e
# 	DATA 0 31
regGFX_IMU_C2PMSG_31 = 0x401f
# 	DATA 0 31
regGFX_IMU_C2PMSG_32 = 0x4020
# 	DATA 0 31
regGFX_IMU_C2PMSG_33 = 0x4021
# 	DATA 0 31
regGFX_IMU_C2PMSG_34 = 0x4022
# 	DATA 0 31
regGFX_IMU_C2PMSG_35 = 0x4023
# 	DATA 0 31
regGFX_IMU_C2PMSG_36 = 0x4024
# 	DATA 0 31
regGFX_IMU_C2PMSG_37 = 0x4025
# 	DATA 0 31
regGFX_IMU_C2PMSG_38 = 0x4026
# 	DATA 0 31
regGFX_IMU_C2PMSG_39 = 0x4027
# 	DATA 0 31
regGFX_IMU_C2PMSG_4 = 0x4004
# 	DATA 0 31
regGFX_IMU_C2PMSG_40 = 0x4028
# 	DATA 0 31
regGFX_IMU_C2PMSG_41 = 0x4029
# 	DATA 0 31
regGFX_IMU_C2PMSG_42 = 0x402a
# 	DATA 0 31
regGFX_IMU_C2PMSG_43 = 0x402b
# 	DATA 0 31
regGFX_IMU_C2PMSG_44 = 0x402c
# 	DATA 0 31
regGFX_IMU_C2PMSG_45 = 0x402d
# 	DATA 0 31
regGFX_IMU_C2PMSG_46 = 0x402e
# 	DATA 0 31
regGFX_IMU_C2PMSG_47 = 0x402f
# 	DATA 0 31
regGFX_IMU_C2PMSG_5 = 0x4005
# 	DATA 0 31
regGFX_IMU_C2PMSG_6 = 0x4006
# 	DATA 0 31
regGFX_IMU_C2PMSG_7 = 0x4007
# 	DATA 0 31
regGFX_IMU_C2PMSG_8 = 0x4008
# 	DATA 0 31
regGFX_IMU_C2PMSG_9 = 0x4009
# 	DATA 0 31
regGFX_IMU_C2PMSG_ACCESS_CTRL0 = 0x4040
# 	ACC0 0 2
# 	ACC1 3 5
# 	ACC2 6 8
# 	ACC3 9 11
# 	ACC4 12 14
# 	ACC5 15 17
# 	ACC6 18 20
# 	ACC7 21 23
regGFX_IMU_C2PMSG_ACCESS_CTRL1 = 0x4041
# 	ACC8_15 0 2
# 	ACC16_23 3 5
# 	ACC24_31 6 8
# 	ACC32_39 9 11
# 	ACC40_47 12 14
regGFX_IMU_CLK_CTRL = 0x409d
# 	CG_OVR 0 0
# 	CG_OVR_CORE 1 1
# 	CLKDIV 4 4
# 	GFXBYPASSCLK_CHGTOG 8 8
# 	GFXBYPASSCLK_DONETOG 9 9
# 	GFXBYPASSCLK_DIV 16 22
# 	COOLDOWN_PERIOD 28 31
regGFX_IMU_CORE_CTRL = 0x40b6
# 	CRESET 0 0
# 	CSTALL 1 1
# 	DRESET 3 3
# 	HALT_ON_RESET 4 4
# 	BREAK_IN 8 8
# 	BREAK_OUT_ACK 9 9
regGFX_IMU_CORE_INT_STATUS = 0x407f
# 	INTERRUPT_24 24 24
# 	INTERRUPT_25 25 25
# 	INTERRUPT_29 29 29
regGFX_IMU_CORE_STATUS = 0x40b7
# 	CBUSY 0 0
# 	PWAIT_MODE 1 1
# 	CINTLEVEL 4 7
# 	BREAK_IN_ACK 8 8
# 	BREAK_OUT 9 9
# 	P_FATAL_ERROR 11 11
# 	FAULT_SEVERITY_LEVEL 24 27
# 	FAULT_TYPE 28 31
regGFX_IMU_DOORBELL_CONTROL = 0x409e
# 	OVR_EN 0 0
# 	FENCE_EN_OVR 1 1
# 	CP_DB_RESP_PEND_COUNT 24 30
# 	FENCE_EN_STATUS 31 31
regGFX_IMU_DPM_ACC = 0x40a9
# 	COUNT 0 23
regGFX_IMU_DPM_CONTROL = 0x40a8
# 	ACC_RESET 0 0
# 	ACC_START 1 1
# 	BUSY_MASK 2 17
regGFX_IMU_DPM_REF_COUNTER = 0x40aa
# 	COUNT 0 23
regGFX_IMU_D_RAM_ADDR = 0x40fc
# 	ADDR 2 15
regGFX_IMU_D_RAM_DATA = 0x40fd
# 	DATA 0 31
regGFX_IMU_FENCE_CTRL = 0x40b0
# 	ENABLED 0 0
# 	ARM_LOG 1 1
# 	FLUSH_ARBITER_CREDITS 3 3
# 	GFX_REG_FENCE_OVR_EN 8 8
# 	GFX_REG_FENCE_OVR 9 9
regGFX_IMU_FENCE_LOG_ADDR = 0x40b2
# 	ADDR 2 19
regGFX_IMU_FENCE_LOG_INIT = 0x40b1
# 	UNIT_ID 0 6
# 	INITIATOR_ID 7 16
regGFX_IMU_FUSE_CTRL = 0x40e0
# 	DIV_OVR 0 4
# 	DIV_OVR_EN 5 5
# 	FORCE_DONE 6 6
regGFX_IMU_FW_GTS_HI = 0x4079
# 	TSTAMP_HI 0 23
regGFX_IMU_FW_GTS_LO = 0x4078
# 	TSTAMP_LO 0 31
regGFX_IMU_GAP_PWROK = 0x40ba
# 	GAP_PWROK 0 0
regGFX_IMU_GFXCLK_BYPASS_CTRL = 0x409c
# 	BYPASS_SEL 0 0
regGFX_IMU_GFX_IH_GASKET_CTRL = 0x40ff
# 	SRSTB 0 0
# 	BUFFER_LEVEL 16 19
# 	BUFFER_OVERFLOW 20 20
regGFX_IMU_GFX_ISO_CTRL = 0x40bf
# 	GFX2IMU_ISOn 0 0
# 	SOC_EA_SDF_VDCI_ISOn_EN 1 1
# 	SOC_UTCL2_ATHUB_VDCI_ISOn_EN 2 2
# 	GFX2SOC_ISOn 3 3
# 	GFX2SOC_CLK_ISOn 4 4
regGFX_IMU_GFX_RESET_CTRL = 0x40bc
# 	HARD_RESETB 0 0
# 	EA_RESETB 1 1
# 	UTCL2_RESETB 2 2
# 	SDMA_RESETB 3 3
# 	GRBM_RESETB 4 4
regGFX_IMU_GTS_OFFSET_HI = 0x407b
# 	GTS_OFFSET_HI 0 23
regGFX_IMU_GTS_OFFSET_LO = 0x407a
# 	GTS_OFFSET_LO 0 31
regGFX_IMU_IH_CTRL_1 = 0x4090
# 	CONTEXT_ID 0 31
regGFX_IMU_IH_CTRL_2 = 0x4091
# 	CONTEXT_ID 0 7
# 	RING_ID 8 15
# 	VM_ID 16 19
# 	SRSTB 31 31
regGFX_IMU_IH_CTRL_3 = 0x4092
# 	SOURCE_ID 0 7
# 	VF_ID 8 12
# 	VF 13 13
regGFX_IMU_IH_STATUS = 0x4093
# 	IH_BUSY 0 0
regGFX_IMU_I_RAM_ADDR = 0x5f90
# 	ADDR 2 15
regGFX_IMU_I_RAM_DATA = 0x5f91
# 	DATA 0 31
regGFX_IMU_MP1_MUTEX = 0x4043
# 	MUTEX 0 1
regGFX_IMU_MSG_FLAGS = 0x403f
# 	STATUS 0 31
regGFX_IMU_PIC_INTR = 0x408c
# 	INTR_n 0 0
regGFX_IMU_PIC_INTR_ID = 0x408d
# 	INTR_n 0 7
regGFX_IMU_PIC_INT_EDGE = 0x4082
# 	EDGE_0 0 0
# 	EDGE_1 1 1
# 	EDGE_2 2 2
# 	EDGE_3 3 3
# 	EDGE_4 4 4
# 	EDGE_5 5 5
# 	EDGE_6 6 6
# 	EDGE_7 7 7
# 	EDGE_8 8 8
# 	EDGE_9 9 9
# 	EDGE_10 10 10
# 	EDGE_11 11 11
# 	EDGE_12 12 12
# 	EDGE_13 13 13
# 	EDGE_14 14 14
# 	EDGE_15 15 15
# 	EDGE_16 16 16
# 	EDGE_17 17 17
# 	EDGE_18 18 18
# 	EDGE_19 19 19
# 	EDGE_20 20 20
# 	EDGE_21 21 21
# 	EDGE_22 22 22
# 	EDGE_23 23 23
# 	EDGE_24 24 24
# 	EDGE_25 25 25
# 	EDGE_26 26 26
# 	EDGE_27 27 27
# 	EDGE_28 28 28
# 	EDGE_29 29 29
# 	EDGE_30 30 30
# 	EDGE_31 31 31
regGFX_IMU_PIC_INT_LVL = 0x4081
# 	LVL_0 0 0
# 	LVL_1 1 1
# 	LVL_2 2 2
# 	LVL_3 3 3
# 	LVL_4 4 4
# 	LVL_5 5 5
# 	LVL_6 6 6
# 	LVL_7 7 7
# 	LVL_8 8 8
# 	LVL_9 9 9
# 	LVL_10 10 10
# 	LVL_11 11 11
# 	LVL_12 12 12
# 	LVL_13 13 13
# 	LVL_14 14 14
# 	LVL_15 15 15
# 	LVL_16 16 16
# 	LVL_17 17 17
# 	LVL_18 18 18
# 	LVL_19 19 19
# 	LVL_20 20 20
# 	LVL_21 21 21
# 	LVL_22 22 22
# 	LVL_23 23 23
# 	LVL_24 24 24
# 	LVL_25 25 25
# 	LVL_26 26 26
# 	LVL_27 27 27
# 	LVL_28 28 28
# 	LVL_29 29 29
# 	LVL_30 30 30
# 	LVL_31 31 31
regGFX_IMU_PIC_INT_MASK = 0x4080
# 	MASK_0 0 0
# 	MASK_1 1 1
# 	MASK_2 2 2
# 	MASK_3 3 3
# 	MASK_4 4 4
# 	MASK_5 5 5
# 	MASK_6 6 6
# 	MASK_7 7 7
# 	MASK_8 8 8
# 	MASK_9 9 9
# 	MASK_10 10 10
# 	MASK_11 11 11
# 	MASK_12 12 12
# 	MASK_13 13 13
# 	MASK_14 14 14
# 	MASK_15 15 15
# 	MASK_16 16 16
# 	MASK_17 17 17
# 	MASK_18 18 18
# 	MASK_19 19 19
# 	MASK_20 20 20
# 	MASK_21 21 21
# 	MASK_22 22 22
# 	MASK_23 23 23
# 	MASK_24 24 24
# 	MASK_25 25 25
# 	MASK_26 26 26
# 	MASK_27 27 27
# 	MASK_28 28 28
# 	MASK_29 29 29
# 	MASK_30 30 30
# 	MASK_31 31 31
regGFX_IMU_PIC_INT_PRI_0 = 0x4083
# 	PRI_0 0 7
# 	PRI_1 8 15
# 	PRI_2 16 23
# 	PRI_3 24 31
regGFX_IMU_PIC_INT_PRI_1 = 0x4084
# 	PRI_4 0 7
# 	PRI_5 8 15
# 	PRI_6 16 23
# 	PRI_7 24 31
regGFX_IMU_PIC_INT_PRI_2 = 0x4085
# 	PRI_8 0 7
# 	PRI_9 8 15
# 	PRI_10 16 23
# 	PRI_11 24 31
regGFX_IMU_PIC_INT_PRI_3 = 0x4086
# 	PRI_12 0 7
# 	PRI_13 8 15
# 	PRI_14 16 23
# 	PRI_15 24 31
regGFX_IMU_PIC_INT_PRI_4 = 0x4087
# 	PRI_16 0 7
# 	PRI_17 8 15
# 	PRI_18 16 23
# 	PRI_19 24 31
regGFX_IMU_PIC_INT_PRI_5 = 0x4088
# 	PRI_20 0 7
# 	PRI_21 8 15
# 	PRI_22 16 23
# 	PRI_23 24 31
regGFX_IMU_PIC_INT_PRI_6 = 0x4089
# 	PRI_24 0 7
# 	PRI_25 8 15
# 	PRI_26 16 23
# 	PRI_27 24 31
regGFX_IMU_PIC_INT_PRI_7 = 0x408a
# 	PRI_28 0 7
# 	PRI_29 8 15
# 	PRI_30 16 23
# 	PRI_31 24 31
regGFX_IMU_PIC_INT_STATUS = 0x408b
# 	INT_STATUS0 0 0
# 	INT_STATUS1 1 1
# 	INT_STATUS2 2 2
# 	INT_STATUS3 3 3
# 	INT_STATUS4 4 4
# 	INT_STATUS5 5 5
# 	INT_STATUS6 6 6
# 	INT_STATUS7 7 7
# 	INT_STATUS8 8 8
# 	INT_STATUS9 9 9
# 	INT_STATUS10 10 10
# 	INT_STATUS11 11 11
# 	INT_STATUS12 12 12
# 	INT_STATUS13 13 13
# 	INT_STATUS14 14 14
# 	INT_STATUS15 15 15
# 	INT_STATUS16 16 16
# 	INT_STATUS17 17 17
# 	INT_STATUS18 18 18
# 	INT_STATUS19 19 19
# 	INT_STATUS20 20 20
# 	INT_STATUS21 21 21
# 	INT_STATUS22 22 22
# 	INT_STATUS23 23 23
# 	INT_STATUS24 24 24
# 	INT_STATUS25 25 25
# 	INT_STATUS26 26 26
# 	INT_STATUS27 27 27
# 	INT_STATUS28 28 28
# 	INT_STATUS29 29 29
# 	INT_STATUS30 30 30
# 	INT_STATUS31 31 31
regGFX_IMU_PROGRAM_CTR = 0x40b5
# 	PC 0 31
regGFX_IMU_PWRMGT_IRQ_CTRL = 0x4042
# 	REQ 0 0
regGFX_IMU_PWROK = 0x40b9
# 	PWROK 0 0
regGFX_IMU_PWROKRAW = 0x40b8
# 	PWROKRAW 0 0
regGFX_IMU_RESETn = 0x40bb
# 	Cpl_RESETn 0 0
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_SIZE = 0x5f83
regGFX_IMU_RLC_CG_CTRL = 0x40a0
# 	FORCE_CGCG 0 0
# 	MGCG_EARLY_EN 1 1
regGFX_IMU_RLC_CMD = 0x404b
# 	CMD 0 31
regGFX_IMU_RLC_DATA_0 = 0x404a
# 	DATA 0 31
regGFX_IMU_RLC_DATA_1 = 0x4049
# 	DATA 0 31
regGFX_IMU_RLC_DATA_2 = 0x4048
# 	DATA 0 31
regGFX_IMU_RLC_DATA_3 = 0x4047
# 	DATA 0 31
regGFX_IMU_RLC_DATA_4 = 0x4046
# 	DATA 0 31
regGFX_IMU_RLC_GTS_OFFSET_HI = 0x407d
# 	GTS_OFFSET_HI 0 23
regGFX_IMU_RLC_GTS_OFFSET_LO = 0x407c
# 	GTS_OFFSET_LO 0 31
regGFX_IMU_RLC_MSG_STATUS = 0x404f
# 	IMU2RLC_BUSY 0 0
# 	IMU2RLC_MSG_ERROR 1 1
# 	RLC2IMU_MSGDONE 16 16
# 	RLC2IMU_CHGTOG 30 30
# 	RLC2IMU_DONETOG 31 31
regGFX_IMU_RLC_MUTEX = 0x404c
# 	MUTEX 0 1
regGFX_IMU_RLC_OVERRIDE = 0x40a3
# 	DS_ALLOW 0 0
regGFX_IMU_RLC_RAM_ADDR_HIGH = 0x40ad
# 	ADDR_MSB 0 15
regGFX_IMU_RLC_RAM_ADDR_LOW = 0x40ae
# 	ADDR_LSB 0 31
regGFX_IMU_RLC_RAM_DATA = 0x40af
# 	DATA 0 31
regGFX_IMU_RLC_RAM_INDEX = 0x40ac
# 	INDEX 0 7
# 	RLC_INDEX 16 23
# 	RAM_VALID 31 31
regGFX_IMU_RLC_RESET_VECTOR = 0x40a2
# 	COLD_VS_GFXOFF 0 0
# 	WARM_RESET_EXIT 2 2
# 	VF_FLR_EXIT 3 3
# 	VECTOR 4 7
regGFX_IMU_RLC_STATUS = 0x4054
# 	PD_ACTIVE 0 0
# 	RLC_ALIVE 1 1
# 	TBD2 2 2
# 	TBD3 3 3
regGFX_IMU_RLC_THROTTLE_GFX = 0x40a1
# 	THROTTLE_EN 0 0
regGFX_IMU_SCRATCH_0 = 0x4068
# 	DATA 0 31
regGFX_IMU_SCRATCH_1 = 0x4069
# 	DATA 0 31
regGFX_IMU_SCRATCH_10 = 0x4072
# 	DATA 0 31
regGFX_IMU_SCRATCH_11 = 0x4073
# 	DATA 0 31
regGFX_IMU_SCRATCH_12 = 0x4074
# 	DATA 0 31
regGFX_IMU_SCRATCH_13 = 0x4075
# 	DATA 0 31
regGFX_IMU_SCRATCH_14 = 0x4076
# 	DATA 0 31
regGFX_IMU_SCRATCH_15 = 0x4077
# 	DATA 0 31
regGFX_IMU_SCRATCH_2 = 0x406a
# 	DATA 0 31
regGFX_IMU_SCRATCH_3 = 0x406b
# 	DATA 0 31
regGFX_IMU_SCRATCH_4 = 0x406c
# 	DATA 0 31
regGFX_IMU_SCRATCH_5 = 0x406d
# 	DATA 0 31
regGFX_IMU_SCRATCH_6 = 0x406e
# 	DATA 0 31
regGFX_IMU_SCRATCH_7 = 0x406f
# 	DATA 0 31
regGFX_IMU_SCRATCH_8 = 0x4070
# 	DATA 0 31
regGFX_IMU_SCRATCH_9 = 0x4071
# 	DATA 0 31
regGFX_IMU_SMUIO_VIDCHG_CTRL = 0x4098
# 	REQ 0 0
# 	DATA 1 9
# 	PSIEN 10 10
# 	ACK 11 11
# 	SRC_SEL 31 31
regGFX_IMU_SOC_ADDR = 0x405a
# 	ADDR 0 31
regGFX_IMU_SOC_DATA = 0x4059
# 	DATA 0 31
regGFX_IMU_SOC_REQ = 0x405b
# 	REQ_BUSY 0 0
# 	R_W 1 1
# 	ERR 31 31
regGFX_IMU_STATUS = 0x4055
# 	ALLOW_GFXOFF 0 0
# 	ALLOW_FA_DCS 1 1
# 	TBD2 2 2
# 	TBD3 3 3
# 	TBD4 4 4
# 	TBD5 5 5
# 	TBD6 6 6
# 	TBD7 7 7
# 	TBD8 8 8
# 	TBD9 9 9
# 	TBD10 10 10
# 	TBD11 11 11
# 	TBD12 12 12
# 	TBD13 13 13
# 	TBD14 14 14
# 	DISABLE_GFXCLK_DS 15 15
regGFX_IMU_TELEMETRY = 0x4060
# 	TELEMETRY_ENTRIES 0 4
# 	TELEMETRY_DATA_SAMPLE_SIZE 5 5
# 	FIFO_OVERFLOW 6 6
# 	FIFO_UNDERFLOW 7 7
# 	FSM_STATE 8 10
# 	SVI_TYPE 12 13
# 	ENABLE_FIFO 30 30
# 	ENABLE_IMU_RLC_TELEMETRY 31 31
regGFX_IMU_TELEMETRY_DATA = 0x4061
# 	CURRENT 0 15
# 	VOLTAGE 16 31
regGFX_IMU_TELEMETRY_TEMPERATURE = 0x4062
# 	TEMPERATURE 0 15
regGFX_IMU_TIMER0_CMP0 = 0x40c4
# 	VALUE 0 31
regGFX_IMU_TIMER0_CMP1 = 0x40c5
# 	VALUE 0 31
regGFX_IMU_TIMER0_CMP3 = 0x40c7
# 	VALUE 0 31
regGFX_IMU_TIMER0_CMP_AUTOINC = 0x40c2
# 	AUTOINC_EN0 0 0
# 	AUTOINC_EN1 1 1
# 	AUTOINC_EN2 2 2
# 	AUTOINC_EN3 3 3
regGFX_IMU_TIMER0_CMP_INTEN = 0x40c3
# 	INT_EN0 0 0
# 	INT_EN1 1 1
# 	INT_EN2 2 2
# 	INT_EN3 3 3
regGFX_IMU_TIMER0_CTRL0 = 0x40c0
# 	START_STOP 0 0
# 	CLEAR 8 8
# 	UP_DOWN 16 16
# 	PULSE_EN 24 24
regGFX_IMU_TIMER0_CTRL1 = 0x40c1
# 	PWM_EN 0 0
# 	TS_MODE 8 8
# 	SAT_EN 16 16
regGFX_IMU_TIMER0_VALUE = 0x40c8
# 	VALUE 0 31
regGFX_IMU_TIMER1_CMP0 = 0x40cd
# 	VALUE 0 31
regGFX_IMU_TIMER1_CMP1 = 0x40ce
# 	VALUE 0 31
regGFX_IMU_TIMER1_CMP3 = 0x40d0
# 	VALUE 0 31
regGFX_IMU_TIMER1_CMP_AUTOINC = 0x40cb
# 	AUTOINC_EN0 0 0
# 	AUTOINC_EN1 1 1
# 	AUTOINC_EN2 2 2
# 	AUTOINC_EN3 3 3
regGFX_IMU_TIMER1_CMP_INTEN = 0x40cc
# 	INT_EN0 0 0
# 	INT_EN1 1 1
# 	INT_EN2 2 2
# 	INT_EN3 3 3
regGFX_IMU_TIMER1_CTRL0 = 0x40c9
# 	START_STOP 0 0
# 	CLEAR 8 8
# 	UP_DOWN 16 16
# 	PULSE_EN 24 24
regGFX_IMU_TIMER1_CTRL1 = 0x40ca
# 	PWM_EN 0 0
# 	TS_MODE 8 8
# 	SAT_EN 16 16
regGFX_IMU_TIMER1_VALUE = 0x40d1
# 	VALUE 0 31
regGFX_IMU_TIMER2_CMP0 = 0x40d6
# 	VALUE 0 31
regGFX_IMU_TIMER2_CMP1 = 0x40d7
# 	VALUE 0 31
regGFX_IMU_TIMER2_CMP3 = 0x40d9
# 	VALUE 0 31
regGFX_IMU_TIMER2_CMP_AUTOINC = 0x40d4
# 	AUTOINC_EN0 0 0
# 	AUTOINC_EN1 1 1
# 	AUTOINC_EN2 2 2
# 	AUTOINC_EN3 3 3
regGFX_IMU_TIMER2_CMP_INTEN = 0x40d5
# 	INT_EN0 0 0
# 	INT_EN1 1 1
# 	INT_EN2 2 2
# 	INT_EN3 3 3
regGFX_IMU_TIMER2_CTRL0 = 0x40d2
# 	START_STOP 0 0
# 	CLEAR 8 8
# 	UP_DOWN 16 16
# 	PULSE_EN 24 24
regGFX_IMU_TIMER2_CTRL1 = 0x40d3
# 	PWM_EN 0 0
# 	TS_MODE 8 8
# 	SAT_EN 16 16
regGFX_IMU_TIMER2_VALUE = 0x40da
# 	VALUE 0 31
regGFX_IMU_VDCI_RESET_CTRL = 0x40be
# 	SOC2GFX_VDCI_RESETn 0 0
# 	SOC_EA_SDF_VDCI_RESET 1 1
# 	SOC_UTCL2_ATHUB_VDCI_RESET 2 2
# 	IMU2GFX_VDCI_RESETn 4 4
regGFX_IMU_VF_CTRL = 0x405c
# 	VF 0 0
# 	VFID 1 6
# 	QOS 7 10
regGFX_PIPE_CONTROL = 0x100d
# 	HYSTERESIS_CNT 0 12
# 	RESERVED 13 15
# 	CONTEXT_SUSPEND_EN 16 16
# 	CONTEXT_SUSPEND_STALL_EN 17 17
regGFX_PIPE_PRIORITY = 0x587f
# 	HP_PIPE_SELECT 0 0
regGL1A_PERFCOUNTER0_HI = 0x35c1
# 	PERFCOUNTER_HI 0 31
regGL1A_PERFCOUNTER0_LO = 0x35c0
# 	PERFCOUNTER_LO 0 31
regGL1A_PERFCOUNTER0_SELECT = 0x3dc0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL1A_PERFCOUNTER0_SELECT1 = 0x3dc1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL1A_PERFCOUNTER1_HI = 0x35c3
# 	PERFCOUNTER_HI 0 31
regGL1A_PERFCOUNTER1_LO = 0x35c2
# 	PERFCOUNTER_LO 0 31
regGL1A_PERFCOUNTER1_SELECT = 0x3dc2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1A_PERFCOUNTER2_HI = 0x35c5
# 	PERFCOUNTER_HI 0 31
regGL1A_PERFCOUNTER2_LO = 0x35c4
# 	PERFCOUNTER_LO 0 31
regGL1A_PERFCOUNTER2_SELECT = 0x3dc3
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1A_PERFCOUNTER3_HI = 0x35c7
# 	PERFCOUNTER_HI 0 31
regGL1A_PERFCOUNTER3_LO = 0x35c6
# 	PERFCOUNTER_LO 0 31
regGL1A_PERFCOUNTER3_SELECT = 0x3dc4
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1C_PERFCOUNTER0_HI = 0x33a1
# 	PERFCOUNTER_HI 0 31
regGL1C_PERFCOUNTER0_LO = 0x33a0
# 	PERFCOUNTER_LO 0 31
regGL1C_PERFCOUNTER0_SELECT = 0x3ba0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL1C_PERFCOUNTER0_SELECT1 = 0x3ba1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL1C_PERFCOUNTER1_HI = 0x33a3
# 	PERFCOUNTER_HI 0 31
regGL1C_PERFCOUNTER1_LO = 0x33a2
# 	PERFCOUNTER_LO 0 31
regGL1C_PERFCOUNTER1_SELECT = 0x3ba2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1C_PERFCOUNTER2_HI = 0x33a5
# 	PERFCOUNTER_HI 0 31
regGL1C_PERFCOUNTER2_LO = 0x33a4
# 	PERFCOUNTER_LO 0 31
regGL1C_PERFCOUNTER2_SELECT = 0x3ba3
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1C_PERFCOUNTER3_HI = 0x33a7
# 	PERFCOUNTER_HI 0 31
regGL1C_PERFCOUNTER3_LO = 0x33a6
# 	PERFCOUNTER_LO 0 31
regGL1C_PERFCOUNTER3_SELECT = 0x3ba4
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1C_STATUS = 0x2d41
# 	INPUT_BUFFER_VC0_FIFO_FULL 0 0
# 	OUTPUT_FIFOS_BUSY 1 1
# 	SRC_DATA_FIFO_VC0_FULL 2 2
# 	GL2_REQ_VC0_STALL 3 3
# 	GL2_DATA_VC0_STALL 4 4
# 	GL2_REQ_VC1_STALL 5 5
# 	GL2_DATA_VC1_STALL 6 6
# 	INPUT_BUFFER_VC0_BUSY 7 7
# 	SRC_DATA_FIFO_VC0_BUSY 8 8
# 	GL2_RH_BUSY 9 9
# 	NUM_REQ_PENDING_FROM_L2 10 19
# 	LATENCY_FIFO_FULL_STALL 20 20
# 	TAG_STALL 21 21
# 	TAG_BUSY 22 22
# 	TAG_ACK_STALL 23 23
# 	TAG_GCR_INV_STALL 24 24
# 	TAG_NO_AVAILABLE_LINE_TO_EVICT_STALL 25 25
# 	TAG_EVICT 26 26
# 	TAG_REQUEST_STATE_OPERATION 27 30
# 	TRACKER_LAST_SET_MATCHES_CURRENT_SET 31 31
regGL1C_UTCL0_CNTL1 = 0x2d42
# 	FORCE_4K_L2_RESP 0 0
# 	GPUVM_64K_DEF 1 1
# 	GPUVM_PERM_MODE 2 2
# 	RESP_MODE 3 4
# 	RESP_FAULT_MODE 5 6
# 	CLIENTID 7 15
# 	REG_INV_VMID 19 22
# 	REG_INV_TOGGLE 24 24
# 	FORCE_MISS 26 26
# 	FORCE_IN_ORDER 25 26
# 	REDUCE_FIFO_DEPTH_BY_2 28 29
# 	REDUCE_CACHE_SIZE_BY_2 30 31
regGL1C_UTCL0_CNTL2 = 0x2d43
# 	SPARE 0 7
# 	COMP_SYNC_DISABLE 8 8
# 	MTYPE_OVRD_DIS 9 9
# 	ANY_LINE_VALID 10 10
# 	FORCE_SNOOP 14 14
# 	DISABLE_BURST 17 17
# 	FORCE_FRAG_2M_TO_64K 26 26
# 	FGCG_DISABLE 30 30
# 	BIG_PAGE_DISABLE 31 31
regGL1C_UTCL0_RETRY = 0x2d45
# 	INCR 0 7
# 	COUNT 8 11
regGL1C_UTCL0_STATUS = 0x2d44
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
regGL1H_ARB_CTRL = 0x2e40
# 	REQ_FGCG_DISABLE 0 0
# 	SRC_FGCG_DISABLE 1 1
# 	RET_FGCG_DISABLE 2 2
# 	CHICKEN_BITS 3 10
# 	PERF_CNTR_EN_OVERRIDE 11 11
regGL1H_ARB_STATUS = 0x2e44
# 	REQ_ARB_BUSY 0 0
# 	CLIENT1_ILLEGAL_REQ 1 1
regGL1H_BURST_CTRL = 0x2e43
# 	MAX_BURST_SIZE 0 2
# 	BURST_DISABLE 3 3
# 	SPARE_BURST_CTRL_BITS 4 5
regGL1H_BURST_MASK = 0x2e42
# 	BURST_ADDR_MASK 0 7
regGL1H_GL1_CREDITS = 0x2e41
# 	GL1_REQ_CREDITS 0 7
regGL1H_ICG_CTRL = 0x50e8
# 	REG_DCLK_OVERRIDE 0 0
# 	REQ_ARB_DCLK_OVERRIDE 1 1
# 	PERFMON_DCLK_OVERRIDE 2 2
# 	REQ_ARB_CLI0_DCLK_OVERRIDE 3 3
# 	REQ_ARB_CLI1_DCLK_OVERRIDE 4 4
# 	REQ_ARB_CLI2_DCLK_OVERRIDE 5 5
# 	REQ_ARB_CLI3_DCLK_OVERRIDE 6 6
# 	SRC_DCLK_OVERRIDE 7 7
# 	RET_DCLK_OVERRIDE 8 8
regGL1H_PERFCOUNTER0_HI = 0x35d1
# 	PERFCOUNTER_HI 0 31
regGL1H_PERFCOUNTER0_LO = 0x35d0
# 	PERFCOUNTER_LO 0 31
regGL1H_PERFCOUNTER0_SELECT = 0x3dd0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL1H_PERFCOUNTER0_SELECT1 = 0x3dd1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL1H_PERFCOUNTER1_HI = 0x35d3
# 	PERFCOUNTER_HI 0 31
regGL1H_PERFCOUNTER1_LO = 0x35d2
# 	PERFCOUNTER_LO 0 31
regGL1H_PERFCOUNTER1_SELECT = 0x3dd2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1H_PERFCOUNTER2_HI = 0x35d5
# 	PERFCOUNTER_HI 0 31
regGL1H_PERFCOUNTER2_LO = 0x35d4
# 	PERFCOUNTER_LO 0 31
regGL1H_PERFCOUNTER2_SELECT = 0x3dd3
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1H_PERFCOUNTER3_HI = 0x35d7
# 	PERFCOUNTER_HI 0 31
regGL1H_PERFCOUNTER3_LO = 0x35d6
# 	PERFCOUNTER_LO 0 31
regGL1H_PERFCOUNTER3_SELECT = 0x3dd4
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL1I_GL1R_MGCG_OVERRIDE = 0x50e4
# 	GL1A_GL1IR_MGCG_SCLK_OVERRIDE 0 0
# 	GL1A_GL1IR_MGCG_RET_DCLK_OVERRIDE 1 1
# 	GL1A_GL1IW_MGCG_SCLK_OVERRIDE 2 2
# 	GL1A_GL1IW_MGCG_RET_DCLK_OVERRIDE 3 3
# 	GL1A_GL1IW_MGCG_SRC_DCLK_OVERRIDE 4 4
# 	GL1A_GL1R_SRC_MGCG_SCLK_OVERRIDE 5 5
# 	GL1A_GL1R_RET_MGCG_SCLK_OVERRIDE 6 6
regGL1I_GL1R_REP_FGCG_OVERRIDE = 0x2d05
# 	GL1A_GL1IR_REP_FGCG_OVERRIDE 0 0
# 	GL1A_GL1IW_REP_FGCG_OVERRIDE 1 1
# 	GL1A_GL1R_SRC_REP_FGCG_OVERRIDE 2 2
# 	GL1A_GL1R_RET_REP_FGCG_OVERRIDE 3 3
regGL1_ARB_STATUS = 0x2d03
# 	REQ_ARB_BUSY 0 0
# 	RET_ARB_BUSY 1 1
regGL1_DRAM_BURST_MASK = 0x2d02
# 	DRAM_BURST_ADDR_MASK 0 7
regGL1_PIPE_STEER = 0x5b84
# 	PIPE0 0 1
# 	PIPE1 2 3
# 	PIPE2 4 5
# 	PIPE3 6 7
regGL2A_ADDR_MATCH_CTRL = 0x2e20
# 	DISABLE 0 31
regGL2A_ADDR_MATCH_MASK = 0x2e21
# 	ADDR_MASK 0 31
regGL2A_ADDR_MATCH_SIZE = 0x2e22
# 	MAX_COUNT 0 2
regGL2A_PERFCOUNTER0_HI = 0x3391
# 	PERFCOUNTER_HI 0 31
regGL2A_PERFCOUNTER0_LO = 0x3390
# 	PERFCOUNTER_LO 0 31
regGL2A_PERFCOUNTER0_SELECT = 0x3b90
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL2A_PERFCOUNTER0_SELECT1 = 0x3b91
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL2A_PERFCOUNTER1_HI = 0x3393
# 	PERFCOUNTER_HI 0 31
regGL2A_PERFCOUNTER1_LO = 0x3392
# 	PERFCOUNTER_LO 0 31
regGL2A_PERFCOUNTER1_SELECT = 0x3b92
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL2A_PERFCOUNTER1_SELECT1 = 0x3b93
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL2A_PERFCOUNTER2_HI = 0x3395
# 	PERFCOUNTER_HI 0 31
regGL2A_PERFCOUNTER2_LO = 0x3394
# 	PERFCOUNTER_LO 0 31
regGL2A_PERFCOUNTER2_SELECT = 0x3b94
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL2A_PERFCOUNTER3_HI = 0x3397
# 	PERFCOUNTER_HI 0 31
regGL2A_PERFCOUNTER3_LO = 0x3396
# 	PERFCOUNTER_LO 0 31
regGL2A_PERFCOUNTER3_SELECT = 0x3b95
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL2A_PRIORITY_CTRL = 0x2e23
# 	DISABLE 0 31
regGL2A_RESP_THROTTLE_CTRL = 0x2e2a
# 	DISABLE 0 15
# 	CREDIT_GL1 16 23
# 	CREDIT_CH 24 31
regGL2C_ADDR_MATCH_MASK = 0x2e03
# 	ADDR_MASK 0 31
regGL2C_ADDR_MATCH_SIZE = 0x2e04
# 	MAX_COUNT 0 2
regGL2C_CM_CTRL0 = 0x2e07
regGL2C_CM_CTRL1 = 0x2e08
# 	BURST_TIMER 8 15
# 	RVF_SIZE 16 19
# 	WRITE_COH_MODE 23 24
# 	MDC_ARB_MODE 25 25
# 	READ_REQ_ONLY 26 26
# 	COMP_TO_CONSTANT_EN 27 27
# 	COMP_TO_SINGLE_EN 28 28
# 	BURST_MODE 29 29
# 	UNCOMP_READBACK_FILTER 30 30
# 	WAIT_ATOMIC_RECOMP_WRITE 31 31
regGL2C_CM_STALL = 0x2e09
# 	QUEUE 0 31
regGL2C_CTRL = 0x2e00
# 	CACHE_SIZE 0 1
# 	RATE 2 3
# 	WRITEBACK_MARGIN 4 7
# 	METADATA_LATENCY_FIFO_SIZE 8 11
# 	SRC_FIFO_SIZE 12 15
# 	LATENCY_FIFO_SIZE 16 19
# 	METADATA_TO_HI_PRIORITY 20 20
# 	LINEAR_SET_HASH 21 21
# 	FORCE_HIT_QUEUE_POP 22 23
# 	MDC_SIZE 24 25
# 	METADATA_TO_HIT_QUEUE 26 26
# 	IGNORE_FULLY_WRITTEN 27 27
# 	MDC_SIDEBAND_FIFO_SIZE 28 31
regGL2C_CTRL2 = 0x2e01
# 	PROBE_FIFO_SIZE 0 3
# 	ADDR_MATCH_DISABLE 4 4
# 	FILL_SIZE_32 5 5
# 	RB_TO_HI_PRIORITY 6 6
# 	HIT_UNDER_MISS_DISABLE 7 7
# 	RO_DISABLE 8 8
# 	FORCE_MDC_INV 9 9
# 	GCR_ARB_CTRL 10 12
# 	GCR_ALL_SET 13 13
# 	FILL_SIZE_64 17 17
# 	USE_EA_EARLYWRRET_ON_WRITEBACK 18 18
# 	WRITEBACK_ALL_WAIT_FOR_ALL_EA_WRITE_COMPLETE 19 19
# 	METADATA_VOLATILE_EN 20 20
# 	RB_VOLATILE_EN 21 21
# 	PROBE_UNSHARED_EN 22 22
# 	MAX_MIN_CTRL 23 24
# 	MDC_UC_TO_C_RO_EN 26 26
regGL2C_CTRL3 = 0x2e0c
# 	METADATA_MTYPE_COHERENCY 0 1
# 	METADATA_NOFILL 3 3
# 	METADATA_NEXT_CL_PREFETCH 4 4
# 	BANK_LINEAR_HASH_MODE 5 5
# 	HTILE_TO_HI_PRIORITY 6 6
# 	UNCACHED_WRITE_ATOMIC_TO_UC_WRITE 7 7
# 	IO_CHANNEL_ENABLE 8 8
# 	FMASK_TO_HI_PRIORITY 9 9
# 	DCC_CMASK_TO_HI_PRIORITY 10 10
# 	BANK_LINEAR_HASH_ENABLE 11 11
# 	HASH_256B_ENABLE 12 12
# 	DECOMP_NBC_IND64_DISABLE 13 13
# 	FORCE_READ_ON_WRITE_OP 14 14
# 	FGCG_OVERRIDE 15 15
# 	FORCE_MTYPE_UC 16 16
# 	DGPU_SHARED_MODE 17 17
# 	WRITE_SET_SECTOR_FULLY_WRITTEN 18 18
# 	EA_READ_SIZE_LIMIT 19 19
# 	READ_BYPASS_AS_UC 20 20
# 	WB_OPT_ENABLE 21 21
# 	WB_OPT_BURST_MAX_COUNT 22 23
# 	SET_GROUP_LINEAR_HASH_ENABLE 24 24
# 	EA_GMI_DISABLE 25 25
# 	SQC_TO_HI_PRIORITY 26 26
# 	INF_NAN_CLAMP 27 27
# 	SCRATCH 28 31
regGL2C_CTRL4 = 0x2e17
# 	METADATA_WR_OP_CID 0 0
# 	SPA_CHANNEL_ENABLE 1 1
# 	SRC_FIFO_MDC_LOW_PRIORITY 2 2
# 	WRITEBACK_FIFO_STALL_ENABLE 3 3
# 	CM_MGCG_MODE 4 4
# 	MDC_MGCG_MODE 5 5
# 	TAG_MGCG_MODE 6 6
# 	CORE_MGCG_MODE 7 7
# 	EXECUTE_MGCG_MODE 8 8
# 	EA_NACK_DISABLE 9 9
# 	NO_WRITE_ACK_TO_HIT_QUEUE 26 26
regGL2C_DISCARD_STALL_CTRL = 0x2e18
# 	LIMIT 0 14
# 	WINDOW 15 29
# 	DROP_NEXT 30 30
# 	ENABLE 31 31
regGL2C_LB_CTR_CTRL = 0x2e0d
# 	START 0 0
# 	LOAD 1 1
# 	CLEAR 2 2
# 	PERF_CNTR_EN_OVERRIDE 31 31
regGL2C_LB_CTR_SEL0 = 0x2e12
# 	SEL0 0 7
# 	DIV0 15 15
# 	SEL1 16 23
# 	DIV1 31 31
regGL2C_LB_CTR_SEL1 = 0x2e13
# 	SEL2 0 7
# 	DIV2 15 15
# 	SEL3 16 23
# 	DIV3 31 31
regGL2C_LB_DATA0 = 0x2e0e
# 	DATA 0 31
regGL2C_LB_DATA1 = 0x2e0f
# 	DATA 0 31
regGL2C_LB_DATA2 = 0x2e10
# 	DATA 0 31
regGL2C_LB_DATA3 = 0x2e11
# 	DATA 0 31
regGL2C_PERFCOUNTER0_HI = 0x3381
# 	PERFCOUNTER_HI 0 31
regGL2C_PERFCOUNTER0_LO = 0x3380
# 	PERFCOUNTER_LO 0 31
regGL2C_PERFCOUNTER0_SELECT = 0x3b80
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL2C_PERFCOUNTER0_SELECT1 = 0x3b81
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL2C_PERFCOUNTER1_HI = 0x3383
# 	PERFCOUNTER_HI 0 31
regGL2C_PERFCOUNTER1_LO = 0x3382
# 	PERFCOUNTER_LO 0 31
regGL2C_PERFCOUNTER1_SELECT = 0x3b82
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGL2C_PERFCOUNTER1_SELECT1 = 0x3b83
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE2 24 27
# 	PERF_MODE3 28 31
regGL2C_PERFCOUNTER2_HI = 0x3385
# 	PERFCOUNTER_HI 0 31
regGL2C_PERFCOUNTER2_LO = 0x3384
# 	PERFCOUNTER_LO 0 31
regGL2C_PERFCOUNTER2_SELECT = 0x3b84
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL2C_PERFCOUNTER3_HI = 0x3387
# 	PERFCOUNTER_HI 0 31
regGL2C_PERFCOUNTER3_LO = 0x3386
# 	PERFCOUNTER_LO 0 31
regGL2C_PERFCOUNTER3_SELECT = 0x3b85
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regGL2C_SOFT_RESET = 0x2e06
# 	HALT_FOR_RESET 0 0
regGL2C_WBINVL2 = 0x2e05
# 	DONE 4 4
regGL2_PIPE_STEER_0 = 0x5b80
# 	PIPE_0_TO_CHAN_IN_Q0 0 2
# 	PIPE_1_TO_CHAN_IN_Q0 4 6
# 	PIPE_2_TO_CHAN_IN_Q0 8 10
# 	PIPE_3_TO_CHAN_IN_Q0 12 14
# 	PIPE_0_TO_CHAN_IN_Q1 16 18
# 	PIPE_1_TO_CHAN_IN_Q1 20 22
# 	PIPE_2_TO_CHAN_IN_Q1 24 26
# 	PIPE_3_TO_CHAN_IN_Q1 28 30
regGL2_PIPE_STEER_1 = 0x5b81
# 	PIPE_0_TO_CHAN_IN_Q2 0 2
# 	PIPE_1_TO_CHAN_IN_Q2 4 6
# 	PIPE_2_TO_CHAN_IN_Q2 8 10
# 	PIPE_3_TO_CHAN_IN_Q2 12 14
# 	PIPE_0_TO_CHAN_IN_Q3 16 18
# 	PIPE_1_TO_CHAN_IN_Q3 20 22
# 	PIPE_2_TO_CHAN_IN_Q3 24 26
# 	PIPE_3_TO_CHAN_IN_Q3 28 30
regGL2_PIPE_STEER_2 = 0x5b82
# 	PIPE_4_TO_CHAN_IN_Q0 0 2
# 	PIPE_5_TO_CHAN_IN_Q0 4 6
# 	PIPE_6_TO_CHAN_IN_Q0 8 10
# 	PIPE_7_TO_CHAN_IN_Q0 12 14
# 	PIPE_4_TO_CHAN_IN_Q1 16 18
# 	PIPE_5_TO_CHAN_IN_Q1 20 22
# 	PIPE_6_TO_CHAN_IN_Q1 24 26
# 	PIPE_7_TO_CHAN_IN_Q1 28 30
regGL2_PIPE_STEER_3 = 0x5b83
# 	PIPE_4_TO_CHAN_IN_Q2 0 2
# 	PIPE_5_TO_CHAN_IN_Q2 4 6
# 	PIPE_6_TO_CHAN_IN_Q2 8 10
# 	PIPE_7_TO_CHAN_IN_Q2 12 14
# 	PIPE_4_TO_CHAN_IN_Q3 16 18
# 	PIPE_5_TO_CHAN_IN_Q3 20 22
# 	PIPE_6_TO_CHAN_IN_Q3 24 26
# 	PIPE_7_TO_CHAN_IN_Q3 28 30
regGRBM_CAM_DATA = 0x5e11
# 	CAM_ADDR 0 15
# 	CAM_REMAPADDR 16 31
regGRBM_CAM_DATA_UPPER = 0x5e12
# 	CAM_ADDR 0 1
# 	CAM_REMAPADDR 16 17
regGRBM_CAM_INDEX = 0x5e10
# 	CAM_INDEX 0 3
regGRBM_CHIP_REVISION = 0xdc1
# 	CHIP_REVISION 0 7
regGRBM_CNTL = 0xda0
# 	READ_TIMEOUT 0 7
# 	REPORT_LAST_RDERR 31 31
regGRBM_DSM_BYPASS = 0xdbe
# 	BYPASS_BITS 0 1
# 	BYPASS_EN 2 2
regGRBM_FENCE_RANGE0 = 0xdca
# 	START 0 15
# 	END 16 31
regGRBM_FENCE_RANGE1 = 0xdcb
# 	START 0 15
# 	END 16 31
regGRBM_GFX_CLKEN_CNTL = 0xdac
# 	PREFIX_DELAY_CNT 0 3
# 	POST_DELAY_CNT 8 12
regGRBM_GFX_CNTL = 0x900
# 	PIPEID 0 1
# 	MEID 2 3
# 	VMID 4 7
# 	QUEUEID 8 10
# 	CTXID 11 13
regGRBM_GFX_CNTL_SR_DATA = 0x5a03
# 	PIPEID 0 1
# 	MEID 2 3
# 	VMID 4 7
# 	QUEUEID 8 10
regGRBM_GFX_CNTL_SR_SELECT = 0x5a02
# 	INDEX 0 2
# 	VF_PF 31 31
regGRBM_GFX_INDEX = 0x2200
# 	INSTANCE_INDEX 0 7
# 	SA_INDEX 8 15
# 	SE_INDEX 16 23
# 	SA_BROADCAST_WRITES 29 29
# 	INSTANCE_BROADCAST_WRITES 30 30
# 	SE_BROADCAST_WRITES 31 31
regGRBM_GFX_INDEX_SR_DATA = 0x5a01
# 	INSTANCE_INDEX 0 7
# 	SA_INDEX 8 15
# 	SE_INDEX 16 23
# 	SA_BROADCAST_WRITES 29 29
# 	INSTANCE_BROADCAST_WRITES 30 30
# 	SE_BROADCAST_WRITES 31 31
regGRBM_GFX_INDEX_SR_SELECT = 0x5a00
# 	INDEX 0 2
# 	VF_PF 31 31
regGRBM_HYP_CAM_DATA = 0x5e11
# 	CAM_ADDR 0 15
# 	CAM_REMAPADDR 16 31
regGRBM_HYP_CAM_DATA_UPPER = 0x5e12
# 	CAM_ADDR 0 1
# 	CAM_REMAPADDR 16 17
regGRBM_HYP_CAM_INDEX = 0x5e10
# 	CAM_INDEX 0 3
regGRBM_IH_CREDIT = 0xdc4
# 	CREDIT_VALUE 0 1
# 	IH_CLIENT_ID 16 23
regGRBM_INT_CNTL = 0xdb8
# 	RDERR_INT_ENABLE 0 0
# 	GUI_IDLE_INT_ENABLE 19 19
regGRBM_INVALID_PIPE = 0xdc9
# 	ADDR 2 19
# 	PIPEID 20 21
# 	MEID 22 23
# 	QUEUEID 24 26
# 	SSRCID 27 30
# 	INVALID_PIPE 31 31
regGRBM_NOWHERE = 0x901
# 	DATA 0 31
regGRBM_PERFCOUNTER0_HI = 0x3041
# 	PERFCOUNTER_HI 0 31
regGRBM_PERFCOUNTER0_LO = 0x3040
# 	PERFCOUNTER_LO 0 31
regGRBM_PERFCOUNTER0_SELECT = 0x3840
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 13 13
# 	SX_BUSY_USER_DEFINED_MASK 14 14
# 	SPI_BUSY_USER_DEFINED_MASK 16 16
# 	SC_BUSY_USER_DEFINED_MASK 17 17
# 	PA_BUSY_USER_DEFINED_MASK 18 18
# 	GRBM_BUSY_USER_DEFINED_MASK 19 19
# 	DB_BUSY_USER_DEFINED_MASK 20 20
# 	CB_BUSY_USER_DEFINED_MASK 21 21
# 	CP_BUSY_USER_DEFINED_MASK 22 22
# 	GDS_BUSY_USER_DEFINED_MASK 24 24
# 	BCI_BUSY_USER_DEFINED_MASK 25 25
# 	RLC_BUSY_USER_DEFINED_MASK 26 26
# 	TCP_BUSY_USER_DEFINED_MASK 27 27
# 	GE_BUSY_USER_DEFINED_MASK 28 28
# 	UTCL2_BUSY_USER_DEFINED_MASK 29 29
# 	EA_BUSY_USER_DEFINED_MASK 30 30
# 	RMI_BUSY_USER_DEFINED_MASK 31 31
regGRBM_PERFCOUNTER0_SELECT_HI = 0x384d
# 	UTCL1_BUSY_USER_DEFINED_MASK 1 1
# 	GL2CC_BUSY_USER_DEFINED_MASK 2 2
# 	SDMA_BUSY_USER_DEFINED_MASK 3 3
# 	CH_BUSY_USER_DEFINED_MASK 4 4
# 	PH_BUSY_USER_DEFINED_MASK 5 5
# 	PMM_BUSY_USER_DEFINED_MASK 6 6
# 	GUS_BUSY_USER_DEFINED_MASK 7 7
# 	GL1CC_BUSY_USER_DEFINED_MASK 8 8
# 	GL1H_BUSY_USER_DEFINED_MASK 9 9
regGRBM_PERFCOUNTER1_HI = 0x3044
# 	PERFCOUNTER_HI 0 31
regGRBM_PERFCOUNTER1_LO = 0x3043
# 	PERFCOUNTER_LO 0 31
regGRBM_PERFCOUNTER1_SELECT = 0x3841
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 13 13
# 	SX_BUSY_USER_DEFINED_MASK 14 14
# 	SPI_BUSY_USER_DEFINED_MASK 16 16
# 	SC_BUSY_USER_DEFINED_MASK 17 17
# 	PA_BUSY_USER_DEFINED_MASK 18 18
# 	GRBM_BUSY_USER_DEFINED_MASK 19 19
# 	DB_BUSY_USER_DEFINED_MASK 20 20
# 	CB_BUSY_USER_DEFINED_MASK 21 21
# 	CP_BUSY_USER_DEFINED_MASK 22 22
# 	GDS_BUSY_USER_DEFINED_MASK 24 24
# 	BCI_BUSY_USER_DEFINED_MASK 25 25
# 	RLC_BUSY_USER_DEFINED_MASK 26 26
# 	TCP_BUSY_USER_DEFINED_MASK 27 27
# 	GE_BUSY_USER_DEFINED_MASK 28 28
# 	UTCL2_BUSY_USER_DEFINED_MASK 29 29
# 	EA_BUSY_USER_DEFINED_MASK 30 30
# 	RMI_BUSY_USER_DEFINED_MASK 31 31
regGRBM_PERFCOUNTER1_SELECT_HI = 0x384e
# 	UTCL1_BUSY_USER_DEFINED_MASK 1 1
# 	GL2CC_BUSY_USER_DEFINED_MASK 2 2
# 	SDMA_BUSY_USER_DEFINED_MASK 3 3
# 	CH_BUSY_USER_DEFINED_MASK 4 4
# 	PH_BUSY_USER_DEFINED_MASK 5 5
# 	PMM_BUSY_USER_DEFINED_MASK 6 6
# 	GUS_BUSY_USER_DEFINED_MASK 7 7
# 	GL1CC_BUSY_USER_DEFINED_MASK 8 8
# 	GL1H_BUSY_USER_DEFINED_MASK 9 9
regGRBM_PWR_CNTL = 0xda3
# 	ALL_REQ_TYPE 0 1
# 	GFX_REQ_TYPE 2 3
# 	ALL_RSP_TYPE 4 5
# 	GFX_RSP_TYPE 6 7
# 	GFX_REQ_EN 14 14
# 	ALL_REQ_EN 15 15
regGRBM_PWR_CNTL2 = 0xdc5
# 	PWR_REQUEST_HALT 16 16
# 	PWR_GFX3D_REQUEST_HALT 20 20
regGRBM_READ_ERROR = 0xdb6
# 	READ_ADDRESS 2 19
# 	READ_PIPEID 20 21
# 	READ_MEID 22 23
# 	READ_ERROR 31 31
regGRBM_READ_ERROR2 = 0xdb7
# 	READ_REQUESTER_MESPIPE0 9 9
# 	READ_REQUESTER_MESPIPE1 10 10
# 	READ_REQUESTER_MESPIPE2 11 11
# 	READ_REQUESTER_MESPIPE3 12 12
# 	READ_REQUESTER_SDMA0 13 13
# 	READ_REQUESTER_SDMA1 14 14
# 	READ_REQUESTER_RLC 18 18
# 	READ_REQUESTER_GDS_DMA 19 19
# 	READ_REQUESTER_ME0PIPE0_CF 20 20
# 	READ_REQUESTER_ME0PIPE0_PF 21 21
# 	READ_REQUESTER_ME0PIPE1_CF 22 22
# 	READ_REQUESTER_ME0PIPE1_PF 23 23
# 	READ_REQUESTER_ME1PIPE0 24 24
# 	READ_REQUESTER_ME1PIPE1 25 25
# 	READ_REQUESTER_ME1PIPE2 26 26
# 	READ_REQUESTER_ME1PIPE3 27 27
# 	READ_REQUESTER_ME2PIPE0 28 28
# 	READ_REQUESTER_ME2PIPE1 29 29
# 	READ_REQUESTER_ME2PIPE2 30 30
# 	READ_REQUESTER_ME2PIPE3 31 31
regGRBM_SCRATCH_REG0 = 0xde0
# 	SCRATCH_REG0 0 31
regGRBM_SCRATCH_REG1 = 0xde1
# 	SCRATCH_REG1 0 31
regGRBM_SCRATCH_REG2 = 0xde2
# 	SCRATCH_REG2 0 31
regGRBM_SCRATCH_REG3 = 0xde3
# 	SCRATCH_REG3 0 31
regGRBM_SCRATCH_REG4 = 0xde4
# 	SCRATCH_REG4 0 31
regGRBM_SCRATCH_REG5 = 0xde5
# 	SCRATCH_REG5 0 31
regGRBM_SCRATCH_REG6 = 0xde6
# 	SCRATCH_REG6 0 31
regGRBM_SCRATCH_REG7 = 0xde7
# 	SCRATCH_REG7 0 31
regGRBM_SE0_PERFCOUNTER_HI = 0x3046
# 	PERFCOUNTER_HI 0 31
regGRBM_SE0_PERFCOUNTER_LO = 0x3045
# 	PERFCOUNTER_LO 0 31
regGRBM_SE0_PERFCOUNTER_SELECT = 0x3842
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE1_PERFCOUNTER_HI = 0x3048
# 	PERFCOUNTER_HI 0 31
regGRBM_SE1_PERFCOUNTER_LO = 0x3047
# 	PERFCOUNTER_LO 0 31
regGRBM_SE1_PERFCOUNTER_SELECT = 0x3843
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE2_PERFCOUNTER_HI = 0x304a
# 	PERFCOUNTER_HI 0 31
regGRBM_SE2_PERFCOUNTER_LO = 0x3049
# 	PERFCOUNTER_LO 0 31
regGRBM_SE2_PERFCOUNTER_SELECT = 0x3844
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE3_PERFCOUNTER_HI = 0x304c
# 	PERFCOUNTER_HI 0 31
regGRBM_SE3_PERFCOUNTER_LO = 0x304b
# 	PERFCOUNTER_LO 0 31
regGRBM_SE3_PERFCOUNTER_SELECT = 0x3845
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE4_PERFCOUNTER_HI = 0x304e
# 	PERFCOUNTER_HI 0 31
regGRBM_SE4_PERFCOUNTER_LO = 0x304d
# 	PERFCOUNTER_LO 0 31
regGRBM_SE4_PERFCOUNTER_SELECT = 0x3846
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE5_PERFCOUNTER_HI = 0x3050
# 	PERFCOUNTER_HI 0 31
regGRBM_SE5_PERFCOUNTER_LO = 0x304f
# 	PERFCOUNTER_LO 0 31
regGRBM_SE5_PERFCOUNTER_SELECT = 0x3847
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SE6_PERFCOUNTER_HI = 0x3052
# 	PERFCOUNTER_HI 0 31
regGRBM_SE6_PERFCOUNTER_LO = 0x3051
# 	PERFCOUNTER_LO 0 31
regGRBM_SE6_PERFCOUNTER_SELECT = 0x3848
# 	PERF_SEL 0 5
# 	DB_CLEAN_USER_DEFINED_MASK 10 10
# 	CB_CLEAN_USER_DEFINED_MASK 11 11
# 	TA_BUSY_USER_DEFINED_MASK 12 12
# 	SX_BUSY_USER_DEFINED_MASK 13 13
# 	SPI_BUSY_USER_DEFINED_MASK 15 15
# 	SC_BUSY_USER_DEFINED_MASK 16 16
# 	DB_BUSY_USER_DEFINED_MASK 17 17
# 	CB_BUSY_USER_DEFINED_MASK 18 18
# 	PA_BUSY_USER_DEFINED_MASK 20 20
# 	BCI_BUSY_USER_DEFINED_MASK 21 21
# 	RMI_BUSY_USER_DEFINED_MASK 22 22
# 	UTCL1_BUSY_USER_DEFINED_MASK 23 23
# 	TCP_BUSY_USER_DEFINED_MASK 24 24
# 	GL1CC_BUSY_USER_DEFINED_MASK 25 25
# 	GL1H_BUSY_USER_DEFINED_MASK 26 26
# 	PC_BUSY_USER_DEFINED_MASK 27 27
# 	SEDC_BUSY_USER_DEFINED_MASK 28 28
regGRBM_SEC_CNTL = 0x5e0d
regGRBM_SE_REMAP_CNTL = 0x5a08
# 	SE0_REMAP_EN 0 0
# 	SE0_REMAP 1 3
# 	SE1_REMAP_EN 4 4
# 	SE1_REMAP 5 7
# 	SE2_REMAP_EN 8 8
# 	SE2_REMAP 9 11
# 	SE3_REMAP_EN 12 12
# 	SE3_REMAP 13 15
# 	SE4_REMAP_EN 16 16
# 	SE4_REMAP 17 19
# 	SE5_REMAP_EN 20 20
# 	SE5_REMAP 21 23
# 	SE6_REMAP_EN 24 24
# 	SE6_REMAP 25 27
# 	SE7_REMAP_EN 28 28
# 	SE7_REMAP 29 31
regGRBM_SKEW_CNTL = 0xda1
# 	SKEW_TOP_THRESHOLD 0 5
# 	SKEW_COUNT 6 11
regGRBM_SOFT_RESET = 0xda8
# 	SOFT_RESET_CP 0 0
# 	SOFT_RESET_RLC 2 2
# 	SOFT_RESET_UTCL2 15 15
# 	SOFT_RESET_GFX 16 16
# 	SOFT_RESET_CPF 17 17
# 	SOFT_RESET_CPC 18 18
# 	SOFT_RESET_CPG 19 19
# 	SOFT_RESET_CAC 20 20
# 	SOFT_RESET_EA 22 22
# 	SOFT_RESET_SDMA0 23 23
# 	SOFT_RESET_SDMA1 24 24
regGRBM_STATUS = 0xda4
# 	ME0PIPE0_CMDFIFO_AVAIL 0 3
# 	SDMA_RQ_PENDING 6 6
# 	ME0PIPE0_CF_RQ_PENDING 7 7
# 	ME0PIPE0_PF_RQ_PENDING 8 8
# 	GDS_DMA_RQ_PENDING 9 9
# 	DB_CLEAN 12 12
# 	CB_CLEAN 13 13
# 	TA_BUSY 14 14
# 	GDS_BUSY 15 15
# 	GE_BUSY_NO_DMA 16 16
# 	SX_BUSY 20 20
# 	GE_BUSY 21 21
# 	SPI_BUSY 22 22
# 	BCI_BUSY 23 23
# 	SC_BUSY 24 24
# 	PA_BUSY 25 25
# 	DB_BUSY 26 26
# 	ANY_ACTIVE 27 27
# 	CP_COHERENCY_BUSY 28 28
# 	CP_BUSY 29 29
# 	CB_BUSY 30 30
# 	GUI_ACTIVE 31 31
regGRBM_STATUS2 = 0xda2
# 	ME0PIPE1_CMDFIFO_AVAIL 0 3
# 	ME0PIPE1_CF_RQ_PENDING 4 4
# 	ME0PIPE1_PF_RQ_PENDING 5 5
# 	ME1PIPE0_RQ_PENDING 6 6
# 	ME1PIPE1_RQ_PENDING 7 7
# 	ME1PIPE2_RQ_PENDING 8 8
# 	ME1PIPE3_RQ_PENDING 9 9
# 	RLC_RQ_PENDING 14 14
# 	UTCL2_BUSY 15 15
# 	EA_BUSY 16 16
# 	RMI_BUSY 17 17
# 	UTCL2_RQ_PENDING 18 18
# 	SDMA_SCH_RQ_PENDING 19 19
# 	EA_LINK_BUSY 20 20
# 	SDMA_BUSY 21 21
# 	SDMA0_RQ_PENDING 22 22
# 	SDMA1_RQ_PENDING 23 23
# 	RLC_BUSY 26 26
# 	TCP_BUSY 27 27
# 	CPF_BUSY 28 28
# 	CPC_BUSY 29 29
# 	CPG_BUSY 30 30
regGRBM_STATUS3 = 0xda7
# 	GRBM_RLC_INTR_CREDIT_PENDING 5 5
# 	GRBM_CPF_INTR_CREDIT_PENDING 7 7
# 	MESPIPE0_RQ_PENDING 8 8
# 	MESPIPE1_RQ_PENDING 9 9
# 	PH_BUSY 13 13
# 	CH_BUSY 14 14
# 	GL2CC_BUSY 15 15
# 	GL1CC_BUSY 16 16
# 	SEDC_BUSY 25 25
# 	PC_BUSY 26 26
# 	GL1H_BUSY 27 27
# 	GUS_LINK_BUSY 28 28
# 	GUS_BUSY 29 29
# 	UTCL1_BUSY 30 30
# 	PMM_BUSY 31 31
regGRBM_STATUS_SE0 = 0xda5
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_STATUS_SE1 = 0xda6
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_STATUS_SE2 = 0xdae
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_STATUS_SE3 = 0xdaf
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_STATUS_SE4 = 0xdb0
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_STATUS_SE5 = 0xdb1
# 	DB_CLEAN 1 1
# 	CB_CLEAN 2 2
# 	UTCL1_BUSY 3 3
# 	TCP_BUSY 4 4
# 	GL1CC_BUSY 5 5
# 	GL1H_BUSY 6 6
# 	PC_BUSY 7 7
# 	SEDC_BUSY 8 8
# 	RMI_BUSY 21 21
# 	BCI_BUSY 22 22
# 	PA_BUSY 24 24
# 	TA_BUSY 25 25
# 	SX_BUSY 26 26
# 	SPI_BUSY 27 27
# 	SC_BUSY 29 29
# 	DB_BUSY 30 30
# 	CB_BUSY 31 31
regGRBM_TRAP_ADDR = 0xdba
# 	DATA 0 17
regGRBM_TRAP_ADDR_MSK = 0xdbb
# 	DATA 0 17
regGRBM_TRAP_OP = 0xdb9
# 	RW 0 0
regGRBM_TRAP_WD = 0xdbc
# 	DATA 0 31
regGRBM_TRAP_WD_MSK = 0xdbd
# 	DATA 0 31
regGRBM_UTCL2_INVAL_RANGE_END = 0xdc7
# 	DATA 0 17
regGRBM_UTCL2_INVAL_RANGE_START = 0xdc6
# 	DATA 0 17
regGRBM_WAIT_IDLE_CLOCKS = 0xdad
# 	WAIT_IDLE_CLOCKS 0 7
regGRBM_WRITE_ERROR = 0xdbf
# 	WRITE_REQUESTER_RLC 0 0
# 	WRITE_SSRCID 2 5
# 	WRITE_VFID 8 11
# 	WRITE_VF 12 12
# 	WRITE_VMID 13 16
# 	TMZ 17 17
# 	WRITE_PIPEID 20 21
# 	WRITE_MEID 22 23
# 	WRITE_ERROR 31 31
regGRTAVFS_CLK_CNTL = 0x4b0e
# 	GRTAVFS_MUX_CLK_SEL 0 0
# 	FORCE_GRTAVFS_CLK_SEL 1 1
# 	RESERVED 2 31
regGRTAVFS_GENERAL_0 = 0x4b02
# 	DATA 0 31
regGRTAVFS_PSM_CNTL = 0x4b0d
# 	PSM_COUNT 0 13
# 	PSM_SAMPLE_EN 14 14
# 	RESERVED 15 31
regGRTAVFS_RTAVFS_RD_DATA = 0x4b03
# 	RTAVFSDATA 0 31
regGRTAVFS_RTAVFS_REG_ADDR = 0x4b00
# 	RTAVFSADDR 0 9
regGRTAVFS_RTAVFS_REG_CTRL = 0x4b04
# 	SET_WR_EN 0 0
# 	SET_RD_EN 1 1
regGRTAVFS_RTAVFS_REG_STATUS = 0x4b05
# 	RTAVFS_WR_ACK 0 0
# 	RTAVFS_RD_DATA_VALID 1 1
regGRTAVFS_RTAVFS_WR_DATA = 0x4b01
# 	RTAVFSDATA 0 31
regGRTAVFS_SE_CLK_CNTL = 0x4b4e
# 	GRTAVFS_MUX_CLK_SEL 0 0
# 	FORCE_GRTAVFS_CLK_SEL 1 1
# 	RESERVED 2 31
regGRTAVFS_SE_GENERAL_0 = 0x4b42
# 	DATA 0 31
regGRTAVFS_SE_PSM_CNTL = 0x4b4d
# 	PSM_COUNT 0 13
# 	PSM_SAMPLE_EN 14 14
# 	RESERVED 15 31
regGRTAVFS_SE_RTAVFS_RD_DATA = 0x4b43
# 	RTAVFSDATA 0 31
regGRTAVFS_SE_RTAVFS_REG_ADDR = 0x4b40
# 	RTAVFSADDR 0 9
regGRTAVFS_SE_RTAVFS_REG_CTRL = 0x4b44
# 	SET_WR_EN 0 0
# 	SET_RD_EN 1 1
regGRTAVFS_SE_RTAVFS_REG_STATUS = 0x4b45
# 	RTAVFS_WR_ACK 0 0
# 	RTAVFS_RD_DATA_VALID 1 1
regGRTAVFS_SE_RTAVFS_WR_DATA = 0x4b41
# 	RTAVFSDATA 0 31
regGRTAVFS_SE_SOFT_RESET = 0x4b4c
# 	RESETN_OVERRIDE 0 0
# 	RESERVED 1 31
regGRTAVFS_SE_TARG_FREQ = 0x4b46
# 	TARGET_FREQUENCY 0 15
# 	REQUEST 16 16
# 	RESERVED 17 31
regGRTAVFS_SE_TARG_VOLT = 0x4b47
# 	TARGET_VOLTAGE 0 9
# 	VALID 10 10
# 	RESERVED 11 31
regGRTAVFS_SOFT_RESET = 0x4b0c
# 	RESETN_OVERRIDE 0 0
# 	RESERVED 1 31
regGRTAVFS_TARG_FREQ = 0x4b06
# 	TARGET_FREQUENCY 0 15
# 	REQUEST 16 16
# 	RESERVED 17 31
regGRTAVFS_TARG_VOLT = 0x4b07
# 	TARGET_VOLTAGE 0 9
# 	VALID 10 10
# 	RESERVED 11 31
regGUS_DRAM_COMBINE_FLUSH = 0x2c1e
# 	GROUP0_TIMER 0 3
# 	GROUP1_TIMER 4 7
# 	GROUP2_TIMER 8 11
# 	GROUP3_TIMER 12 15
# 	GROUP4_TIMER 16 19
# 	GROUP5_TIMER 20 23
regGUS_DRAM_COMBINE_RD_WR_EN = 0x2c1f
# 	GROUP0_TIMER 0 1
# 	GROUP1_TIMER 2 3
# 	GROUP2_TIMER 4 5
# 	GROUP3_TIMER 6 7
# 	GROUP4_TIMER 8 9
# 	GROUP5_TIMER 10 11
regGUS_DRAM_GROUP_BURST = 0x2c31
# 	DRAM_LIMIT_LO 0 7
# 	DRAM_LIMIT_HI 8 15
regGUS_DRAM_PRI_AGE_COEFF = 0x2c21
# 	GROUP0_AGE_COEFFICIENT 0 2
# 	GROUP1_AGE_COEFFICIENT 3 5
# 	GROUP2_AGE_COEFFICIENT 6 8
# 	GROUP3_AGE_COEFFICIENT 9 11
# 	GROUP4_AGE_COEFFICIENT 12 14
# 	GROUP5_AGE_COEFFICIENT 15 17
regGUS_DRAM_PRI_AGE_RATE = 0x2c20
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP4_AGING_RATE 12 14
# 	GROUP5_AGING_RATE 15 17
regGUS_DRAM_PRI_FIXED = 0x2c23
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
# 	GROUP4_FIXED_COEFFICIENT 12 14
# 	GROUP5_FIXED_COEFFICIENT 15 17
regGUS_DRAM_PRI_QUANT1_PRI1 = 0x2c2b
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_DRAM_PRI_QUANT1_PRI2 = 0x2c2c
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_DRAM_PRI_QUANT1_PRI3 = 0x2c2d
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_DRAM_PRI_QUANT1_PRI4 = 0x2c2e
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_DRAM_PRI_QUANT1_PRI5 = 0x2c2f
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_DRAM_PRI_QUANT_PRI1 = 0x2c26
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_DRAM_PRI_QUANT_PRI2 = 0x2c27
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_DRAM_PRI_QUANT_PRI3 = 0x2c28
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_DRAM_PRI_QUANT_PRI4 = 0x2c29
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_DRAM_PRI_QUANT_PRI5 = 0x2c2a
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_DRAM_PRI_QUEUING = 0x2c22
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
# 	GROUP4_QUEUING_COEFFICIENT 12 14
# 	GROUP5_QUEUING_COEFFICIENT 15 17
regGUS_DRAM_PRI_URGENCY_COEFF = 0x2c24
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP4_URGENCY_COEFFICIENT 12 14
# 	GROUP5_URGENCY_COEFFICIENT 15 17
regGUS_DRAM_PRI_URGENCY_MODE = 0x2c25
# 	GROUP0_URGENCY_MODE 0 0
# 	GROUP1_URGENCY_MODE 1 1
# 	GROUP2_URGENCY_MODE 2 2
# 	GROUP3_URGENCY_MODE 3 3
# 	GROUP4_URGENCY_MODE 4 4
# 	GROUP5_URGENCY_MODE 5 5
regGUS_ERR_STATUS = 0x2c3e
# 	SDP_RDRSP_STATUS 0 3
# 	SDP_WRRSP_STATUS 4 7
# 	SDP_RDRSP_DATASTATUS 8 9
# 	SDP_RDRSP_DATAPARITY_ERROR 10 10
# 	CLEAR_ERROR_STATUS 11 11
# 	BUSY_ON_ERROR 12 12
# 	FUE_FLAG 13 13
regGUS_ICG_CTRL = 0x50f4
# 	SOFT_OVERRIDE_DRAM 0 0
# 	SOFT_OVERRIDE_WRITE 1 1
# 	SOFT_OVERRIDE_READ 2 2
# 	SOFT_OVERRIDE_RETURN_DEMUX 3 3
# 	SOFT_OVERRIDE_RETURN_WRITE 4 4
# 	SOFT_OVERRIDE_RETURN_READ 5 5
# 	SOFT_OVERRIDE_REGISTER 6 6
# 	SOFT_OVERRIDE_PERFMON 7 7
# 	SOFT_OVERRIDE_STATIC 8 8
# 	SPARE1 9 17
regGUS_IO_GROUP_BURST = 0x2c30
# 	RD_LIMIT_LO 0 7
# 	RD_LIMIT_HI 8 15
# 	WR_LIMIT_LO 16 23
# 	WR_LIMIT_HI 24 31
regGUS_IO_RD_COMBINE_FLUSH = 0x2c00
# 	GROUP0_TIMER 0 3
# 	GROUP1_TIMER 4 7
# 	GROUP2_TIMER 8 11
# 	GROUP3_TIMER 12 15
# 	GROUP4_TIMER 16 19
# 	GROUP5_TIMER 20 23
# 	COMB_MODE 24 25
regGUS_IO_RD_PRI_AGE_COEFF = 0x2c04
# 	GROUP0_AGE_COEFFICIENT 0 2
# 	GROUP1_AGE_COEFFICIENT 3 5
# 	GROUP2_AGE_COEFFICIENT 6 8
# 	GROUP3_AGE_COEFFICIENT 9 11
# 	GROUP4_AGE_COEFFICIENT 12 14
# 	GROUP5_AGE_COEFFICIENT 15 17
regGUS_IO_RD_PRI_AGE_RATE = 0x2c02
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP4_AGING_RATE 12 14
# 	GROUP5_AGING_RATE 15 17
regGUS_IO_RD_PRI_FIXED = 0x2c08
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
# 	GROUP4_FIXED_COEFFICIENT 12 14
# 	GROUP5_FIXED_COEFFICIENT 15 17
regGUS_IO_RD_PRI_QUANT1_PRI1 = 0x2c16
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_RD_PRI_QUANT1_PRI2 = 0x2c17
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_RD_PRI_QUANT1_PRI3 = 0x2c18
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_RD_PRI_QUANT1_PRI4 = 0x2c19
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_RD_PRI_QUANT_PRI1 = 0x2c0e
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_RD_PRI_QUANT_PRI2 = 0x2c0f
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_RD_PRI_QUANT_PRI3 = 0x2c10
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_RD_PRI_QUANT_PRI4 = 0x2c11
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_RD_PRI_QUEUING = 0x2c06
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
# 	GROUP4_QUEUING_COEFFICIENT 12 14
# 	GROUP5_QUEUING_COEFFICIENT 15 17
regGUS_IO_RD_PRI_URGENCY_COEFF = 0x2c0a
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP4_URGENCY_COEFFICIENT 12 14
# 	GROUP5_URGENCY_COEFFICIENT 15 17
regGUS_IO_RD_PRI_URGENCY_MODE = 0x2c0c
# 	GROUP0_URGENCY_MODE 0 0
# 	GROUP1_URGENCY_MODE 1 1
# 	GROUP2_URGENCY_MODE 2 2
# 	GROUP3_URGENCY_MODE 3 3
# 	GROUP4_URGENCY_MODE 4 4
# 	GROUP5_URGENCY_MODE 5 5
regGUS_IO_WR_COMBINE_FLUSH = 0x2c01
# 	GROUP0_TIMER 0 3
# 	GROUP1_TIMER 4 7
# 	GROUP2_TIMER 8 11
# 	GROUP3_TIMER 12 15
# 	GROUP4_TIMER 16 19
# 	GROUP5_TIMER 20 23
# 	COMB_MODE 24 25
regGUS_IO_WR_PRI_AGE_COEFF = 0x2c05
# 	GROUP0_AGE_COEFFICIENT 0 2
# 	GROUP1_AGE_COEFFICIENT 3 5
# 	GROUP2_AGE_COEFFICIENT 6 8
# 	GROUP3_AGE_COEFFICIENT 9 11
# 	GROUP4_AGE_COEFFICIENT 12 14
# 	GROUP5_AGE_COEFFICIENT 15 17
regGUS_IO_WR_PRI_AGE_RATE = 0x2c03
# 	GROUP0_AGING_RATE 0 2
# 	GROUP1_AGING_RATE 3 5
# 	GROUP2_AGING_RATE 6 8
# 	GROUP3_AGING_RATE 9 11
# 	GROUP4_AGING_RATE 12 14
# 	GROUP5_AGING_RATE 15 17
regGUS_IO_WR_PRI_FIXED = 0x2c09
# 	GROUP0_FIXED_COEFFICIENT 0 2
# 	GROUP1_FIXED_COEFFICIENT 3 5
# 	GROUP2_FIXED_COEFFICIENT 6 8
# 	GROUP3_FIXED_COEFFICIENT 9 11
# 	GROUP4_FIXED_COEFFICIENT 12 14
# 	GROUP5_FIXED_COEFFICIENT 15 17
regGUS_IO_WR_PRI_QUANT1_PRI1 = 0x2c1a
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_WR_PRI_QUANT1_PRI2 = 0x2c1b
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_WR_PRI_QUANT1_PRI3 = 0x2c1c
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_WR_PRI_QUANT1_PRI4 = 0x2c1d
# 	GROUP4_THRESHOLD 0 7
# 	GROUP5_THRESHOLD 8 15
regGUS_IO_WR_PRI_QUANT_PRI1 = 0x2c12
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_WR_PRI_QUANT_PRI2 = 0x2c13
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_WR_PRI_QUANT_PRI3 = 0x2c14
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_WR_PRI_QUANT_PRI4 = 0x2c15
# 	GROUP0_THRESHOLD 0 7
# 	GROUP1_THRESHOLD 8 15
# 	GROUP2_THRESHOLD 16 23
# 	GROUP3_THRESHOLD 24 31
regGUS_IO_WR_PRI_QUEUING = 0x2c07
# 	GROUP0_QUEUING_COEFFICIENT 0 2
# 	GROUP1_QUEUING_COEFFICIENT 3 5
# 	GROUP2_QUEUING_COEFFICIENT 6 8
# 	GROUP3_QUEUING_COEFFICIENT 9 11
# 	GROUP4_QUEUING_COEFFICIENT 12 14
# 	GROUP5_QUEUING_COEFFICIENT 15 17
regGUS_IO_WR_PRI_URGENCY_COEFF = 0x2c0b
# 	GROUP0_URGENCY_COEFFICIENT 0 2
# 	GROUP1_URGENCY_COEFFICIENT 3 5
# 	GROUP2_URGENCY_COEFFICIENT 6 8
# 	GROUP3_URGENCY_COEFFICIENT 9 11
# 	GROUP4_URGENCY_COEFFICIENT 12 14
# 	GROUP5_URGENCY_COEFFICIENT 15 17
regGUS_IO_WR_PRI_URGENCY_MODE = 0x2c0d
# 	GROUP0_URGENCY_MODE 0 0
# 	GROUP1_URGENCY_MODE 1 1
# 	GROUP2_URGENCY_MODE 2 2
# 	GROUP3_URGENCY_MODE 3 3
# 	GROUP4_URGENCY_MODE 4 4
# 	GROUP5_URGENCY_MODE 5 5
regGUS_L1_CH0_CMD_IN = 0x2c46
# 	COUNT 0 31
regGUS_L1_CH0_CMD_OUT = 0x2c47
# 	COUNT 0 31
regGUS_L1_CH0_DATA_IN = 0x2c48
# 	COUNT 0 31
regGUS_L1_CH0_DATA_OUT = 0x2c49
# 	COUNT 0 31
regGUS_L1_CH0_DATA_U_IN = 0x2c4a
# 	COUNT 0 31
regGUS_L1_CH0_DATA_U_OUT = 0x2c4b
# 	COUNT 0 31
regGUS_L1_CH1_CMD_IN = 0x2c4c
# 	COUNT 0 31
regGUS_L1_CH1_CMD_OUT = 0x2c4d
# 	COUNT 0 31
regGUS_L1_CH1_DATA_IN = 0x2c4e
# 	COUNT 0 31
regGUS_L1_CH1_DATA_OUT = 0x2c4f
# 	COUNT 0 31
regGUS_L1_CH1_DATA_U_IN = 0x2c50
# 	COUNT 0 31
regGUS_L1_CH1_DATA_U_OUT = 0x2c51
# 	COUNT 0 31
regGUS_L1_SA0_CMD_IN = 0x2c52
# 	COUNT 0 31
regGUS_L1_SA0_CMD_OUT = 0x2c53
# 	COUNT 0 31
regGUS_L1_SA0_DATA_IN = 0x2c54
# 	COUNT 0 31
regGUS_L1_SA0_DATA_OUT = 0x2c55
# 	COUNT 0 31
regGUS_L1_SA0_DATA_U_IN = 0x2c56
# 	COUNT 0 31
regGUS_L1_SA0_DATA_U_OUT = 0x2c57
# 	COUNT 0 31
regGUS_L1_SA1_CMD_IN = 0x2c58
# 	COUNT 0 31
regGUS_L1_SA1_CMD_OUT = 0x2c59
# 	COUNT 0 31
regGUS_L1_SA1_DATA_IN = 0x2c5a
# 	COUNT 0 31
regGUS_L1_SA1_DATA_OUT = 0x2c5b
# 	COUNT 0 31
regGUS_L1_SA1_DATA_U_IN = 0x2c5c
# 	COUNT 0 31
regGUS_L1_SA1_DATA_U_OUT = 0x2c5d
# 	COUNT 0 31
regGUS_L1_SA2_CMD_IN = 0x2c5e
# 	COUNT 0 31
regGUS_L1_SA2_CMD_OUT = 0x2c5f
# 	COUNT 0 31
regGUS_L1_SA2_DATA_IN = 0x2c60
# 	COUNT 0 31
regGUS_L1_SA2_DATA_OUT = 0x2c61
# 	COUNT 0 31
regGUS_L1_SA2_DATA_U_IN = 0x2c62
# 	COUNT 0 31
regGUS_L1_SA2_DATA_U_OUT = 0x2c63
# 	COUNT 0 31
regGUS_L1_SA3_CMD_IN = 0x2c64
# 	COUNT 0 31
regGUS_L1_SA3_CMD_OUT = 0x2c65
# 	COUNT 0 31
regGUS_L1_SA3_DATA_IN = 0x2c66
# 	COUNT 0 31
regGUS_L1_SA3_DATA_OUT = 0x2c67
# 	COUNT 0 31
regGUS_L1_SA3_DATA_U_IN = 0x2c68
# 	COUNT 0 31
regGUS_L1_SA3_DATA_U_OUT = 0x2c69
# 	COUNT 0 31
regGUS_LATENCY_SAMPLING = 0x2c3d
# 	SAMPLER0_DRAM 0 0
# 	SAMPLER1_DRAM 1 1
# 	SAMPLER0_IO 2 2
# 	SAMPLER1_IO 3 3
# 	SAMPLER0_READ 4 4
# 	SAMPLER1_READ 5 5
# 	SAMPLER0_WRITE 6 6
# 	SAMPLER1_WRITE 7 7
# 	SAMPLER0_ATOMIC_RET 8 8
# 	SAMPLER1_ATOMIC_RET 9 9
# 	SAMPLER0_ATOMIC_NORET 10 10
# 	SAMPLER1_ATOMIC_NORET 11 11
# 	SAMPLER0_VC 12 19
# 	SAMPLER1_VC 20 27
regGUS_MISC = 0x2c3c
# 	RELATIVE_PRI_IN_DRAM_ARB 0 0
# 	RELATIVE_PRI_IN_IO_RD_ARB 1 1
# 	RELATIVE_PRI_IN_IO_WR_ARB 2 2
# 	EARLY_SDP_ORIGDATA 3 3
# 	LINKMGR_DYNAMIC_MODE 4 5
# 	LINKMGR_HALT_THRESHOLD 6 7
# 	LINKMGR_RECONNECT_DELAY 8 9
# 	LINKMGR_IDLE_THRESHOLD 10 14
# 	SEND0_IOWR_ONLY 15 15
regGUS_MISC2 = 0x2c3f
# 	IO_RDWR_PRIORITY_ENABLE 0 0
# 	CH_L1_RO_MASK 1 1
# 	SA0_L1_RO_MASK 2 2
# 	SA1_L1_RO_MASK 3 3
# 	SA2_L1_RO_MASK 4 4
# 	SA3_L1_RO_MASK 5 5
# 	CH_L1_PERF_MASK 6 6
# 	SA0_L1_PERF_MASK 7 7
# 	SA1_L1_PERF_MASK 8 8
# 	SA2_L1_PERF_MASK 9 9
# 	SA3_L1_PERF_MASK 10 10
# 	FP_ATOMICS_ENABLE 11 11
# 	L1_RET_CLKEN 12 12
# 	FGCLKEN_HIGH 13 13
# 	BLOCK_REQUESTS 14 14
# 	REQUESTS_BLOCKED 15 15
# 	RIO_ICG_L1_ROUTER_BUSY_MASK 16 16
# 	WIO_ICG_L1_ROUTER_BUSY_MASK 17 17
# 	DRAM_ICG_L1_ROUTER_BUSY_MASK 18 18
regGUS_MISC3 = 0x2c6a
# 	FP_ATOMICS_LOG 0 0
# 	CLEAR_LOG 1 1
regGUS_PERFCOUNTER0_CFG = 0x3e03
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGUS_PERFCOUNTER1_CFG = 0x3e04
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regGUS_PERFCOUNTER2_HI = 0x3641
# 	PERFCOUNTER_HI 0 31
regGUS_PERFCOUNTER2_LO = 0x3640
# 	PERFCOUNTER_LO 0 31
regGUS_PERFCOUNTER2_MODE = 0x3e02
# 	COMPARE_MODE0 0 1
# 	COMPARE_MODE1 2 3
# 	COMPARE_MODE2 4 5
# 	COMPARE_MODE3 6 7
# 	COMPARE_VALUE0 8 11
# 	COMPARE_VALUE1 12 15
# 	COMPARE_VALUE2 16 19
# 	COMPARE_VALUE3 20 23
regGUS_PERFCOUNTER2_SELECT = 0x3e00
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regGUS_PERFCOUNTER2_SELECT1 = 0x3e01
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regGUS_PERFCOUNTER_HI = 0x3643
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regGUS_PERFCOUNTER_LO = 0x3642
# 	COUNTER_LO 0 31
regGUS_PERFCOUNTER_RSLT_CNTL = 0x3e05
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regGUS_SDP_ARB_FINAL = 0x2c32
# 	HI_DRAM_BURST_LIMIT 0 4
# 	DRAM_BURST_LIMIT 5 9
# 	IO_BURST_LIMIT 10 14
# 	BURST_LIMIT_MULTIPLIER 15 16
# 	ERREVENT_ON_ERROR 17 17
# 	HALTREQ_ON_ERROR 18 18
regGUS_SDP_CREDITS = 0x2c34
# 	TAG_LIMIT 0 7
# 	WR_RESP_CREDITS 8 14
# 	RD_RESP_CREDITS 16 22
regGUS_SDP_ENABLE = 0x2c45
# 	ENABLE 0 0
regGUS_SDP_QOS_VC_PRIORITY = 0x2c33
# 	VC2_IORD 0 3
# 	VC3_IOWR 4 7
# 	VC4_DRAM 8 11
# 	VC4_HI_DRAM 12 15
regGUS_SDP_REQ_CNTL = 0x2c3b
# 	REQ_PASS_PW_OVERRIDE_READ 0 0
# 	REQ_PASS_PW_OVERRIDE_WRITE 1 1
# 	REQ_PASS_PW_OVERRIDE_ATOMIC 2 2
# 	REQ_CHAIN_OVERRIDE_DRAM 3 3
# 	INNER_DOMAIN_MODE 4 4
regGUS_SDP_TAG_RESERVE0 = 0x2c35
# 	VC0 0 7
# 	VC1 8 15
# 	VC2 16 23
# 	VC3 24 31
regGUS_SDP_TAG_RESERVE1 = 0x2c36
# 	VC4 0 7
# 	VC5 8 15
# 	VC6 16 23
# 	VC7 24 31
regGUS_SDP_VCC_RESERVE0 = 0x2c37
# 	VC0_CREDITS 0 5
# 	VC1_CREDITS 6 11
# 	VC2_CREDITS 12 17
# 	VC3_CREDITS 18 23
# 	VC4_CREDITS 24 29
regGUS_SDP_VCC_RESERVE1 = 0x2c38
# 	VC5_CREDITS 0 5
# 	VC6_CREDITS 6 11
# 	VC7_CREDITS 12 17
# 	DISTRIBUTE_POOL 31 31
regGUS_SDP_VCD_RESERVE0 = 0x2c39
# 	VC0_CREDITS 0 5
# 	VC1_CREDITS 6 11
# 	VC2_CREDITS 12 17
# 	VC3_CREDITS 18 23
# 	VC4_CREDITS 24 29
regGUS_SDP_VCD_RESERVE1 = 0x2c3a
# 	VC5_CREDITS 0 5
# 	VC6_CREDITS 6 11
# 	VC7_CREDITS 12 17
# 	DISTRIBUTE_POOL 31 31
regGUS_WRRSP_FIFO_CNTL = 0x2c6b
# 	THRESHOLD 0 5
regIA_ENHANCE = 0x29c
# 	MISC 0 31
regIA_UTCL1_CNTL = 0xfe6
# 	XNACK_REDO_TIMER_CNT 0 19
# 	VMID_RESET_MODE 23 23
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	MTYPE_OVERRIDE 29 29
# 	LLC_NOALLOC_OVERRIDE 30 30
regIA_UTCL1_STATUS = 0xfe7
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29
regIA_UTCL1_STATUS_2 = 0xfd7
# 	IA_BUSY 0 0
# 	IA_DMA_BUSY 1 1
# 	IA_DMA_REQ_BUSY 2 2
# 	IA_GRP_BUSY 3 3
# 	IA_ADC_BUSY 4 4
# 	FAULT_DETECTED 5 5
# 	RETRY_DETECTED 6 6
# 	PRT_DETECTED 7 7
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29
regICG_CHA_CTRL = 0x50f1
# 	REG_CLK_OVERRIDE 0 0
# 	REQ_CLI_CLK_OVERRIDE 1 1
# 	REQ_ARB_CLK_OVERRIDE 2 2
# 	RET_CLK_OVERRIDE 3 3
# 	REQ_CREDIT_CLK_OVERRIDE 4 4
# 	PERFMON_CLK_OVERRIDE 5 5
regICG_CHCG_CLK_CTRL = 0x5144
# 	GLOBAL_CLK_OVERRIDE 0 0
# 	GLOBAL_NONHARVESTABLE_CLK_OVERRIDE 1 1
# 	REQUEST_CLK_OVERRIDE 2 2
# 	SRC_DATA_CLK_OVERRIDE 3 3
# 	RETURN_CLK_OVERRIDE 4 4
# 	GRBM_CLK_OVERRIDE 5 5
# 	PERF_CLK_OVERRIDE 6 6
regICG_CHC_CLK_CTRL = 0x5140
# 	GLOBAL_CLK_OVERRIDE 0 0
# 	GLOBAL_NONHARVESTABLE_CLK_OVERRIDE 1 1
# 	REQUEST_CLK_OVERRIDE 2 2
# 	SRC_DATA_CLK_OVERRIDE 3 3
# 	RETURN_CLK_OVERRIDE 4 4
# 	GRBM_CLK_OVERRIDE 5 5
# 	PERF_CLK_OVERRIDE 6 6
regICG_GL1A_CTRL = 0x50f0
# 	REG_CLK_OVERRIDE 0 0
# 	REQ_CLI_CLK_OVERRIDE 1 1
# 	REQ_ARB_CLK_OVERRIDE 2 2
# 	RET_CLK_OVERRIDE 3 3
# 	REQ_CREDIT_CLK_OVERRIDE 4 4
# 	PERFMON_CLK_OVERRIDE 5 5
regICG_GL1C_CLK_CTRL = 0x50ec
# 	GLOBAL_CLK_OVERRIDE 0 0
# 	GLOBAL_NONHARVESTABLE_CLK_OVERRIDE 1 1
# 	REQUEST_CLK_OVERRIDE 2 2
# 	VM_CLK_OVERRIDE 3 3
# 	TAG_CLK_OVERRIDE 4 4
# 	GCR_CLK_OVERRIDE 5 5
# 	SRC_DATA_CLK_OVERRIDE 6 6
# 	RETURN_CLK_OVERRIDE 7 7
# 	GRBM_CLK_OVERRIDE 8 8
# 	PERF_CLK_OVERRIDE 9 9
# 	LATENCY_FIFO_CLK_OVERRIDE 10 10
regICG_LDS_CLK_CTRL = 0x5114
# 	LDS_DLOAD0_OVERRIDE 0 0
# 	LDS_DLOAD1_OVERRIDE 1 1
# 	LDS_WGP_ARB_OVERRIDE 2 2
# 	LDS_TD_OVERRIDE 3 3
# 	LDS_ATTR_WR_OVERRIDE 4 4
# 	LDS_CONFIG_REG_OVERRIDE 5 5
# 	LDS_IDX_PIPE_OVERRIDE 6 6
# 	LDS_IDX_DIR_OVERRIDE 7 7
# 	LDS_IDX_WR_OVERRIDE 8 8
# 	LDS_IDX_INPUT_QUEUE_OVERRIDE 9 9
# 	LDS_MEM_OVERRIDE 10 10
# 	LDS_IDX_OUTPUT_ALIGNER_OVERRIDE 11 11
# 	LDS_DIR_OUTPUT_ALIGNER_OVERRIDE 12 12
# 	LDS_IDX_BANK_CONFLICT_OVERRIDE 13 13
# 	LDS_IDX_SCHED_INPUT_OVERRIDE 14 14
# 	LDS_IDX_SCHED_OUTPUT_OVERRIDE 15 15
# 	LDS_IDX_SCHED_PIPE_OVERRIDE 16 16
# 	LDS_IDX_SCHEDULER_OVERRIDE 17 17
# 	LDS_IDX_RDRTN_OVERRIDE 18 18
# 	LDS_SP_DONE_OVERRIDE 19 19
# 	LDS_SQC_PERF_OVERRIDE 20 20
# 	LDS_SP_READ_OVERRIDE 21 21
# 	SQ_LDS_VMEMCMD_OVERRIDE 22 22
# 	SP_LDS_VMEMREQ_OVERRIDE 23 23
# 	SPI_LDS_STALL_OVERRIDE 24 24
# 	MEM_WR_OVERRIDE 25 25
# 	LDS_CLK_OVERRIDE_UNUSED 26 31
regICG_SP_CLK_CTRL = 0x5093
# 	CLK_OVERRIDE 0 31
regLDS_CONFIG = 0x10a2
# 	ADDR_OUT_OF_RANGE_REPORTING 0 0
# 	CONF_BIT_1 1 1
# 	WAVE32_INTERP_DUAL_ISSUE_DISABLE 2 2
# 	SP_TDDATA_FGCG_OVERRIDE 3 3
# 	SQC_PERF_FGCG_OVERRIDE 4 4
# 	CONF_BIT_5 5 5
# 	CONF_BIT_6 6 6
# 	CONF_BIT_7 7 7
# 	CONF_BIT_8 8 8
regPA_CL_CLIP_CNTL = 0x204
# 	UCP_ENA_0 0 0
# 	UCP_ENA_1 1 1
# 	UCP_ENA_2 2 2
# 	UCP_ENA_3 3 3
# 	UCP_ENA_4 4 4
# 	UCP_ENA_5 5 5
# 	PS_UCP_Y_SCALE_NEG 13 13
# 	PS_UCP_MODE 14 15
# 	CLIP_DISABLE 16 16
# 	UCP_CULL_ONLY_ENA 17 17
# 	BOUNDARY_EDGE_FLAG_ENA 18 18
# 	DX_CLIP_SPACE_DEF 19 19
# 	DIS_CLIP_ERR_DETECT 20 20
# 	VTX_KILL_OR 21 21
# 	DX_RASTERIZATION_KILL 22 22
# 	DX_LINEAR_ATTR_CLIP_ENA 24 24
# 	VTE_VPORT_PROVOKE_DISABLE 25 25
# 	ZCLIP_NEAR_DISABLE 26 26
# 	ZCLIP_FAR_DISABLE 27 27
# 	ZCLIP_PROG_NEAR_ENA 28 28
regPA_CL_CNTL_STATUS = 0x1024
# 	CL_BUSY 31 31
regPA_CL_ENHANCE = 0x1025
# 	CLIP_VTX_REORDER_ENA 0 0
# 	NUM_CLIP_SEQ 1 2
# 	CLIPPED_PRIM_SEQ_STALL 3 3
# 	VE_NAN_PROC_DISABLE 4 4
# 	IGNORE_PIPELINE_RESET 6 6
# 	KILL_INNER_EDGE_FLAGS 7 7
# 	NGG_PA_TO_ALL_SC 8 8
# 	TC_LATENCY_TIME_STAMP_RESOLUTION 9 10
# 	NGG_BYPASS_PRIM_FILTER 11 11
# 	NGG_SIDEBAND_MEMORY_DEPTH 12 13
# 	NGG_PRIM_INDICES_FIFO_DEPTH 14 16
# 	PROG_NEAR_CLIP_PLANE_ENABLE 17 17
# 	POLY_INNER_EDGE_FLAG_DISABLE 18 18
# 	TC_REQUEST_PERF_CNTR_ENABLE 19 19
# 	DISABLE_PA_PH_INTF_FINE_CLOCK_GATE 20 20
# 	DISABLE_PA_SX_REQ_INTF_FINE_CLOCK_GATE 21 21
# 	ENABLE_PA_RATE_CNTL 22 22
# 	CLAMP_NEGATIVE_BB_TO_ZERO 23 23
# 	ECO_SPARE3 28 28
# 	ECO_SPARE2 29 29
# 	ECO_SPARE1 30 30
# 	ECO_SPARE0 31 31
regPA_CL_GB_HORZ_CLIP_ADJ = 0x2fc
# 	DATA_REGISTER 0 31
regPA_CL_GB_HORZ_DISC_ADJ = 0x2fd
# 	DATA_REGISTER 0 31
regPA_CL_GB_VERT_CLIP_ADJ = 0x2fa
# 	DATA_REGISTER 0 31
regPA_CL_GB_VERT_DISC_ADJ = 0x2fb
# 	DATA_REGISTER 0 31
regPA_CL_NANINF_CNTL = 0x208
# 	VTE_XY_INF_DISCARD 0 0
# 	VTE_Z_INF_DISCARD 1 1
# 	VTE_W_INF_DISCARD 2 2
# 	VTE_0XNANINF_IS_0 3 3
# 	VTE_XY_NAN_RETAIN 4 4
# 	VTE_Z_NAN_RETAIN 5 5
# 	VTE_W_NAN_RETAIN 6 6
# 	VTE_W_RECIP_NAN_IS_0 7 7
# 	VS_XY_NAN_TO_INF 8 8
# 	VS_XY_INF_RETAIN 9 9
# 	VS_Z_NAN_TO_INF 10 10
# 	VS_Z_INF_RETAIN 11 11
# 	VS_W_NAN_TO_INF 12 12
# 	VS_W_INF_RETAIN 13 13
# 	VS_CLIP_DIST_INF_DISCARD 14 14
# 	VTE_NO_OUTPUT_NEG_0 20 20
regPA_CL_NGG_CNTL = 0x20e
# 	VERTEX_REUSE_OFF 0 0
# 	INDEX_BUF_EDGE_FLAG_ENA 1 1
# 	VERTEX_REUSE_DEPTH 2 9
regPA_CL_POINT_CULL_RAD = 0x1f8
# 	DATA_REGISTER 0 31
regPA_CL_POINT_SIZE = 0x1f7
# 	DATA_REGISTER 0 31
regPA_CL_POINT_X_RAD = 0x1f5
# 	DATA_REGISTER 0 31
regPA_CL_POINT_Y_RAD = 0x1f6
# 	DATA_REGISTER 0 31
regPA_CL_PROG_NEAR_CLIP_Z = 0x187
# 	DATA_REGISTER 0 31
regPA_CL_UCP_0_W = 0x172
# 	DATA_REGISTER 0 31
regPA_CL_UCP_0_X = 0x16f
# 	DATA_REGISTER 0 31
regPA_CL_UCP_0_Y = 0x170
# 	DATA_REGISTER 0 31
regPA_CL_UCP_0_Z = 0x171
# 	DATA_REGISTER 0 31
regPA_CL_UCP_1_W = 0x176
# 	DATA_REGISTER 0 31
regPA_CL_UCP_1_X = 0x173
# 	DATA_REGISTER 0 31
regPA_CL_UCP_1_Y = 0x174
# 	DATA_REGISTER 0 31
regPA_CL_UCP_1_Z = 0x175
# 	DATA_REGISTER 0 31
regPA_CL_UCP_2_W = 0x17a
# 	DATA_REGISTER 0 31
regPA_CL_UCP_2_X = 0x177
# 	DATA_REGISTER 0 31
regPA_CL_UCP_2_Y = 0x178
# 	DATA_REGISTER 0 31
regPA_CL_UCP_2_Z = 0x179
# 	DATA_REGISTER 0 31
regPA_CL_UCP_3_W = 0x17e
# 	DATA_REGISTER 0 31
regPA_CL_UCP_3_X = 0x17b
# 	DATA_REGISTER 0 31
regPA_CL_UCP_3_Y = 0x17c
# 	DATA_REGISTER 0 31
regPA_CL_UCP_3_Z = 0x17d
# 	DATA_REGISTER 0 31
regPA_CL_UCP_4_W = 0x182
# 	DATA_REGISTER 0 31
regPA_CL_UCP_4_X = 0x17f
# 	DATA_REGISTER 0 31
regPA_CL_UCP_4_Y = 0x180
# 	DATA_REGISTER 0 31
regPA_CL_UCP_4_Z = 0x181
# 	DATA_REGISTER 0 31
regPA_CL_UCP_5_W = 0x186
# 	DATA_REGISTER 0 31
regPA_CL_UCP_5_X = 0x183
# 	DATA_REGISTER 0 31
regPA_CL_UCP_5_Y = 0x184
# 	DATA_REGISTER 0 31
regPA_CL_UCP_5_Z = 0x185
# 	DATA_REGISTER 0 31
regPA_CL_VPORT_XOFFSET = 0x110
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_1 = 0x116
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_10 = 0x14c
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_11 = 0x152
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_12 = 0x158
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_13 = 0x15e
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_14 = 0x164
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_15 = 0x16a
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_2 = 0x11c
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_3 = 0x122
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_4 = 0x128
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_5 = 0x12e
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_6 = 0x134
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_7 = 0x13a
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_8 = 0x140
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XOFFSET_9 = 0x146
# 	VPORT_XOFFSET 0 31
regPA_CL_VPORT_XSCALE = 0x10f
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_1 = 0x115
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_10 = 0x14b
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_11 = 0x151
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_12 = 0x157
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_13 = 0x15d
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_14 = 0x163
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_15 = 0x169
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_2 = 0x11b
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_3 = 0x121
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_4 = 0x127
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_5 = 0x12d
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_6 = 0x133
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_7 = 0x139
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_8 = 0x13f
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_XSCALE_9 = 0x145
# 	VPORT_XSCALE 0 31
regPA_CL_VPORT_YOFFSET = 0x112
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_1 = 0x118
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_10 = 0x14e
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_11 = 0x154
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_12 = 0x15a
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_13 = 0x160
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_14 = 0x166
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_15 = 0x16c
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_2 = 0x11e
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_3 = 0x124
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_4 = 0x12a
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_5 = 0x130
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_6 = 0x136
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_7 = 0x13c
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_8 = 0x142
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YOFFSET_9 = 0x148
# 	VPORT_YOFFSET 0 31
regPA_CL_VPORT_YSCALE = 0x111
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_1 = 0x117
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_10 = 0x14d
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_11 = 0x153
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_12 = 0x159
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_13 = 0x15f
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_14 = 0x165
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_15 = 0x16b
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_2 = 0x11d
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_3 = 0x123
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_4 = 0x129
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_5 = 0x12f
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_6 = 0x135
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_7 = 0x13b
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_8 = 0x141
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_YSCALE_9 = 0x147
# 	VPORT_YSCALE 0 31
regPA_CL_VPORT_ZOFFSET = 0x114
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_1 = 0x11a
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_10 = 0x150
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_11 = 0x156
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_12 = 0x15c
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_13 = 0x162
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_14 = 0x168
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_15 = 0x16e
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_2 = 0x120
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_3 = 0x126
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_4 = 0x12c
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_5 = 0x132
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_6 = 0x138
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_7 = 0x13e
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_8 = 0x144
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZOFFSET_9 = 0x14a
# 	VPORT_ZOFFSET 0 31
regPA_CL_VPORT_ZSCALE = 0x113
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_1 = 0x119
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_10 = 0x14f
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_11 = 0x155
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_12 = 0x15b
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_13 = 0x161
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_14 = 0x167
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_15 = 0x16d
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_2 = 0x11f
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_3 = 0x125
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_4 = 0x12b
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_5 = 0x131
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_6 = 0x137
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_7 = 0x13d
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_8 = 0x143
# 	VPORT_ZSCALE 0 31
regPA_CL_VPORT_ZSCALE_9 = 0x149
# 	VPORT_ZSCALE 0 31
regPA_CL_VRS_CNTL = 0x212
# 	VERTEX_RATE_COMBINER_MODE 0 2
# 	PRIMITIVE_RATE_COMBINER_MODE 3 5
# 	HTILE_RATE_COMBINER_MODE 6 8
# 	SAMPLE_ITER_COMBINER_MODE 9 11
# 	EXPOSE_VRS_PIXELS_MASK 13 13
# 	CMASK_RATE_HINT_FORCE_ZERO 14 14
regPA_CL_VS_OUT_CNTL = 0x207
# 	CLIP_DIST_ENA_0 0 0
# 	CLIP_DIST_ENA_1 1 1
# 	CLIP_DIST_ENA_2 2 2
# 	CLIP_DIST_ENA_3 3 3
# 	CLIP_DIST_ENA_4 4 4
# 	CLIP_DIST_ENA_5 5 5
# 	CLIP_DIST_ENA_6 6 6
# 	CLIP_DIST_ENA_7 7 7
# 	CULL_DIST_ENA_0 8 8
# 	CULL_DIST_ENA_1 9 9
# 	CULL_DIST_ENA_2 10 10
# 	CULL_DIST_ENA_3 11 11
# 	CULL_DIST_ENA_4 12 12
# 	CULL_DIST_ENA_5 13 13
# 	CULL_DIST_ENA_6 14 14
# 	CULL_DIST_ENA_7 15 15
# 	USE_VTX_POINT_SIZE 16 16
# 	USE_VTX_EDGE_FLAG 17 17
# 	USE_VTX_RENDER_TARGET_INDX 18 18
# 	USE_VTX_VIEWPORT_INDX 19 19
# 	USE_VTX_KILL_FLAG 20 20
# 	VS_OUT_MISC_VEC_ENA 21 21
# 	VS_OUT_CCDIST0_VEC_ENA 22 22
# 	VS_OUT_CCDIST1_VEC_ENA 23 23
# 	VS_OUT_MISC_SIDE_BUS_ENA 24 24
# 	USE_VTX_LINE_WIDTH 27 27
# 	USE_VTX_VRS_RATE 28 28
# 	BYPASS_VTX_RATE_COMBINER 29 29
# 	BYPASS_PRIM_RATE_COMBINER 30 30
regPA_CL_VTE_CNTL = 0x206
# 	VPORT_X_SCALE_ENA 0 0
# 	VPORT_X_OFFSET_ENA 1 1
# 	VPORT_Y_SCALE_ENA 2 2
# 	VPORT_Y_OFFSET_ENA 3 3
# 	VPORT_Z_SCALE_ENA 4 4
# 	VPORT_Z_OFFSET_ENA 5 5
# 	VTX_XY_FMT 8 8
# 	VTX_Z_FMT 9 9
# 	VTX_W0_FMT 10 10
# 	PERFCOUNTER_REF 11 11
regPA_PH_ENHANCE = 0x95f
# 	ECO_SPARE0 0 0
# 	ECO_SPARE1 1 1
# 	ECO_SPARE2 2 2
# 	ECO_SPARE3 3 3
# 	DISABLE_PH_SC_INTF_FINE_CLOCK_GATE 4 4
# 	DISABLE_FOPKT 5 5
# 	DISABLE_FOPKT_SCAN_POST_RESET 6 6
# 	DISABLE_PH_SC_INTF_CLKEN_CLOCK_GATE 7 7
# 	DISABLE_PH_PERF_REG_FGCG 9 9
# 	ENABLE_PH_INTF_CLKEN_STRETCH 10 12
# 	DISABLE_USE_LAST_PH_ARBITER_PERFCOUNTER_SAMPLE_EVENT 13 13
# 	USE_PERFCOUNTER_START_STOP_EVENTS 14 14
# 	FORCE_PH_PERFCOUNTER_SAMPLE_ENABLE_ON 15 15
# 	PH_SPI_GE_THROTTLE_MODE 16 16
# 	PH_SPI_GE_THROTTLE_MODE_DISABLE 17 17
# 	PH_SPI_GE_THROTTLE_PERFCOUNTER_COUNT_MODE 18 18
regPA_PH_INTERFACE_FIFO_SIZE = 0x95e
# 	PA_PH_IF_FIFO_SIZE 0 9
# 	PH_SC_IF_FIFO_SIZE 16 21
regPA_PH_PERFCOUNTER0_HI = 0x3581
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER0_LO = 0x3580
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER0_SELECT = 0x3d80
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_PH_PERFCOUNTER0_SELECT1 = 0x3d81
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_PH_PERFCOUNTER1_HI = 0x3583
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER1_LO = 0x3582
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER1_SELECT = 0x3d82
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_PH_PERFCOUNTER1_SELECT1 = 0x3d90
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_PH_PERFCOUNTER2_HI = 0x3585
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER2_LO = 0x3584
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER2_SELECT = 0x3d83
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_PH_PERFCOUNTER2_SELECT1 = 0x3d91
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_PH_PERFCOUNTER3_HI = 0x3587
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER3_LO = 0x3586
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER3_SELECT = 0x3d84
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_PH_PERFCOUNTER3_SELECT1 = 0x3d92
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_PH_PERFCOUNTER4_HI = 0x3589
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER4_LO = 0x3588
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER4_SELECT = 0x3d85
# 	PERF_SEL 0 9
regPA_PH_PERFCOUNTER5_HI = 0x358b
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER5_LO = 0x358a
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER5_SELECT = 0x3d86
# 	PERF_SEL 0 9
regPA_PH_PERFCOUNTER6_HI = 0x358d
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER6_LO = 0x358c
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER6_SELECT = 0x3d87
# 	PERF_SEL 0 9
regPA_PH_PERFCOUNTER7_HI = 0x358f
# 	PERFCOUNTER_HI 0 31
regPA_PH_PERFCOUNTER7_LO = 0x358e
# 	PERFCOUNTER_LO 0 31
regPA_PH_PERFCOUNTER7_SELECT = 0x3d88
# 	PERF_SEL 0 9
regPA_RATE_CNTL = 0x188
# 	VERTEX_RATE 0 3
# 	PRIM_RATE 4 7
regPA_SC_AA_CONFIG = 0x2f8
# 	MSAA_NUM_SAMPLES 0 2
# 	AA_MASK_CENTROID_DTMN 4 4
# 	MAX_SAMPLE_DIST 13 16
# 	MSAA_EXPOSED_SAMPLES 20 22
# 	DETAIL_TO_EXPOSED_MODE 24 25
# 	COVERAGE_TO_SHADER_SELECT 26 27
# 	SAMPLE_COVERAGE_ENCODING 28 28
# 	COVERED_CENTROID_IS_CENTER 29 29
regPA_SC_AA_MASK_X0Y0_X1Y0 = 0x30e
# 	AA_MASK_X0Y0 0 15
# 	AA_MASK_X1Y0 16 31
regPA_SC_AA_MASK_X0Y1_X1Y1 = 0x30f
# 	AA_MASK_X0Y1 0 15
# 	AA_MASK_X1Y1 16 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0 = 0x2fe
# 	S0_X 0 3
# 	S0_Y 4 7
# 	S1_X 8 11
# 	S1_Y 12 15
# 	S2_X 16 19
# 	S2_Y 20 23
# 	S3_X 24 27
# 	S3_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_1 = 0x2ff
# 	S4_X 0 3
# 	S4_Y 4 7
# 	S5_X 8 11
# 	S5_Y 12 15
# 	S6_X 16 19
# 	S6_Y 20 23
# 	S7_X 24 27
# 	S7_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_2 = 0x300
# 	S8_X 0 3
# 	S8_Y 4 7
# 	S9_X 8 11
# 	S9_Y 12 15
# 	S10_X 16 19
# 	S10_Y 20 23
# 	S11_X 24 27
# 	S11_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_3 = 0x301
# 	S12_X 0 3
# 	S12_Y 4 7
# 	S13_X 8 11
# 	S13_Y 12 15
# 	S14_X 16 19
# 	S14_Y 20 23
# 	S15_X 24 27
# 	S15_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0 = 0x306
# 	S0_X 0 3
# 	S0_Y 4 7
# 	S1_X 8 11
# 	S1_Y 12 15
# 	S2_X 16 19
# 	S2_Y 20 23
# 	S3_X 24 27
# 	S3_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_1 = 0x307
# 	S4_X 0 3
# 	S4_Y 4 7
# 	S5_X 8 11
# 	S5_Y 12 15
# 	S6_X 16 19
# 	S6_Y 20 23
# 	S7_X 24 27
# 	S7_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_2 = 0x308
# 	S8_X 0 3
# 	S8_Y 4 7
# 	S9_X 8 11
# 	S9_Y 12 15
# 	S10_X 16 19
# 	S10_Y 20 23
# 	S11_X 24 27
# 	S11_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_3 = 0x309
# 	S12_X 0 3
# 	S12_Y 4 7
# 	S13_X 8 11
# 	S13_Y 12 15
# 	S14_X 16 19
# 	S14_Y 20 23
# 	S15_X 24 27
# 	S15_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0 = 0x302
# 	S0_X 0 3
# 	S0_Y 4 7
# 	S1_X 8 11
# 	S1_Y 12 15
# 	S2_X 16 19
# 	S2_Y 20 23
# 	S3_X 24 27
# 	S3_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_1 = 0x303
# 	S4_X 0 3
# 	S4_Y 4 7
# 	S5_X 8 11
# 	S5_Y 12 15
# 	S6_X 16 19
# 	S6_Y 20 23
# 	S7_X 24 27
# 	S7_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_2 = 0x304
# 	S8_X 0 3
# 	S8_Y 4 7
# 	S9_X 8 11
# 	S9_Y 12 15
# 	S10_X 16 19
# 	S10_Y 20 23
# 	S11_X 24 27
# 	S11_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_3 = 0x305
# 	S12_X 0 3
# 	S12_Y 4 7
# 	S13_X 8 11
# 	S13_Y 12 15
# 	S14_X 16 19
# 	S14_Y 20 23
# 	S15_X 24 27
# 	S15_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0 = 0x30a
# 	S0_X 0 3
# 	S0_Y 4 7
# 	S1_X 8 11
# 	S1_Y 12 15
# 	S2_X 16 19
# 	S2_Y 20 23
# 	S3_X 24 27
# 	S3_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_1 = 0x30b
# 	S4_X 0 3
# 	S4_Y 4 7
# 	S5_X 8 11
# 	S5_Y 12 15
# 	S6_X 16 19
# 	S6_Y 20 23
# 	S7_X 24 27
# 	S7_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_2 = 0x30c
# 	S8_X 0 3
# 	S8_Y 4 7
# 	S9_X 8 11
# 	S9_Y 12 15
# 	S10_X 16 19
# 	S10_Y 20 23
# 	S11_X 24 27
# 	S11_Y 28 31
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_3 = 0x30d
# 	S12_X 0 3
# 	S12_Y 4 7
# 	S13_X 8 11
# 	S13_Y 12 15
# 	S14_X 16 19
# 	S14_Y 20 23
# 	S15_X 24 27
# 	S15_Y 28 31
regPA_SC_ATM_CNTL = 0x94d
# 	SC_PC_IF_SIZE 0 5
# 	DISABLE_SC_PC_IF_FGCG_EN 7 7
# 	MAX_ATTRIBUTES_IN_WAVE 8 15
# 	DISABLE_MAX_ATTRIBUTES 16 16
# 	SELECT_MAX_ATTRIBUTES 17 17
regPA_SC_BINNER_CNTL_0 = 0x311
# 	BINNING_MODE 0 1
# 	BIN_SIZE_X 2 2
# 	BIN_SIZE_Y 3 3
# 	BIN_SIZE_X_EXTEND 4 6
# 	BIN_SIZE_Y_EXTEND 7 9
# 	CONTEXT_STATES_PER_BIN 10 12
# 	PERSISTENT_STATES_PER_BIN 13 17
# 	DISABLE_START_OF_PRIM 18 18
# 	FPOVS_PER_BATCH 19 26
# 	OPTIMAL_BIN_SELECTION 27 27
# 	FLUSH_ON_BINNING_TRANSITION 28 28
# 	BIN_MAPPING_MODE 29 30
regPA_SC_BINNER_CNTL_1 = 0x312
# 	MAX_ALLOC_COUNT 0 15
# 	MAX_PRIM_PER_BATCH 16 31
regPA_SC_BINNER_CNTL_2 = 0x315
# 	BIN_SIZE_X_MULT_BY_1P5X 0 0
# 	BIN_SIZE_Y_MULT_BY_1P5X 1 1
# 	ENABLE_LIGHT_VOLUME_RENDERING_OPTIMIZATION 2 2
# 	DUAL_LIGHT_SHAFT_IN_DRAW 3 3
# 	LIGHT_SHAFT_DRAW_CALL_LIMIT 4 6
# 	CONTEXT_DONE_EVENTS_PER_BIN 7 10
# 	ZPP_ENABLED 11 11
# 	ZPP_OPTIMIZATION_ENABLED 12 12
# 	ZPP_AREA_THRESHOLD 13 20
# 	DISABLE_NOPCEXPORT_BREAKBATCH_CONDITION 21 21
regPA_SC_BINNER_CNTL_OVERRIDE = 0x946
# 	BINNING_MODE 0 1
# 	CONTEXT_STATES_PER_BIN 10 12
# 	PERSISTENT_STATES_PER_BIN 13 17
# 	FPOVS_PER_BATCH 19 26
# 	DIRECT_OVERRIDE_MODE 27 27
# 	OVERRIDE 28 31
regPA_SC_BINNER_EVENT_CNTL_0 = 0x950
# 	RESERVED_0 0 1
# 	SAMPLE_STREAMOUTSTATS1 2 3
# 	SAMPLE_STREAMOUTSTATS2 4 5
# 	SAMPLE_STREAMOUTSTATS3 6 7
# 	CACHE_FLUSH_TS 8 9
# 	CONTEXT_DONE 10 11
# 	CACHE_FLUSH 12 13
# 	CS_PARTIAL_FLUSH 14 15
# 	VGT_STREAMOUT_SYNC 16 17
# 	RESERVED_9 18 19
# 	VGT_STREAMOUT_RESET 20 21
# 	END_OF_PIPE_INCR_DE 22 23
# 	END_OF_PIPE_IB_END 24 25
# 	RST_PIX_CNT 26 27
# 	BREAK_BATCH 28 29
# 	VS_PARTIAL_FLUSH 30 31
regPA_SC_BINNER_EVENT_CNTL_1 = 0x951
# 	PS_PARTIAL_FLUSH 0 1
# 	FLUSH_HS_OUTPUT 2 3
# 	FLUSH_DFSM 4 5
# 	RESET_TO_LOWEST_VGT 6 7
# 	CACHE_FLUSH_AND_INV_TS_EVENT 8 9
# 	WAIT_SYNC 10 11
# 	CACHE_FLUSH_AND_INV_EVENT 12 13
# 	PERFCOUNTER_START 14 15
# 	PERFCOUNTER_STOP 16 17
# 	PIPELINESTAT_START 18 19
# 	PIPELINESTAT_STOP 20 21
# 	PERFCOUNTER_SAMPLE 22 23
# 	FLUSH_ES_OUTPUT 24 25
# 	BIN_CONF_OVERRIDE_CHECK 26 27
# 	SAMPLE_PIPELINESTAT 28 29
# 	SO_VGTSTREAMOUT_FLUSH 30 31
regPA_SC_BINNER_EVENT_CNTL_2 = 0x952
# 	SAMPLE_STREAMOUTSTATS 0 1
# 	RESET_VTX_CNT 2 3
# 	BLOCK_CONTEXT_DONE 4 5
# 	RESERVED_35 6 7
# 	VGT_FLUSH 8 9
# 	TGID_ROLLOVER 10 11
# 	SQ_NON_EVENT 12 13
# 	SC_SEND_DB_VPZ 14 15
# 	BOTTOM_OF_PIPE_TS 16 17
# 	RESERVED_41 18 19
# 	DB_CACHE_FLUSH_AND_INV 20 21
# 	FLUSH_AND_INV_DB_DATA_TS 22 23
# 	FLUSH_AND_INV_DB_META 24 25
# 	FLUSH_AND_INV_CB_DATA_TS 26 27
# 	FLUSH_AND_INV_CB_META 28 29
# 	CS_DONE 30 31
regPA_SC_BINNER_EVENT_CNTL_3 = 0x953
# 	PS_DONE 0 1
# 	FLUSH_AND_INV_CB_PIXEL_DATA 2 3
# 	RESERVED_50 4 5
# 	THREAD_TRACE_START 6 7
# 	THREAD_TRACE_STOP 8 9
# 	THREAD_TRACE_MARKER 10 11
# 	THREAD_TRACE_DRAW 12 13
# 	THREAD_TRACE_FINISH 14 15
# 	PIXEL_PIPE_STAT_CONTROL 16 17
# 	PIXEL_PIPE_STAT_DUMP 18 19
# 	PIXEL_PIPE_STAT_RESET 20 21
# 	CONTEXT_SUSPEND 22 23
# 	OFFCHIP_HS_DEALLOC 24 25
# 	ENABLE_NGG_PIPELINE 26 27
# 	ENABLE_PIPELINE_NOT_USED 28 29
# 	DRAW_DONE 30 31
regPA_SC_BINNER_PERF_CNTL_0 = 0x955
# 	BIN_HIST_NUM_PRIMS_THRESHOLD 0 9
# 	BATCH_HIST_NUM_PRIMS_THRESHOLD 10 19
# 	BIN_HIST_NUM_CONTEXT_THRESHOLD 20 22
# 	BATCH_HIST_NUM_CONTEXT_THRESHOLD 23 25
regPA_SC_BINNER_PERF_CNTL_1 = 0x956
# 	BIN_HIST_NUM_PERSISTENT_STATE_THRESHOLD 0 4
# 	BATCH_HIST_NUM_PERSISTENT_STATE_THRESHOLD 5 9
# 	BATCH_HIST_NUM_TRIV_REJECTED_PRIMS_THRESHOLD 10 25
regPA_SC_BINNER_PERF_CNTL_2 = 0x957
# 	BATCH_HIST_NUM_ROWS_PER_PRIM_THRESHOLD 0 10
# 	BATCH_HIST_NUM_COLUMNS_PER_ROW_THRESHOLD 11 21
regPA_SC_BINNER_PERF_CNTL_3 = 0x958
# 	BATCH_HIST_NUM_PS_WAVE_BREAKS_THRESHOLD 0 31
regPA_SC_BINNER_TIMEOUT_COUNTER = 0x954
# 	THRESHOLD 0 31
regPA_SC_CENTROID_PRIORITY_0 = 0x2f5
# 	DISTANCE_0 0 3
# 	DISTANCE_1 4 7
# 	DISTANCE_2 8 11
# 	DISTANCE_3 12 15
# 	DISTANCE_4 16 19
# 	DISTANCE_5 20 23
# 	DISTANCE_6 24 27
# 	DISTANCE_7 28 31
regPA_SC_CENTROID_PRIORITY_1 = 0x2f6
# 	DISTANCE_8 0 3
# 	DISTANCE_9 4 7
# 	DISTANCE_10 8 11
# 	DISTANCE_11 12 15
# 	DISTANCE_12 16 19
# 	DISTANCE_13 20 23
# 	DISTANCE_14 24 27
# 	DISTANCE_15 28 31
regPA_SC_CLIPRECT_0_BR = 0x85
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_CLIPRECT_0_TL = 0x84
# 	TL_X 0 14
# 	TL_Y 16 30
regPA_SC_CLIPRECT_1_BR = 0x87
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_CLIPRECT_1_TL = 0x86
# 	TL_X 0 14
# 	TL_Y 16 30
regPA_SC_CLIPRECT_2_BR = 0x89
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_CLIPRECT_2_TL = 0x88
# 	TL_X 0 14
# 	TL_Y 16 30
regPA_SC_CLIPRECT_3_BR = 0x8b
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_CLIPRECT_3_TL = 0x8a
# 	TL_X 0 14
# 	TL_Y 16 30
regPA_SC_CLIPRECT_RULE = 0x83
# 	CLIP_RULE 0 15
regPA_SC_CONSERVATIVE_RASTERIZATION_CNTL = 0x313
# 	OVER_RAST_ENABLE 0 0
# 	OVER_RAST_SAMPLE_SELECT 1 4
# 	UNDER_RAST_ENABLE 5 5
# 	UNDER_RAST_SAMPLE_SELECT 6 9
# 	PBB_UNCERTAINTY_REGION_ENABLE 10 10
# 	ZMM_TRI_EXTENT 11 11
# 	ZMM_TRI_OFFSET 12 12
# 	OVERRIDE_OVER_RAST_INNER_TO_NORMAL 13 13
# 	OVERRIDE_UNDER_RAST_INNER_TO_NORMAL 14 14
# 	DEGENERATE_OVERRIDE_INNER_TO_NORMAL_DISABLE 15 15
# 	UNCERTAINTY_REGION_MODE 16 17
# 	OUTER_UNCERTAINTY_EDGERULE_OVERRIDE 18 18
# 	INNER_UNCERTAINTY_EDGERULE_OVERRIDE 19 19
# 	NULL_SQUAD_AA_MASK_ENABLE 20 20
# 	COVERAGE_AA_MASK_ENABLE 21 21
# 	PREZ_AA_MASK_ENABLE 22 22
# 	POSTZ_AA_MASK_ENABLE 23 23
# 	CENTROID_SAMPLE_OVERRIDE 24 24
# 	UNCERTAINTY_REGION_MULT 25 26
# 	UNCERTAINTY_REGION_PBB_MULT 27 28
regPA_SC_DSM_CNTL = 0x948
# 	FORCE_EOV_REZ_0 0 0
# 	FORCE_EOV_REZ_1 1 1
regPA_SC_EDGERULE = 0x8c
# 	ER_TRI 0 3
# 	ER_POINT 4 7
# 	ER_RECT 8 11
# 	ER_LINE_LR 12 17
# 	ER_LINE_RL 18 23
# 	ER_LINE_TB 24 27
# 	ER_LINE_BT 28 31
regPA_SC_ENHANCE = 0x941
# 	ENABLE_PA_SC_OUT_OF_ORDER 0 0
# 	DISABLE_SC_DB_TILE_FIX 1 1
# 	DISABLE_AA_MASK_FULL_FIX 2 2
# 	ENABLE_1XMSAA_SAMPLE_LOCATIONS 3 3
# 	ENABLE_1XMSAA_SAMPLE_LOC_CENTROID 4 4
# 	DISABLE_SCISSOR_FIX 5 5
# 	SEND_UNLIT_STILES_TO_PACKER 6 6
# 	DISABLE_DUALGRAD_PERF_OPTIMIZATION 7 7
# 	DISABLE_SC_PROCESS_RESET_PRIM 8 8
# 	DISABLE_SC_PROCESS_RESET_SUPERTILE 9 9
# 	DISABLE_SC_PROCESS_RESET_TILE 10 10
# 	DISABLE_PA_SC_GUIDANCE 11 11
# 	DISABLE_EOV_ALL_CTRL_ONLY_COMBINATIONS 12 12
# 	ENABLE_MULTICYCLE_BUBBLE_FREEZE 13 13
# 	DISABLE_OUT_OF_ORDER_PA_SC_GUIDANCE 14 14
# 	ENABLE_OUT_OF_ORDER_POLY_MODE 15 15
# 	DISABLE_OUT_OF_ORDER_EOP_SYNC_NULL_PRIMS_LAST 16 16
# 	DISABLE_OUT_OF_ORDER_THRESHOLD_SWITCHING 17 17
# 	ENABLE_OUT_OF_ORDER_THRESHOLD_SWITCH_AT_EOPG_ONLY 18 18
# 	DISABLE_OUT_OF_ORDER_DESIRED_FIFO_EMPTY_SWITCHING 19 19
# 	DISABLE_OUT_OF_ORDER_SELECTED_FIFO_EMPTY_SWITCHING 20 20
# 	DISABLE_OUT_OF_ORDER_EMPTY_SWITCHING_HYSTERYSIS 21 21
# 	ENABLE_OUT_OF_ORDER_DESIRED_FIFO_IS_NEXT_FEID 22 22
# 	DISABLE_OOO_NO_EOPG_SKEW_DESIRED_FIFO_IS_CURRENT_FIFO 23 23
# 	OOO_DISABLE_EOP_ON_FIRST_LIVE_PRIM_HIT 24 24
# 	OOO_DISABLE_EOPG_SKEW_THRESHOLD_SWITCHING 25 25
# 	DISABLE_EOP_LINE_STIPPLE_RESET 26 26
# 	DISABLE_VPZ_EOP_LINE_STIPPLE_RESET 27 27
# 	IOO_DISABLE_SCAN_UNSELECTED_FIFOS_FOR_DUAL_GFX_RING_CHANGE 28 28
# 	OOO_USE_ABSOLUTE_FIFO_COUNT_IN_THRESHOLD_SWITCHING 29 29
regPA_SC_ENHANCE_1 = 0x942
# 	REALIGN_DQUADS_OVERRIDE_ENABLE 0 0
# 	REALIGN_DQUADS_OVERRIDE 1 2
# 	DISABLE_SC_BINNING 3 3
# 	BYPASS_PBB 4 4
# 	DISABLE_NONBINNED_LIVE_PRIM_DG1_LS0_CL0_EOPKT_POKE 5 5
# 	ECO_SPARE1 6 6
# 	ECO_SPARE2 7 7
# 	ECO_SPARE3 8 8
# 	DISABLE_SC_PROCESS_RESET_PBB 9 9
# 	DISABLE_PBB_SCISSOR_OPT 10 10
# 	ENABLE_DFSM_FLUSH_EVENT_TO_FLUSH_POPS_CAM 11 11
# 	DISABLE_SC_DB_TILE_INTF_FINE_CLOCK_GATE 14 14
# 	DISABLE_PACKER_ODC_ENHANCE 16 16
# 	OPTIMAL_BIN_SELECTION 18 18
# 	DISABLE_FORCE_SOP_ALL_EVENTS 19 19
# 	DISABLE_PBB_CLK_OPTIMIZATION 20 20
# 	DISABLE_PBB_SCISSOR_CLK_OPTIMIZATION 21 21
# 	DISABLE_PBB_BINNING_CLK_OPTIMIZATION 22 22
# 	DISABLE_INTF_CG 23 23
# 	IOO_DISABLE_EOP_ON_FIRST_LIVE_PRIM_HIT 24 24
# 	DISABLE_SHADER_PROFILING_FOR_POWER 25 25
# 	FLUSH_ON_BINNING_TRANSITION 26 26
# 	DISABLE_QUAD_PROC_FDCE_ENHANCE 27 27
# 	DISABLE_SC_PS_PA_ARBITER_FIX 28 28
# 	DISABLE_SC_PS_PA_ARBITER_FIX_1 29 29
# 	PASS_VPZ_EVENT_TO_SPI 30 30
regPA_SC_ENHANCE_2 = 0x943
# 	DISABLE_SC_MEM_MACRO_FINE_CLOCK_GATE 0 0
# 	DISABLE_SC_DB_QUAD_INTF_FINE_CLOCK_GATE 1 1
# 	DISABLE_SC_BCI_QUAD_INTF_FINE_CLOCK_GATE 2 2
# 	DISABLE_SC_BCI_PRIM_INTF_FINE_CLOCK_GATE 3 3
# 	ENABLE_LPOV_WAVE_BREAK 4 4
# 	ENABLE_FPOV_WAVE_BREAK 5 5
# 	ENABLE_SC_SEND_DB_VPZ_FOR_EN_PRIM_PAYLOAD 7 7
# 	DISABLE_BREAK_BATCH_ON_GFX_PIPE_SWITCH 8 8
# 	DISABLE_FULL_TILE_WAVE_BREAK 9 9
# 	ENABLE_VPZ_INJECTION_BEFORE_NULL_PRIMS 10 10
# 	PBB_TIMEOUT_THRESHOLD_MODE 11 11
# 	DISABLE_PACKER_GRAD_FDCE_ENHANCE 12 12
# 	DISABLE_SC_SPI_INTF_EARLY_WAKEUP 13 13
# 	DISABLE_SC_BCI_INTF_EARLY_WAKEUP 14 14
# 	DISABLE_EXPOSED_GT_DETAIL_RATE_TILE_COV_ADJ 15 15
# 	PBB_WARP_CLK_MAIN_CLK_WAKEUP 16 16
# 	PBB_MAIN_CLK_REG_BUSY_WAKEUP 17 17
# 	DISABLE_BREAK_BATCH_ON_GFX_PIPELINE_RESET 18 18
# 	DISABLE_SC_DBR_DATAPATH_FGCG 21 21
# 	PROCESS_RESET_FORCE_STILE_MASK_TO_ZERO 23 23
# 	BREAK_WHEN_ONE_NULL_PRIM_BATCH 26 26
# 	NULL_PRIM_BREAK_BATCH_LIMIT 27 29
# 	DISABLE_MAX_DEALLOC_FORCE_EOV_RESET_N_WAVES_COUNT 30 30
# 	RSVD 31 31
regPA_SC_ENHANCE_3 = 0x944
# 	FORCE_USE_OF_SC_CENTROID_DATA 0 0
# 	DISABLE_RB_MASK_COPY_FOR_NONP2_SA_PAIR_HARVEST 2 2
# 	FORCE_PBB_WORKLOAD_MODE_TO_ZERO 3 3
# 	DISABLE_PKR_BCI_QUAD_NEW_PRIM_DATA_LOAD_OPTIMIZATION 4 4
# 	DISABLE_CP_CONTEXT_DONE_PERFCOUNT_SAMPLE_EN 5 5
# 	ENABLE_SINGLE_PA_EOPKT_FIRST_PHASE_FILTER 6 6
# 	ENABLE_SINGLE_PA_EOPKT_LAST_PHASE_FILTER 7 7
# 	ENABLE_SINGLE_PA_EOPKT_LAST_PHASE_FILTER_FOR_PBB_BINNED_PRIMS 8 8
# 	DISABLE_SET_VPZ_DIRTY_EOPKT_LAST_PHASE_ONLY 9 9
# 	DISABLE_PBB_EOP_OPTIMIZATION_WITH_SAME_CONTEXT_BATCHES 10 10
# 	DISABLE_FAST_NULL_PRIM_OPTIMIZATION 11 11
# 	USE_PBB_PRIM_STORAGE_WHEN_STALLED 12 12
# 	DISABLE_LIGHT_VOLUME_RENDERING_OPTIMIZATION 13 13
# 	DISABLE_ZPRE_PASS_OPTIMIZATION 14 14
# 	DISABLE_EVENT_INCLUSION_IN_CONTEXT_STATES_PER_BIN 15 15
# 	DISABLE_PIXEL_WAIT_SYNC_COUNTERS 16 16
# 	DISABLE_SC_CPG_PSINVOC_SEDC_ISOLATION_ACCUM 17 17
# 	DISABLE_SC_QP_VRS_RATE_FB_FINE_CLOCK_GATE 18 18
# 	DISABLE_SC_QP_VRS_RATE_CACHE_RD_FINE_CLOCK_GATE 19 19
# 	DISABLE_PKR_FORCE_EOV_MAX_REZ_CNT_FOR_SPI_BACKPRESSURE_ONLY 20 20
# 	DISABLE_PKR_FORCE_EOV_MAX_CLK_CNT_FOR_SPI_BACKPRESSURE_ONLY 21 21
# 	DO_NOT_INCLUDE_OREO_WAVEID_IN_FORCE_EOV_MAX_CNT_DISABLE 22 22
# 	DISABLE_PWS_PRE_DEPTH_WAIT_SYNC_VPZ_INSERTION 23 23
# 	PKR_CNT_FORCE_EOV_AT_QS_EMPTY_ONLY 24 24
# 	PKR_S0_FORCE_EOV_STALL 25 25
# 	PKR_S1_FORCE_EOV_STALL 26 26
# 	PKR_S2_FORCE_EOV_STALL 27 27
# 	ECO_SPARE0 28 28
# 	ECO_SPARE1 29 29
# 	ECO_SPARE2 30 30
# 	ECO_SPARE3 31 31
regPA_SC_FIFO_DEPTH_CNTL = 0x1035
# 	DEPTH 0 9
regPA_SC_FIFO_SIZE = 0x94a
# 	SC_FRONTEND_PRIM_FIFO_SIZE 0 5
# 	SC_BACKEND_PRIM_FIFO_SIZE 6 14
# 	SC_HIZ_TILE_FIFO_SIZE 15 20
# 	SC_EARLYZ_TILE_FIFO_SIZE 21 31
regPA_SC_FORCE_EOV_MAX_CNTS = 0x94f
# 	FORCE_EOV_MAX_CLK_CNT 0 15
# 	FORCE_EOV_MAX_REZ_CNT 16 31
regPA_SC_GENERIC_SCISSOR_BR = 0x91
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_GENERIC_SCISSOR_TL = 0x90
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_HP3D_TRAP_SCREEN_COUNT = 0x22ac
# 	COUNT 0 15
regPA_SC_HP3D_TRAP_SCREEN_H = 0x22a9
# 	X_COORD 0 13
regPA_SC_HP3D_TRAP_SCREEN_HV_EN = 0x22a8
# 	ENABLE_HV_PRE_SHADER 0 0
# 	FORCE_PRE_SHADER_ALL_PIXELS 1 1
regPA_SC_HP3D_TRAP_SCREEN_HV_LOCK = 0x95c
# 	DISABLE_NON_PRIV_WRITES 0 0
regPA_SC_HP3D_TRAP_SCREEN_OCCURRENCE = 0x22ab
# 	COUNT 0 15
regPA_SC_HP3D_TRAP_SCREEN_V = 0x22aa
# 	Y_COORD 0 13
regPA_SC_IF_FIFO_SIZE = 0x94b
# 	SC_DB_TILE_IF_FIFO_SIZE 0 5
# 	SC_DB_QUAD_IF_FIFO_SIZE 6 11
# 	SC_SPI_IF_FIFO_SIZE 12 17
# 	SC_BCI_IF_FIFO_SIZE 18 23
regPA_SC_LINE_CNTL = 0x2f7
# 	EXPAND_LINE_WIDTH 9 9
# 	LAST_PIXEL 10 10
# 	PERPENDICULAR_ENDCAP_ENA 11 11
# 	DX10_DIAMOND_TEST_ENA 12 12
# 	EXTRA_DX_DY_PRECISION 13 13
regPA_SC_LINE_STIPPLE = 0x283
# 	LINE_PATTERN 0 15
# 	REPEAT_COUNT 16 23
# 	PATTERN_BIT_ORDER 28 28
# 	AUTO_RESET_CNTL 29 30
regPA_SC_LINE_STIPPLE_STATE = 0x2281
# 	CURRENT_PTR 0 3
# 	CURRENT_COUNT 8 15
regPA_SC_MODE_CNTL_0 = 0x292
# 	MSAA_ENABLE 0 0
# 	VPORT_SCISSOR_ENABLE 1 1
# 	LINE_STIPPLE_ENABLE 2 2
# 	SEND_UNLIT_STILES_TO_PKR 3 3
# 	ALTERNATE_RBS_PER_TILE 5 5
# 	COARSE_TILE_STARTS_ON_EVEN_RB 6 6
regPA_SC_MODE_CNTL_1 = 0x293
# 	WALK_SIZE 0 0
# 	WALK_ALIGNMENT 1 1
# 	WALK_ALIGN8_PRIM_FITS_ST 2 2
# 	WALK_FENCE_ENABLE 3 3
# 	WALK_FENCE_SIZE 4 6
# 	SUPERTILE_WALK_ORDER_ENABLE 7 7
# 	TILE_WALK_ORDER_ENABLE 8 8
# 	TILE_COVER_DISABLE 9 9
# 	TILE_COVER_NO_SCISSOR 10 10
# 	ZMM_LINE_EXTENT 11 11
# 	ZMM_LINE_OFFSET 12 12
# 	ZMM_RECT_EXTENT 13 13
# 	KILL_PIX_POST_HI_Z 14 14
# 	KILL_PIX_POST_DETAIL_MASK 15 15
# 	PS_ITER_SAMPLE 16 16
# 	MULTI_SHADER_ENGINE_PRIM_DISCARD_ENABLE 17 17
# 	MULTI_GPU_SUPERTILE_ENABLE 18 18
# 	GPU_ID_OVERRIDE_ENABLE 19 19
# 	GPU_ID_OVERRIDE 20 23
# 	MULTI_GPU_PRIM_DISCARD_ENABLE 24 24
# 	FORCE_EOV_CNTDWN_ENABLE 25 25
# 	FORCE_EOV_REZ_ENABLE 26 26
# 	OUT_OF_ORDER_PRIMITIVE_ENABLE 27 27
# 	OUT_OF_ORDER_WATER_MARK 28 30
regPA_SC_NGG_MODE_CNTL = 0x314
# 	MAX_DEALLOCS_IN_WAVE 0 10
# 	DISABLE_FPOG_AND_DEALLOC_CONFLICT 12 12
# 	DISABLE_MAX_DEALLOC 13 13
# 	DISABLE_MAX_ATTRIBUTES 14 14
# 	MAX_FPOVS_IN_WAVE 16 23
# 	MAX_ATTRIBUTES_IN_WAVE 24 31
regPA_SC_P3D_TRAP_SCREEN_COUNT = 0x22a4
# 	COUNT 0 15
regPA_SC_P3D_TRAP_SCREEN_H = 0x22a1
# 	X_COORD 0 13
regPA_SC_P3D_TRAP_SCREEN_HV_EN = 0x22a0
# 	ENABLE_HV_PRE_SHADER 0 0
# 	FORCE_PRE_SHADER_ALL_PIXELS 1 1
regPA_SC_P3D_TRAP_SCREEN_HV_LOCK = 0x95b
# 	DISABLE_NON_PRIV_WRITES 0 0
regPA_SC_P3D_TRAP_SCREEN_OCCURRENCE = 0x22a3
# 	COUNT 0 15
regPA_SC_P3D_TRAP_SCREEN_V = 0x22a2
# 	Y_COORD 0 13
regPA_SC_PACKER_WAVE_ID_CNTL = 0x94c
# 	WAVE_TABLE_SIZE 0 9
# 	SC_DB_WAVE_IF_FIFO_SIZE 10 15
# 	DISABLE_SC_DB_WAVE_IF_FGCG_EN 16 16
# 	SC_SPI_WAVE_IF_FIFO_SIZE 17 22
# 	DISABLE_SC_SPI_WAVE_IF_FGCG_EN 23 23
# 	DISABLE_OREO_CONFLICT_QUAD 31 31
regPA_SC_PBB_OVERRIDE_FLAG = 0x947
# 	OVERRIDE 0 0
# 	PIPE_ID 1 1
regPA_SC_PERFCOUNTER0_HI = 0x3141
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER0_LO = 0x3140
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER0_SELECT = 0x3940
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_SC_PERFCOUNTER0_SELECT1 = 0x3941
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_SC_PERFCOUNTER1_HI = 0x3143
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER1_LO = 0x3142
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER1_SELECT = 0x3942
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER2_HI = 0x3145
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER2_LO = 0x3144
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER2_SELECT = 0x3943
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER3_HI = 0x3147
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER3_LO = 0x3146
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER3_SELECT = 0x3944
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER4_HI = 0x3149
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER4_LO = 0x3148
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER4_SELECT = 0x3945
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER5_HI = 0x314b
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER5_LO = 0x314a
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER5_SELECT = 0x3946
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER6_HI = 0x314d
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER6_LO = 0x314c
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER6_SELECT = 0x3947
# 	PERF_SEL 0 9
regPA_SC_PERFCOUNTER7_HI = 0x314f
# 	PERFCOUNTER_HI 0 31
regPA_SC_PERFCOUNTER7_LO = 0x314e
# 	PERFCOUNTER_LO 0 31
regPA_SC_PERFCOUNTER7_SELECT = 0x3948
# 	PERF_SEL 0 9
regPA_SC_PKR_WAVE_TABLE_CNTL = 0x94e
# 	SIZE 0 5
regPA_SC_RASTER_CONFIG = 0xd4
# 	RB_MAP_PKR0 0 1
# 	RB_MAP_PKR1 2 3
# 	RB_XSEL2 4 5
# 	RB_XSEL 6 6
# 	RB_YSEL 7 7
# 	PKR_MAP 8 9
# 	PKR_XSEL 10 11
# 	PKR_YSEL 12 13
# 	PKR_XSEL2 14 15
# 	SC_MAP 16 17
# 	SC_XSEL 18 19
# 	SC_YSEL 20 21
# 	SE_MAP 24 25
# 	SE_XSEL 26 27
# 	SE_YSEL 28 29
regPA_SC_RASTER_CONFIG_1 = 0xd5
# 	SE_PAIR_MAP 0 1
# 	SE_PAIR_XSEL 2 3
# 	SE_PAIR_YSEL 4 5
regPA_SC_SCREEN_EXTENT_CONTROL = 0xd6
# 	SLICE_EVEN_ENABLE 0 1
# 	SLICE_ODD_ENABLE 2 3
regPA_SC_SCREEN_EXTENT_MAX_0 = 0x2285
# 	X 0 15
# 	Y 16 31
regPA_SC_SCREEN_EXTENT_MAX_1 = 0x228b
# 	X 0 15
# 	Y 16 31
regPA_SC_SCREEN_EXTENT_MIN_0 = 0x2284
# 	X 0 15
# 	Y 16 31
regPA_SC_SCREEN_EXTENT_MIN_1 = 0x2286
# 	X 0 15
# 	Y 16 31
regPA_SC_SCREEN_SCISSOR_BR = 0xd
# 	BR_X 0 15
# 	BR_Y 16 31
regPA_SC_SCREEN_SCISSOR_TL = 0xc
# 	TL_X 0 15
# 	TL_Y 16 31
regPA_SC_SHADER_CONTROL = 0x310
# 	REALIGN_DQUADS_AFTER_N_WAVES 0 1
# 	LOAD_COLLISION_WAVEID 2 2
# 	LOAD_INTRAWAVE_COLLISION 3 3
# 	WAVE_BREAK_REGION_SIZE 5 6
# 	DISABLE_OREO_CONFLICT_QUAD 7 7
regPA_SC_TILE_STEERING_CREST_OVERRIDE = 0x949
# 	ONE_RB_MODE_ENABLE 0 0
# 	SE_SELECT 1 2
# 	RB_SELECT 5 6
# 	SA_SELECT 8 10
# 	FORCE_TILE_STEERING_OVERRIDE_USE 31 31
regPA_SC_TILE_STEERING_OVERRIDE = 0xd7
# 	ENABLE 0 0
# 	NUM_SC 12 13
# 	NUM_RB_PER_SC 16 17
# 	NUM_PACKER_PER_SC 20 21
regPA_SC_TRAP_SCREEN_COUNT = 0x22b4
# 	COUNT 0 15
regPA_SC_TRAP_SCREEN_H = 0x22b1
# 	X_COORD 0 13
regPA_SC_TRAP_SCREEN_HV_EN = 0x22b0
# 	ENABLE_HV_PRE_SHADER 0 0
# 	FORCE_PRE_SHADER_ALL_PIXELS 1 1
regPA_SC_TRAP_SCREEN_HV_LOCK = 0x95d
# 	DISABLE_NON_PRIV_WRITES 0 0
regPA_SC_TRAP_SCREEN_OCCURRENCE = 0x22b3
# 	COUNT 0 15
regPA_SC_TRAP_SCREEN_V = 0x22b2
# 	Y_COORD 0 13
regPA_SC_VPORT_SCISSOR_0_BR = 0x95
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_0_TL = 0x94
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_10_BR = 0xa9
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_10_TL = 0xa8
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_11_BR = 0xab
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_11_TL = 0xaa
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_12_BR = 0xad
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_12_TL = 0xac
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_13_BR = 0xaf
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_13_TL = 0xae
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_14_BR = 0xb1
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_14_TL = 0xb0
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_15_BR = 0xb3
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_15_TL = 0xb2
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_1_BR = 0x97
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_1_TL = 0x96
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_2_BR = 0x99
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_2_TL = 0x98
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_3_BR = 0x9b
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_3_TL = 0x9a
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_4_BR = 0x9d
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_4_TL = 0x9c
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_5_BR = 0x9f
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_5_TL = 0x9e
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_6_BR = 0xa1
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_6_TL = 0xa0
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_7_BR = 0xa3
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_7_TL = 0xa2
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_8_BR = 0xa5
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_8_TL = 0xa4
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_SCISSOR_9_BR = 0xa7
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_VPORT_SCISSOR_9_TL = 0xa6
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_SC_VPORT_ZMAX_0 = 0xb5
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_1 = 0xb7
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_10 = 0xc9
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_11 = 0xcb
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_12 = 0xcd
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_13 = 0xcf
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_14 = 0xd1
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_15 = 0xd3
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_2 = 0xb9
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_3 = 0xbb
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_4 = 0xbd
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_5 = 0xbf
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_6 = 0xc1
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_7 = 0xc3
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_8 = 0xc5
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMAX_9 = 0xc7
# 	VPORT_ZMAX 0 31
regPA_SC_VPORT_ZMIN_0 = 0xb4
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_1 = 0xb6
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_10 = 0xc8
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_11 = 0xca
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_12 = 0xcc
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_13 = 0xce
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_14 = 0xd0
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_15 = 0xd2
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_2 = 0xb8
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_3 = 0xba
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_4 = 0xbc
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_5 = 0xbe
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_6 = 0xc0
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_7 = 0xc2
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_8 = 0xc4
# 	VPORT_ZMIN 0 31
regPA_SC_VPORT_ZMIN_9 = 0xc6
# 	VPORT_ZMIN 0 31
regPA_SC_VRS_OVERRIDE_CNTL = 0xf4
# 	VRS_OVERRIDE_RATE_COMBINER_MODE 0 2
# 	VRS_RATE 4 7
# 	VRS_SURFACE_ENABLE 12 12
# 	RATE_HINT_WRITE_BACK_ENABLE 13 13
# 	VRS_FEEDBACK_RATE_OVERRIDE 14 14
regPA_SC_VRS_RATE_BASE = 0xfc
# 	BASE_256B 0 31
regPA_SC_VRS_RATE_BASE_EXT = 0xfd
# 	BASE_256B 0 7
# 	TB_SYNC_SIM_ID 28 31
regPA_SC_VRS_RATE_CACHE_CNTL = 0xf9
# 	BIG_PAGE_RD 0 0
# 	BIG_PAGE_WR 1 1
# 	L1_RD_POLICY 2 3
# 	L2_RD_POLICY 4 5
# 	L2_WR_POLICY 6 7
# 	LLC_RD_NOALLOC 8 8
# 	LLC_WR_NOALLOC 9 9
# 	NOFILL_RD 10 10
# 	NOFILL_WR 11 11
# 	PERF_CNTR_EN_RD 12 12
# 	PERF_CNTR_EN_WR 13 13
regPA_SC_VRS_RATE_FEEDBACK_BASE = 0xf5
# 	BASE_256B 0 31
regPA_SC_VRS_RATE_FEEDBACK_BASE_EXT = 0xf6
# 	BASE_256B 0 7
regPA_SC_VRS_RATE_FEEDBACK_SIZE_XY = 0xf7
# 	X_MAX 0 10
# 	Y_MAX 16 26
regPA_SC_VRS_RATE_SIZE_XY = 0xfe
# 	X_MAX 0 10
# 	Y_MAX 16 26
regPA_SC_VRS_SURFACE_CNTL = 0x940
# 	VRC_CONTEXT_DONE_SYNC_DISABLE 6 6
# 	VRS_FEEDBACK_RATE_OVERRIDE 7 7
# 	VRC_FLUSH_EVENT_MASK_DISABLE 8 12
# 	VRC_PREFETCH_DISABLE 13 13
# 	VRC_FLUSH_NO_INV_DISABLE 14 14
# 	VRC_NONSTALLING_FLUSH_DISABLE 15 15
# 	VRC_PARTIAL_FLUSH_DISABLE 16 16
# 	VRC_AUTO_FLUSH 17 17
# 	VRC_EOP_SYNC_DISABLE 18 18
# 	VRC_MAX_TAGS 19 25
# 	VRC_EVICT_POINT 26 31
regPA_SC_VRS_SURFACE_CNTL_1 = 0x960
# 	FORCE_SC_VRS_RATE_FINE 0 0
# 	FORCE_SC_VRS_RATE_FINE_SHADER_KILL_ENABLE 1 1
# 	FORCE_SC_VRS_RATE_FINE_MASK_OPS_ENABLE 2 2
# 	FORCE_SC_VRS_RATE_FINE_RATE_16XAA 3 3
# 	FORCE_SC_VRS_RATE_FINE_Z_OR_STENCIL 4 4
# 	FORCE_SC_VRS_RATE_FINE_PRE_SHADER_DEPTH_COVERAGE_ENABLED 5 5
# 	FORCE_SC_VRS_RATE_FINE_POST_DEPTH_IMPORT 6 6
# 	FORCE_SC_VRS_RATE_FINE_POPS 7 7
# 	USE_ONLY_VRS_RATE_FINE_CFG 8 8
# 	DISABLE_SSAA_VRS_RATE_NORMALIZATION 12 12
# 	DISABLE_PS_ITER_RATE_COMBINER_PASSTHRU_OVERRIDE 15 15
# 	DISABLE_CMASK_RATE_HINT_FORCE_ZERO_OVERRIDE 19 19
# 	DISABLE_SSAA_DETAIL_TO_EXPOSED_RATE_CLAMPING 20 20
# 	VRS_ECO_SPARE_0 21 21
# 	VRS_ECO_SPARE_1 22 22
# 	VRS_ECO_SPARE_2 23 23
# 	VRS_ECO_SPARE_3 24 24
# 	VRS_ECO_SPARE_4 25 25
# 	VRS_ECO_SPARE_5 26 26
# 	VRS_ECO_SPARE_6 27 27
# 	VRS_ECO_SPARE_7 28 28
# 	VRS_ECO_SPARE_8 29 29
# 	VRS_ECO_SPARE_9 30 30
# 	VRS_ECO_SPARE_10 31 31
regPA_SC_WINDOW_OFFSET = 0x80
# 	WINDOW_X_OFFSET 0 15
# 	WINDOW_Y_OFFSET 16 31
regPA_SC_WINDOW_SCISSOR_BR = 0x82
# 	BR_X 0 14
# 	BR_Y 16 30
regPA_SC_WINDOW_SCISSOR_TL = 0x81
# 	TL_X 0 14
# 	TL_Y 16 30
# 	WINDOW_OFFSET_DISABLE 31 31
regPA_STATE_STEREO_X = 0x211
# 	STEREO_X_OFFSET 0 31
regPA_STEREO_CNTL = 0x210
# 	STEREO_MODE 1 4
# 	RT_SLICE_MODE 5 7
# 	RT_SLICE_OFFSET 8 11
# 	VP_ID_MODE 16 18
# 	VP_ID_OFFSET 19 22
regPA_SU_CNTL_STATUS = 0x1034
# 	SU_BUSY 31 31
regPA_SU_HARDWARE_SCREEN_OFFSET = 0x8d
# 	HW_SCREEN_OFFSET_X 0 8
# 	HW_SCREEN_OFFSET_Y 16 24
regPA_SU_LINE_CNTL = 0x282
# 	WIDTH 0 15
regPA_SU_LINE_STIPPLE_CNTL = 0x209
# 	LINE_STIPPLE_RESET 0 1
# 	EXPAND_FULL_LENGTH 2 2
# 	FRACTIONAL_ACCUM 3 3
regPA_SU_LINE_STIPPLE_SCALE = 0x20a
# 	LINE_STIPPLE_SCALE 0 31
regPA_SU_LINE_STIPPLE_VALUE = 0x2280
# 	LINE_STIPPLE_VALUE 0 23
regPA_SU_OVER_RASTERIZATION_CNTL = 0x20f
# 	DISCARD_0_AREA_TRIANGLES 0 0
# 	DISCARD_0_AREA_LINES 1 1
# 	DISCARD_0_AREA_POINTS 2 2
# 	DISCARD_0_AREA_RECTANGLES 3 3
# 	USE_PROVOKING_ZW 4 4
regPA_SU_PERFCOUNTER0_HI = 0x3101
# 	PERFCOUNTER_HI 0 31
regPA_SU_PERFCOUNTER0_LO = 0x3100
# 	PERFCOUNTER_LO 0 31
regPA_SU_PERFCOUNTER0_SELECT = 0x3900
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_SU_PERFCOUNTER0_SELECT1 = 0x3901
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_SU_PERFCOUNTER1_HI = 0x3103
# 	PERFCOUNTER_HI 0 31
regPA_SU_PERFCOUNTER1_LO = 0x3102
# 	PERFCOUNTER_LO 0 31
regPA_SU_PERFCOUNTER1_SELECT = 0x3902
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_SU_PERFCOUNTER1_SELECT1 = 0x3903
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_SU_PERFCOUNTER2_HI = 0x3105
# 	PERFCOUNTER_HI 0 31
regPA_SU_PERFCOUNTER2_LO = 0x3104
# 	PERFCOUNTER_LO 0 31
regPA_SU_PERFCOUNTER2_SELECT = 0x3904
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_SU_PERFCOUNTER2_SELECT1 = 0x3905
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_SU_PERFCOUNTER3_HI = 0x3107
# 	PERFCOUNTER_HI 0 31
regPA_SU_PERFCOUNTER3_LO = 0x3106
# 	PERFCOUNTER_LO 0 31
regPA_SU_PERFCOUNTER3_SELECT = 0x3906
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPA_SU_PERFCOUNTER3_SELECT1 = 0x3907
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPA_SU_POINT_MINMAX = 0x281
# 	MIN_SIZE 0 15
# 	MAX_SIZE 16 31
regPA_SU_POINT_SIZE = 0x280
# 	HEIGHT 0 15
# 	WIDTH 16 31
regPA_SU_POLY_OFFSET_BACK_OFFSET = 0x2e3
# 	OFFSET 0 31
regPA_SU_POLY_OFFSET_BACK_SCALE = 0x2e2
# 	SCALE 0 31
regPA_SU_POLY_OFFSET_CLAMP = 0x2df
# 	CLAMP 0 31
regPA_SU_POLY_OFFSET_DB_FMT_CNTL = 0x2de
# 	POLY_OFFSET_NEG_NUM_DB_BITS 0 7
# 	POLY_OFFSET_DB_IS_FLOAT_FMT 8 8
regPA_SU_POLY_OFFSET_FRONT_OFFSET = 0x2e1
# 	OFFSET 0 31
regPA_SU_POLY_OFFSET_FRONT_SCALE = 0x2e0
# 	SCALE 0 31
regPA_SU_PRIM_FILTER_CNTL = 0x20b
# 	TRIANGLE_FILTER_DISABLE 0 0
# 	LINE_FILTER_DISABLE 1 1
# 	POINT_FILTER_DISABLE 2 2
# 	RECTANGLE_FILTER_DISABLE 3 3
# 	TRIANGLE_EXPAND_ENA 4 4
# 	LINE_EXPAND_ENA 5 5
# 	POINT_EXPAND_ENA 6 6
# 	RECTANGLE_EXPAND_ENA 7 7
# 	PRIM_EXPAND_CONSTANT 8 15
# 	XMAX_RIGHT_EXCLUSION 30 30
# 	YMAX_BOTTOM_EXCLUSION 31 31
regPA_SU_SC_MODE_CNTL = 0x205
# 	CULL_FRONT 0 0
# 	CULL_BACK 1 1
# 	FACE 2 2
# 	POLY_MODE 3 4
# 	POLYMODE_FRONT_PTYPE 5 7
# 	POLYMODE_BACK_PTYPE 8 10
# 	POLY_OFFSET_FRONT_ENABLE 11 11
# 	POLY_OFFSET_BACK_ENABLE 12 12
# 	POLY_OFFSET_PARA_ENABLE 13 13
# 	VTX_WINDOW_OFFSET_ENABLE 16 16
# 	PROVOKING_VTX_LAST 19 19
# 	PERSP_CORR_DIS 20 20
# 	MULTI_PRIM_IB_ENA 21 21
# 	RIGHT_TRIANGLE_ALTERNATE_GRADIENT_REF 22 22
# 	NEW_QUAD_DECOMPOSITION 23 23
# 	KEEP_TOGETHER_ENABLE 24 24
regPA_SU_SMALL_PRIM_FILTER_CNTL = 0x20c
# 	SMALL_PRIM_FILTER_ENABLE 0 0
# 	TRIANGLE_FILTER_DISABLE 1 1
# 	LINE_FILTER_DISABLE 2 2
# 	POINT_FILTER_DISABLE 3 3
# 	RECTANGLE_FILTER_DISABLE 4 4
regPA_SU_VTX_CNTL = 0x2f9
# 	PIX_CENTER 0 0
# 	ROUND_MODE 1 2
# 	QUANT_MODE 3 5
regPCC_PERF_COUNTER = 0x1b0c
# 	PCC_PERF_COUNTER 0 31
regPCC_PWRBRK_HYSTERESIS_CTRL = 0x1b03
# 	PCC_MAX_HYSTERESIS 0 7
# 	PWRBRK_MAX_HYSTERESIS 8 15
regPCC_STALL_PATTERN_1_2 = 0x1af6
# 	PCC_STALL_PATTERN_1 0 14
# 	PCC_STALL_PATTERN_2 16 30
regPCC_STALL_PATTERN_3_4 = 0x1af7
# 	PCC_STALL_PATTERN_3 0 14
# 	PCC_STALL_PATTERN_4 16 30
regPCC_STALL_PATTERN_5_6 = 0x1af8
# 	PCC_STALL_PATTERN_5 0 14
# 	PCC_STALL_PATTERN_6 16 30
regPCC_STALL_PATTERN_7 = 0x1af9
# 	PCC_STALL_PATTERN_7 0 14
regPCC_STALL_PATTERN_CTRL = 0x1af4
# 	PCC_STEP_INTERVAL 0 9
# 	PCC_BEGIN_STEP 10 14
# 	PCC_END_STEP 15 19
# 	PCC_THROTTLE_PATTERN_BIT_NUMS 20 23
# 	PCC_INST_THROT_INCR 24 24
# 	PCC_INST_THROT_DECR 25 25
# 	PCC_DITHER_MODE 26 26
regPC_PERFCOUNTER0_HI = 0x318c
# 	PERFCOUNTER_HI 0 31
regPC_PERFCOUNTER0_LO = 0x318d
# 	PERFCOUNTER_LO 0 31
regPC_PERFCOUNTER0_SELECT = 0x398c
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPC_PERFCOUNTER0_SELECT1 = 0x3990
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPC_PERFCOUNTER1_HI = 0x318e
# 	PERFCOUNTER_HI 0 31
regPC_PERFCOUNTER1_LO = 0x318f
# 	PERFCOUNTER_LO 0 31
regPC_PERFCOUNTER1_SELECT = 0x398d
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPC_PERFCOUNTER1_SELECT1 = 0x3991
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPC_PERFCOUNTER2_HI = 0x3190
# 	PERFCOUNTER_HI 0 31
regPC_PERFCOUNTER2_LO = 0x3191
# 	PERFCOUNTER_LO 0 31
regPC_PERFCOUNTER2_SELECT = 0x398e
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPC_PERFCOUNTER2_SELECT1 = 0x3992
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPC_PERFCOUNTER3_HI = 0x3192
# 	PERFCOUNTER_HI 0 31
regPC_PERFCOUNTER3_LO = 0x3193
# 	PERFCOUNTER_LO 0 31
regPC_PERFCOUNTER3_SELECT = 0x398f
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regPC_PERFCOUNTER3_SELECT1 = 0x3993
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regPMM_CNTL = 0x1582
# 	PMM_DISABLE 0 0
# 	ABIT_FORCE_FLUSH 1 1
# 	ABIT_TIMER_THRESHOLD 2 5
# 	ABIT_TIMER_DISABLE 6 6
# 	ABIT_TIMER_RESET 7 7
# 	INTERRUPT_PRIORITY 8 9
# 	PMM_INTERRUPTS_DISABLE 10 10
# 	RESERVED 11 31
regPMM_CNTL2 = 0x1999
# 	ABIT_FORCE_FLUSH_OVERRIDE 24 24
# 	ABIT_TIMER_FLUSH_OVERRIDE 25 25
# 	PMM_IH_INTERRUPT_CREDITS_OVERRIDE 26 29
# 	ABIT_INTR_ON_FLUSH_DONE 30 30
# 	RESERVED 31 31
regPMM_STATUS = 0x1583
# 	PMM_IDLE 0 0
# 	ABIT_FORCE_FLUSH_IN_PROGRESS 1 1
# 	ABIT_FORCE_FLUSH_DONE 2 2
# 	ABIT_TIMER_FLUSH_IN_PROGRESS 3 3
# 	ABIT_TIMER_FLUSH_DONE 4 4
# 	ABIT_TIMER_RUNNING 5 5
# 	PMM_INTERRUPTS_PENDING 6 6
# 	ABIT_FLUSH_ERROR 7 7
# 	ABIT_TIMER_RESET_CDC_IN_PROGRESS 8 8
# 	ABIT_TIMER_ENABLE_CDC_IN_PROGRESS 9 9
# 	ABIT_TIMER_THRESHOLD_CDC_IN_PROGRESS 10 10
# 	RESERVED 11 31
regPWRBRK_PERF_COUNTER = 0x1b0d
# 	PWRBRK_PERF_COUNTER 0 31
regPWRBRK_STALL_PATTERN_1_2 = 0x1afa
# 	PWRBRK_STALL_PATTERN_1 0 14
# 	PWRBRK_STALL_PATTERN_2 16 30
regPWRBRK_STALL_PATTERN_3_4 = 0x1afb
# 	PWRBRK_STALL_PATTERN_3 0 14
# 	PWRBRK_STALL_PATTERN_4 16 30
regPWRBRK_STALL_PATTERN_5_6 = 0x1afc
# 	PWRBRK_STALL_PATTERN_5 0 14
# 	PWRBRK_STALL_PATTERN_6 16 30
regPWRBRK_STALL_PATTERN_7 = 0x1afd
# 	PWRBRK_STALL_PATTERN_7 0 14
regPWRBRK_STALL_PATTERN_CTRL = 0x1af5
# 	PWRBRK_STEP_INTERVAL 0 9
# 	PWRBRK_BEGIN_STEP 10 14
# 	PWRBRK_END_STEP 15 19
# 	PWRBRK_THROTTLE_PATTERN_BIT_NUMS 20 23
regRLC_AUTO_PG_CTRL = 0x4c55
# 	AUTO_PG_EN 0 0
# 	AUTO_GRBM_REG_SAVE_ON_IDLE_EN 1 1
# 	AUTO_WAKE_UP_EN 2 2
# 	GRBM_REG_SAVE_GFX_IDLE_THRESHOLD 3 18
# 	PG_AFTER_GRBM_REG_SAVE_THRESHOLD 19 31
regRLC_BUSY_CLK_CNTL = 0x5b30
# 	BUSY_OFF_LATENCY 0 5
# 	GRBM_BUSY_OFF_LATENCY 8 13
regRLC_CAC_MASK_CNTL = 0x4d45
# 	RLC_CAC_MASK 0 31
regRLC_CAPTURE_GPU_CLOCK_COUNT = 0x4c26
# 	CAPTURE 0 0
# 	RESERVED 1 31
regRLC_CAPTURE_GPU_CLOCK_COUNT_1 = 0x4cea
# 	CAPTURE 0 0
# 	RESERVED 1 31
regRLC_CAPTURE_GPU_CLOCK_COUNT_2 = 0x4cef
# 	CAPTURE 0 0
# 	RESERVED 1 31
regRLC_CGCG_CGLS_CTRL = 0x4c49
# 	CGCG_EN 0 0
# 	CGLS_EN 1 1
# 	CGLS_REP_COMPANSAT_DELAY 2 7
# 	CGCG_GFX_IDLE_THRESHOLD 8 26
# 	CGCG_CONTROLLER 27 27
# 	CGCG_REG_CTRL 28 28
# 	SLEEP_MODE 29 30
# 	SIM_SILICON_EN 31 31
regRLC_CGCG_CGLS_CTRL_3D = 0x4cc5
# 	CGCG_EN 0 0
# 	CGLS_EN 1 1
# 	CGLS_REP_COMPANSAT_DELAY 2 7
# 	CGCG_GFX_IDLE_THRESHOLD 8 26
# 	CGCG_CONTROLLER 27 27
# 	CGCG_REG_CTRL 28 28
# 	SLEEP_MODE 29 30
# 	SIM_SILICON_EN 31 31
regRLC_CGCG_RAMP_CTRL = 0x4c4a
# 	DOWN_DIV_START_UNIT 0 3
# 	DOWN_DIV_STEP_UNIT 4 7
# 	UP_DIV_START_UNIT 8 11
# 	UP_DIV_STEP_UNIT 12 15
# 	STEP_DELAY_CNT 16 27
# 	STEP_DELAY_UNIT 28 31
regRLC_CGCG_RAMP_CTRL_3D = 0x4cc6
# 	DOWN_DIV_START_UNIT 0 3
# 	DOWN_DIV_STEP_UNIT 4 7
# 	UP_DIV_START_UNIT 8 11
# 	UP_DIV_STEP_UNIT 12 15
# 	STEP_DELAY_CNT 16 27
# 	STEP_DELAY_UNIT 28 31
regRLC_CGTT_MGCG_OVERRIDE = 0x4c48
# 	RLC_REPEATER_FGCG_OVERRIDE 0 0
# 	RLC_CGTT_SCLK_OVERRIDE 1 1
# 	GFXIP_MGCG_OVERRIDE 2 2
# 	GFXIP_CGCG_OVERRIDE 3 3
# 	GFXIP_CGLS_OVERRIDE 4 4
# 	GRBM_CGTT_SCLK_OVERRIDE 5 5
# 	GFXIP_MGLS_OVERRIDE 6 6
# 	GFXIP_GFX3D_CG_OVERRIDE 7 7
# 	GFXIP_FGCG_OVERRIDE 8 8
# 	GFXIP_REPEATER_FGCG_OVERRIDE 9 9
# 	PERFMON_CLOCK_STATE 10 10
# 	RESERVED_16_11 11 16
# 	GC_CAC_MGCG_CLK_CNTL 17 17
# 	SE_CAC_MGCG_CLK_CNTL 18 18
# 	RESERVED_31_19 19 31
regRLC_CLK_CNTL = 0x5b31
# 	RLC_SRM_ICG_OVERRIDE 0 0
# 	RLC_IMU_ICG_OVERRIDE 1 1
# 	RLC_SPM_ICG_OVERRIDE 2 2
# 	RLC_SPM_RSPM_ICG_OVERRIDE 3 3
# 	RLC_GPM_ICG_OVERRIDE 4 4
# 	RLC_CMN_ICG_OVERRIDE 5 5
# 	RLC_TC_ICG_OVERRIDE 6 6
# 	RLC_REG_ICG_OVERRIDE 7 7
# 	RLC_SRAM_CLK_GATER_OVERRIDE 8 8
# 	RESERVED_9 9 9
# 	RLC_SPP_ICG_OVERRIDE 10 10
# 	RESERVED_11 11 11
# 	RLC_TC_FGCG_REP_OVERRIDE 12 12
# 	RESERVED_15 15 15
# 	RLC_UTCL2_FGCG_OVERRIDE 18 18
# 	RLC_IH_GASKET_ICG_OVERRIDE 19 19
# 	RESERVED 20 31
regRLC_CLK_COUNT_CTRL = 0x4c34
# 	GFXCLK_RUN 0 0
# 	GFXCLK_RESET 1 1
# 	GFXCLK_SAMPLE 2 2
# 	REFCLK_RUN 3 3
# 	REFCLK_RESET 4 4
# 	REFCLK_SAMPLE 5 5
regRLC_CLK_COUNT_GFXCLK_LSB = 0x4c30
# 	COUNTER 0 31
regRLC_CLK_COUNT_GFXCLK_MSB = 0x4c31
# 	COUNTER 0 31
regRLC_CLK_COUNT_REFCLK_LSB = 0x4c32
# 	COUNTER 0 31
regRLC_CLK_COUNT_REFCLK_MSB = 0x4c33
# 	COUNTER 0 31
regRLC_CLK_COUNT_STAT = 0x4c35
# 	GFXCLK_VALID 0 0
# 	REFCLK_VALID 1 1
# 	REFCLK_RUN_RESYNC 2 2
# 	REFCLK_RESET_RESYNC 3 3
# 	REFCLK_SAMPLE_RESYNC 4 4
# 	RESERVED 5 31
regRLC_CLK_RESIDENCY_CNTR_CTRL = 0x4d49
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	RESERVED 5 31
regRLC_CLK_RESIDENCY_EVENT_CNTR = 0x4d51
# 	DATA 0 31
regRLC_CLK_RESIDENCY_REF_CNTR = 0x4d59
# 	DATA 0 31
regRLC_CNTL = 0x4c00
# 	RLC_ENABLE_F32 0 0
# 	FORCE_RETRY 1 1
# 	READ_CACHE_DISABLE 2 2
# 	RLC_STEP_F32 3 3
# 	RESERVED 4 31
regRLC_CP_EOF_INT = 0x98b
# 	INTERRUPT 0 0
# 	RESERVED 1 31
regRLC_CP_EOF_INT_CNT = 0x98c
# 	CNT 0 31
regRLC_CP_SCHEDULERS = 0x98a
# 	scheduler0 0 7
# 	scheduler1 8 15
regRLC_CP_STAT_INVAL_CTRL = 0x4d0a
# 	CPG_STAT_INVAL_PEND_EN 0 0
# 	CPC_STAT_INVAL_PEND_EN 1 1
# 	CPF_STAT_INVAL_PEND_EN 2 2
regRLC_CP_STAT_INVAL_STAT = 0x4d09
# 	CPG_STAT_INVAL_PEND 0 0
# 	CPC_STAT_INVAL_PEND 1 1
# 	CPF_STAT_INVAL_PEND 2 2
# 	CPG_STAT_INVAL_PEND_CHANGED 3 3
# 	CPC_STAT_INVAL_PEND_CHANGED 4 4
# 	CPF_STAT_INVAL_PEND_CHANGED 5 5
regRLC_CSIB_ADDR_HI = 0x988
# 	ADDRESS 0 15
regRLC_CSIB_ADDR_LO = 0x987
# 	ADDRESS 0 31
regRLC_CSIB_LENGTH = 0x989
# 	LENGTH 0 31
regRLC_DS_RESIDENCY_CNTR_CTRL = 0x4d4a
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	RESERVED 5 31
regRLC_DS_RESIDENCY_EVENT_CNTR = 0x4d52
# 	DATA 0 31
regRLC_DS_RESIDENCY_REF_CNTR = 0x4d5a
# 	DATA 0 31
regRLC_DYN_PG_REQUEST = 0x4c4c
# 	PG_REQUEST_WGP_MASK 0 31
regRLC_DYN_PG_STATUS = 0x4c4b
# 	PG_STATUS_WGP_MASK 0 31
regRLC_F32_UCODE_VERSION = 0x4c03
# 	THREAD0_VERSION 0 9
# 	THREAD1_VERSION 10 19
# 	THREAD2_VERSION 20 29
regRLC_FWL_FIRST_VIOL_ADDR = 0x5f26
# 	VIOL_ADDR 0 17
# 	VIOL_APERTURE_ID 18 29
# 	VIOL_OP 30 30
# 	RESERVED 31 31
regRLC_GENERAL_RESIDENCY_CNTR_CTRL = 0x4d4d
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	RESERVED 5 31
regRLC_GENERAL_RESIDENCY_EVENT_CNTR = 0x4d55
# 	DATA 0 31
regRLC_GENERAL_RESIDENCY_REF_CNTR = 0x4d5d
# 	DATA 0 31
regRLC_GFX_IH_ARBITER_STAT = 0x4d5f
# 	CLIENT_GRANTED 0 15
# 	RESERVED 16 27
# 	LAST_CLIENT_GRANTED 28 31
regRLC_GFX_IH_CLIENT_CTRL = 0x4d5e
# 	SE_INTERRUPT_MASK 0 7
# 	SDMA_INTERRUPT_MASK 8 11
# 	UTCL2_INTERRUPT_MASK 12 12
# 	PMM_INTERRUPT_MASK 13 13
# 	RESERVED_15_14 14 15
# 	SE_INTERRUPT_ERROR_CLEAR 16 23
# 	SDMA_INTERRUPT_ERROR_CLEAR 24 27
# 	UTCL2_INTERRUPT_ERROR_CLEAR 28 28
# 	PMM_INTERRUPT_ERROR_CLEAR 29 29
# 	RESERVED_31_30 30 31
regRLC_GFX_IH_CLIENT_OTHER_STAT = 0x4d63
# 	UTCL2_BUFFER_LEVEL 0 3
# 	UTCL2_BUFFER_LOADING 4 4
# 	UTCL2_BUFFER_OVERFLOW 5 5
# 	UTCL2_PROTOCOL_ERROR 6 6
# 	UTCL2_RESERVED 7 7
# 	PMM_BUFFER_LEVEL 8 11
# 	PMM_BUFFER_LOADING 12 12
# 	PMM_BUFFER_OVERFLOW 13 13
# 	PMM_PROTOCOL_ERROR 14 14
# 	PMM_RESERVED 15 15
# 	RESERVED_31_16 16 31
regRLC_GFX_IH_CLIENT_SDMA_STAT = 0x4d62
# 	SDMA0_BUFFER_LEVEL 0 3
# 	SDMA0_BUFFER_LOADING 4 4
# 	SDMA0_BUFFER_OVERFLOW 5 5
# 	SDMA0_PROTOCOL_ERROR 6 6
# 	SDMA0_RESERVED 7 7
# 	SDMA1_BUFFER_LEVEL 8 11
# 	SDMA1_BUFFER_LOADING 12 12
# 	SDMA1_BUFFER_OVERFLOW 13 13
# 	SDMA1_PROTOCOL_ERROR 14 14
# 	SDMA1_RESERVED 15 15
# 	SDMA2_BUFFER_LEVEL 16 19
# 	SDMA2_BUFFER_LOADING 20 20
# 	SDMA2_BUFFER_OVERFLOW 21 21
# 	SDMA2_PROTOCOL_ERROR 22 22
# 	SDMA2_RESERVED 23 23
# 	SDMA3_BUFFER_LEVEL 24 27
# 	SDMA3_BUFFER_LOADING 28 28
# 	SDMA3_BUFFER_OVERFLOW 29 29
# 	SDMA3_PROTOCOL_ERROR 30 30
# 	SDMA3_RESERVED 31 31
regRLC_GFX_IH_CLIENT_SE_STAT_H = 0x4d61
# 	SE4_BUFFER_LEVEL 0 3
# 	SE4_BUFFER_LOADING 4 4
# 	SE4_BUFFER_OVERFLOW 5 5
# 	SE4_PROTOCOL_ERROR 6 6
# 	SE4_RESERVED 7 7
# 	SE5_BUFFER_LEVEL 8 11
# 	SE5_BUFFER_LOADING 12 12
# 	SE5_BUFFER_OVERFLOW 13 13
# 	SE5_PROTOCOL_ERROR 14 14
# 	SE5_RESERVED 15 15
# 	SE6_BUFFER_LEVEL 16 19
# 	SE6_BUFFER_LOADING 20 20
# 	SE6_BUFFER_OVERFLOW 21 21
# 	SE6_PROTOCOL_ERROR 22 22
# 	SE6_RESERVED 23 23
# 	SE7_BUFFER_LEVEL 24 27
# 	SE7_BUFFER_LOADING 28 28
# 	SE7_BUFFER_OVERFLOW 29 29
# 	SE7_PROTOCOL_ERROR 30 30
# 	SE7_RESERVED 31 31
regRLC_GFX_IH_CLIENT_SE_STAT_L = 0x4d60
# 	SE0_BUFFER_LEVEL 0 3
# 	SE0_BUFFER_LOADING 4 4
# 	SE0_BUFFER_OVERFLOW 5 5
# 	SE0_PROTOCOL_ERROR 6 6
# 	SE0_RESERVED 7 7
# 	SE1_BUFFER_LEVEL 8 11
# 	SE1_BUFFER_LOADING 12 12
# 	SE1_BUFFER_OVERFLOW 13 13
# 	SE1_PROTOCOL_ERROR 14 14
# 	SE1_RESERVED 15 15
# 	SE2_BUFFER_LEVEL 16 19
# 	SE2_BUFFER_LOADING 20 20
# 	SE2_BUFFER_OVERFLOW 21 21
# 	SE2_PROTOCOL_ERROR 22 22
# 	SE2_RESERVED 23 23
# 	SE3_BUFFER_LEVEL 24 27
# 	SE3_BUFFER_LOADING 28 28
# 	SE3_BUFFER_OVERFLOW 29 29
# 	SE3_PROTOCOL_ERROR 30 30
# 	SE3_RESERVED 31 31
regRLC_GFX_IMU_CMD = 0x4053
# 	CMD 0 31
regRLC_GFX_IMU_DATA_0 = 0x4052
# 	DATA 0 31
regRLC_GPM_CP_DMA_COMPLETE_T0 = 0x4c29
# 	DATA 0 0
# 	RESERVED 1 31
regRLC_GPM_CP_DMA_COMPLETE_T1 = 0x4c2a
# 	DATA 0 0
# 	RESERVED 1 31
regRLC_GPM_GENERAL_0 = 0x4c63
# 	DATA 0 31
regRLC_GPM_GENERAL_1 = 0x4c64
# 	DATA 0 31
regRLC_GPM_GENERAL_10 = 0x4caf
# 	DATA 0 31
regRLC_GPM_GENERAL_11 = 0x4cb0
# 	DATA 0 31
regRLC_GPM_GENERAL_12 = 0x4cb1
# 	DATA 0 31
regRLC_GPM_GENERAL_13 = 0x4cdd
# 	DATA 0 31
regRLC_GPM_GENERAL_14 = 0x4cde
# 	DATA 0 31
regRLC_GPM_GENERAL_15 = 0x4cdf
# 	DATA 0 31
regRLC_GPM_GENERAL_16 = 0x4c76
# 	DATA 0 31
regRLC_GPM_GENERAL_2 = 0x4c65
# 	DATA 0 31
regRLC_GPM_GENERAL_3 = 0x4c66
# 	DATA 0 31
regRLC_GPM_GENERAL_4 = 0x4c67
# 	DATA 0 31
regRLC_GPM_GENERAL_5 = 0x4c68
# 	DATA 0 31
regRLC_GPM_GENERAL_6 = 0x4c69
# 	DATA 0 31
regRLC_GPM_GENERAL_7 = 0x4c6a
# 	DATA 0 31
regRLC_GPM_GENERAL_8 = 0x4cad
# 	DATA 0 31
regRLC_GPM_GENERAL_9 = 0x4cae
# 	DATA 0 31
regRLC_GPM_INT_DISABLE_TH0 = 0x4c7c
# 	DISABLE_INT 0 31
regRLC_GPM_INT_FORCE_TH0 = 0x4c7e
# 	FORCE_INT 0 31
regRLC_GPM_INT_STAT_TH0 = 0x4cdc
# 	STATUS 0 31
regRLC_GPM_IRAM_ADDR = 0x5b62
# 	ADDR 0 31
regRLC_GPM_IRAM_DATA = 0x5b63
# 	DATA 0 31
regRLC_GPM_LEGACY_INT_CLEAR = 0x4c17
# 	SPP_PVT_INT_CHANGED 0 0
# 	CP_RLC_STAT_INVAL_PEND_CHANGED 1 1
# 	RLC_EOF_INT_CHANGED 2 2
# 	RLC_PG_CNTL_CHANGED 3 3
regRLC_GPM_LEGACY_INT_DISABLE = 0x4c7d
# 	SPP_PVT_INT_CHANGED 0 0
# 	CP_RLC_STAT_INVAL_PEND_CHANGED 1 1
# 	RLC_EOF_INT_CHANGED 2 2
# 	RLC_PG_CNTL_CHANGED 3 3
regRLC_GPM_LEGACY_INT_STAT = 0x4c16
# 	SPP_PVT_INT_CHANGED 0 0
# 	CP_RLC_STAT_INVAL_PEND_CHANGED 1 1
# 	RLC_EOF_INT_CHANGED 2 2
# 	RLC_PG_CNTL_CHANGED 3 3
regRLC_GPM_PERF_COUNT_0 = 0x2140
# 	FEATURE_SEL 0 3
# 	SE_INDEX 4 7
# 	SA_INDEX 8 11
# 	WGP_INDEX 12 15
# 	EVENT_SEL 16 17
# 	UNUSED 18 19
# 	ENABLE 20 20
# 	RESERVED 21 31
regRLC_GPM_PERF_COUNT_1 = 0x2141
# 	FEATURE_SEL 0 3
# 	SE_INDEX 4 7
# 	SA_INDEX 8 11
# 	WGP_INDEX 12 15
# 	EVENT_SEL 16 17
# 	UNUSED 18 19
# 	ENABLE 20 20
# 	RESERVED 21 31
regRLC_GPM_SCRATCH_ADDR = 0x5b6e
# 	ADDR 0 15
regRLC_GPM_SCRATCH_DATA = 0x5b6f
# 	DATA 0 31
regRLC_GPM_STAT = 0x4e6b
# 	RLC_BUSY 0 0
# 	GFX_POWER_STATUS 1 1
# 	GFX_CLOCK_STATUS 2 2
# 	GFX_LS_STATUS 3 3
# 	GFX_PIPELINE_POWER_STATUS 4 4
# 	CNTX_IDLE_BEING_PROCESSED 5 5
# 	CNTX_BUSY_BEING_PROCESSED 6 6
# 	GFX_IDLE_BEING_PROCESSED 7 7
# 	CMP_BUSY_BEING_PROCESSED 8 8
# 	SAVING_REGISTERS 9 9
# 	RESTORING_REGISTERS 10 10
# 	GFX3D_BLOCKS_CHANGING_POWER_STATE 11 11
# 	CMP_BLOCKS_CHANGING_POWER_STATE 12 12
# 	STATIC_WGP_POWERING_UP 13 13
# 	STATIC_WGP_POWERING_DOWN 14 14
# 	DYN_WGP_POWERING_UP 15 15
# 	DYN_WGP_POWERING_DOWN 16 16
# 	ABORTED_PD_SEQUENCE 17 17
# 	CMP_power_status 18 18
# 	GFX_LS_STATUS_3D 19 19
# 	GFX_CLOCK_STATUS_3D 20 20
# 	MGCG_OVERRIDE_STATUS 21 21
# 	RLC_EXEC_ROM_CODE 22 22
# 	FGCG_OVERRIDE_STATUS 23 23
# 	PG_ERROR_STATUS 24 31
regRLC_GPM_THREAD_ENABLE = 0x4c45
# 	THREAD0_ENABLE 0 0
# 	THREAD1_ENABLE 1 1
# 	THREAD2_ENABLE 2 2
# 	THREAD3_ENABLE 3 3
# 	RESERVED 4 31
regRLC_GPM_THREAD_INVALIDATE_CACHE = 0x4c2b
# 	THREAD0_INVALIDATE_CACHE 0 0
# 	THREAD1_INVALIDATE_CACHE 1 1
# 	THREAD2_INVALIDATE_CACHE 2 2
# 	THREAD3_INVALIDATE_CACHE 3 3
# 	RESERVED 4 31
regRLC_GPM_THREAD_PRIORITY = 0x4c44
# 	THREAD0_PRIORITY 0 7
# 	THREAD1_PRIORITY 8 15
# 	THREAD2_PRIORITY 16 23
# 	THREAD3_PRIORITY 24 31
regRLC_GPM_THREAD_RESET = 0x4c28
# 	THREAD0_RESET 0 0
# 	THREAD1_RESET 1 1
# 	THREAD2_RESET 2 2
# 	THREAD3_RESET 3 3
# 	RESERVED 4 31
regRLC_GPM_TIMER_CTRL = 0x4c13
# 	TIMER_0_EN 0 0
# 	TIMER_1_EN 1 1
# 	TIMER_2_EN 2 2
# 	TIMER_3_EN 3 3
# 	TIMER_4_EN 4 4
# 	RESERVED_1 5 7
# 	TIMER_0_AUTO_REARM 8 8
# 	TIMER_1_AUTO_REARM 9 9
# 	TIMER_2_AUTO_REARM 10 10
# 	TIMER_3_AUTO_REARM 11 11
# 	TIMER_4_AUTO_REARM 12 12
# 	RESERVED_2 13 15
# 	TIMER_0_INT_CLEAR 16 16
# 	TIMER_1_INT_CLEAR 17 17
# 	TIMER_2_INT_CLEAR 18 18
# 	TIMER_3_INT_CLEAR 19 19
# 	TIMER_4_INT_CLEAR 20 20
# 	RESERVED 21 31
regRLC_GPM_TIMER_INT_0 = 0x4c0e
# 	TIMER 0 31
regRLC_GPM_TIMER_INT_1 = 0x4c0f
# 	TIMER 0 31
regRLC_GPM_TIMER_INT_2 = 0x4c10
# 	TIMER 0 31
regRLC_GPM_TIMER_INT_3 = 0x4c11
# 	TIMER 0 31
regRLC_GPM_TIMER_INT_4 = 0x4c12
# 	TIMER 0 31
regRLC_GPM_TIMER_STAT = 0x4c14
# 	TIMER_0_STAT 0 0
# 	TIMER_1_STAT 1 1
# 	TIMER_2_STAT 2 2
# 	TIMER_3_STAT 3 3
# 	TIMER_4_STAT 4 4
# 	RESERVED_1 5 7
# 	TIMER_0_ENABLE_SYNC 8 8
# 	TIMER_1_ENABLE_SYNC 9 9
# 	TIMER_2_ENABLE_SYNC 10 10
# 	TIMER_3_ENABLE_SYNC 11 11
# 	TIMER_4_ENABLE_SYNC 12 12
# 	RESERVED_2 13 15
# 	TIMER_0_AUTO_REARM_SYNC 16 16
# 	TIMER_1_AUTO_REARM_SYNC 17 17
# 	TIMER_2_AUTO_REARM_SYNC 18 18
# 	TIMER_3_AUTO_REARM_SYNC 19 19
# 	TIMER_4_AUTO_REARM_SYNC 20 20
# 	RESERVED 21 31
regRLC_GPM_UCODE_ADDR = 0x5b60
# 	UCODE_ADDR 0 13
# 	RESERVED 14 31
regRLC_GPM_UCODE_DATA = 0x5b61
# 	UCODE_DATA 0 31
regRLC_GPM_UTCL1_CNTL_0 = 0x4cb2
# 	XNACK_REDO_TIMER_CNT 0 19
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	RESERVED 30 31
regRLC_GPM_UTCL1_CNTL_1 = 0x4cb3
# 	XNACK_REDO_TIMER_CNT 0 19
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	RESERVED 30 31
regRLC_GPM_UTCL1_CNTL_2 = 0x4cb4
# 	XNACK_REDO_TIMER_CNT 0 19
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	RESERVED 30 31
regRLC_GPM_UTCL1_TH0_ERROR_1 = 0x4cbe
# 	Translated_ReqError 0 1
# 	Translated_ReqErrorVmid 2 5
# 	Translated_ReqErrorAddr_MSB 6 9
regRLC_GPM_UTCL1_TH0_ERROR_2 = 0x4cc0
# 	Translated_ReqErrorAddr_LSB 0 31
regRLC_GPM_UTCL1_TH1_ERROR_1 = 0x4cc1
# 	Translated_ReqError 0 1
# 	Translated_ReqErrorVmid 2 5
# 	Translated_ReqErrorAddr_MSB 6 9
regRLC_GPM_UTCL1_TH1_ERROR_2 = 0x4cc2
# 	Translated_ReqErrorAddr_LSB 0 31
regRLC_GPM_UTCL1_TH2_ERROR_1 = 0x4cc3
# 	Translated_ReqError 0 1
# 	Translated_ReqErrorVmid 2 5
# 	Translated_ReqErrorAddr_MSB 6 9
regRLC_GPM_UTCL1_TH2_ERROR_2 = 0x4cc4
# 	Translated_ReqErrorAddr_LSB 0 31
regRLC_GPR_REG1 = 0x4c79
# 	DATA 0 31
regRLC_GPR_REG2 = 0x4c7a
# 	DATA 0 31
regRLC_GPU_CLOCK_32 = 0x4c42
# 	GPU_CLOCK_32 0 31
regRLC_GPU_CLOCK_32_RES_SEL = 0x4c41
# 	RES_SEL 0 5
# 	RESERVED 6 31
regRLC_GPU_CLOCK_COUNT_LSB = 0x4c24
# 	GPU_CLOCKS_LSB 0 31
regRLC_GPU_CLOCK_COUNT_LSB_1 = 0x4cfb
# 	GPU_CLOCKS_LSB 0 31
regRLC_GPU_CLOCK_COUNT_LSB_2 = 0x4ceb
# 	GPU_CLOCKS_LSB 0 31
regRLC_GPU_CLOCK_COUNT_MSB = 0x4c25
# 	GPU_CLOCKS_MSB 0 31
regRLC_GPU_CLOCK_COUNT_MSB_1 = 0x4cfc
# 	GPU_CLOCKS_MSB 0 31
regRLC_GPU_CLOCK_COUNT_MSB_2 = 0x4cec
# 	GPU_CLOCKS_MSB 0 31
regRLC_GPU_CLOCK_COUNT_SPM_LSB = 0x4de4
# 	GPU_CLOCKS_LSB 0 31
regRLC_GPU_CLOCK_COUNT_SPM_MSB = 0x4de5
# 	GPU_CLOCKS_MSB 0 31
regRLC_GPU_IOV_CFG_REG1 = 0x5b35
# 	CMD_TYPE 0 3
# 	CMD_EXECUTE 4 4
# 	CMD_EXECUTE_INTR_EN 5 5
# 	RESERVED 6 7
# 	FCN_ID 8 15
# 	NEXT_FCN_ID 16 23
# 	RESERVED1 24 31
regRLC_GPU_IOV_CFG_REG2 = 0x5b36
# 	CMD_STATUS 0 3
# 	RESERVED 4 31
regRLC_GPU_IOV_CFG_REG6 = 0x5b06
# 	CNTXT_SIZE 0 6
# 	CNTXT_LOCATION 7 7
# 	RESERVED 8 9
# 	CNTXT_OFFSET 10 31
regRLC_GPU_IOV_CFG_REG8 = 0x5b20
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_F32_CNTL = 0x5b46
# 	ENABLE 0 0
regRLC_GPU_IOV_F32_INVALIDATE_CACHE = 0x5b4b
# 	INVALIDATE_CACHE 0 0
regRLC_GPU_IOV_F32_RESET = 0x5b47
# 	RESET 0 0
regRLC_GPU_IOV_INT_DISABLE = 0x5b4e
# 	DISABLE_INT 0 31
regRLC_GPU_IOV_INT_FORCE = 0x5b4f
# 	FORCE_INT 0 31
regRLC_GPU_IOV_INT_STAT = 0x5b3f
# 	STATUS 0 31
regRLC_GPU_IOV_PERF_CNT_CNTL = 0x3cc3
# 	ENABLE 0 0
# 	MODE_SELECT 1 1
# 	RESET 2 2
# 	RESERVED 3 31
regRLC_GPU_IOV_PERF_CNT_RD_ADDR = 0x3cc6
# 	VFID 0 3
# 	CNT_ID 4 5
# 	RESERVED 6 31
regRLC_GPU_IOV_PERF_CNT_RD_DATA = 0x3cc7
# 	DATA 0 31
regRLC_GPU_IOV_PERF_CNT_WR_ADDR = 0x3cc4
# 	VFID 0 3
# 	CNT_ID 4 5
# 	RESERVED 6 31
regRLC_GPU_IOV_PERF_CNT_WR_DATA = 0x3cc5
# 	DATA 0 31
regRLC_GPU_IOV_RLC_RESPONSE = 0x5b4d
# 	RESP 0 31
regRLC_GPU_IOV_SCH_0 = 0x5b38
# 	ACTIVE_FUNCTIONS 0 31
regRLC_GPU_IOV_SCH_1 = 0x5b3b
# 	DATA 0 31
regRLC_GPU_IOV_SCH_2 = 0x5b3c
# 	DATA 0 31
regRLC_GPU_IOV_SCH_3 = 0x5b3a
# 	Time_Quanta_Def 0 31
regRLC_GPU_IOV_SCH_BLOCK = 0x5b34
# 	Sch_Block_ID 0 3
# 	Sch_Block_Ver 4 7
# 	Sch_Block_Size 8 15
# 	RESERVED 16 31
regRLC_GPU_IOV_SCRATCH_ADDR = 0x5b50
# 	ADDR 0 15
regRLC_GPU_IOV_SCRATCH_DATA = 0x5b51
# 	DATA 0 31
regRLC_GPU_IOV_SDMA0_BUSY_STATUS = 0x5bc8
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA0_STATUS = 0x5bc0
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA1_BUSY_STATUS = 0x5bc9
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA1_STATUS = 0x5bc1
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA2_BUSY_STATUS = 0x5bca
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA2_STATUS = 0x5bc2
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA3_BUSY_STATUS = 0x5bcb
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA3_STATUS = 0x5bc3
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA4_BUSY_STATUS = 0x5bcc
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA4_STATUS = 0x5bc4
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA5_BUSY_STATUS = 0x5bcd
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA5_STATUS = 0x5bc5
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA6_BUSY_STATUS = 0x5bce
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA6_STATUS = 0x5bc6
# 	STATUS 0 31
regRLC_GPU_IOV_SDMA7_BUSY_STATUS = 0x5bcf
# 	VM_BUSY_STATUS 0 31
regRLC_GPU_IOV_SDMA7_STATUS = 0x5bc7
# 	STATUS 0 31
regRLC_GPU_IOV_SMU_RESPONSE = 0x5b4a
# 	RESP 0 31
regRLC_GPU_IOV_UCODE_ADDR = 0x5b48
# 	UCODE_ADDR 0 11
# 	RESERVED 12 31
regRLC_GPU_IOV_UCODE_DATA = 0x5b49
# 	UCODE_DATA 0 31
regRLC_GPU_IOV_VF_DOORBELL_STATUS = 0x5b2a
# 	VF_DOORBELL_STATUS 0 30
# 	PF_DOORBELL_STATUS 31 31
regRLC_GPU_IOV_VF_DOORBELL_STATUS_CLR = 0x5b2c
# 	VF_DOORBELL_STATUS_CLR 0 30
# 	PF_DOORBELL_STATUS_CLR 31 31
regRLC_GPU_IOV_VF_DOORBELL_STATUS_SET = 0x5b2b
# 	VF_DOORBELL_STATUS_SET 0 30
# 	PF_DOORBELL_STATUS_SET 31 31
regRLC_GPU_IOV_VF_ENABLE = 0x5b00
# 	VF_ENABLE 0 0
# 	RESERVED 1 15
# 	VF_NUM 16 31
regRLC_GPU_IOV_VF_MASK = 0x5b2d
# 	VF_MASK 0 30
regRLC_GPU_IOV_VM_BUSY_STATUS = 0x5b37
# 	VM_BUSY_STATUS 0 31
regRLC_GTS_OFFSET_LSB = 0x5b79
# 	DATA 0 31
regRLC_GTS_OFFSET_MSB = 0x5b7a
# 	DATA 0 31
regRLC_HYP_RLCG_UCODE_CHKSUM = 0x5b43
# 	UCODE_CHKSUM 0 31
regRLC_HYP_RLCP_UCODE_CHKSUM = 0x5b44
# 	UCODE_CHKSUM 0 31
regRLC_HYP_RLCV_UCODE_CHKSUM = 0x5b45
# 	UCODE_CHKSUM 0 31
regRLC_HYP_SEMAPHORE_0 = 0x5b2e
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_HYP_SEMAPHORE_1 = 0x5b2f
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_HYP_SEMAPHORE_2 = 0x5b52
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_HYP_SEMAPHORE_3 = 0x5b53
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_IH_COOKIE = 0x5b41
# 	DATA 0 31
regRLC_IH_COOKIE_CNTL = 0x5b42
# 	CREDIT 0 1
# 	RESET_COUNTER 2 2
regRLC_IMU_BOOTLOAD_ADDR_HI = 0x4e10
# 	ADDR_HI 0 31
regRLC_IMU_BOOTLOAD_ADDR_LO = 0x4e11
# 	ADDR_LO 0 31
regRLC_IMU_BOOTLOAD_SIZE = 0x4e12
# 	SIZE 0 25
# 	RESERVED 26 31
regRLC_IMU_MISC = 0x4e16
# 	THROTTLE_GFX 0 0
# 	EARLY_MGCG 1 1
# 	RESERVED 2 31
regRLC_IMU_RESET_VECTOR = 0x4e17
# 	COLD_BOOT_EXIT 0 0
# 	VDDGFX_EXIT 1 1
# 	VECTOR 2 7
# 	RESERVED 8 31
regRLC_INT_STAT = 0x4c18
# 	LAST_CP_RLC_INT_ID 0 7
# 	CP_RLC_INT_PENDING 8 8
# 	RESERVED 9 31
regRLC_JUMP_TABLE_RESTORE = 0x4c1e
# 	ADDR 0 31
regRLC_LX6_CNTL = 0x4d80
# 	BRESET 0 0
# 	RUNSTALL 1 1
# 	PDEBUG_ENABLE 2 2
# 	STAT_VECTOR_SEL 3 3
regRLC_LX6_DRAM_ADDR = 0x5b68
# 	ADDR 0 10
regRLC_LX6_DRAM_DATA = 0x5b69
# 	DATA 0 31
regRLC_LX6_IRAM_ADDR = 0x5b6a
# 	ADDR 0 11
regRLC_LX6_IRAM_DATA = 0x5b6b
# 	DATA 0 31
regRLC_MAX_PG_WGP = 0x4c54
# 	MAX_POWERED_UP_WGP 0 7
# 	SPARE 8 31
regRLC_MEM_SLP_CNTL = 0x4e00
# 	RLC_MEM_LS_EN 0 0
# 	RLC_MEM_DS_EN 1 1
# 	RLC_SRM_MEM_LS_OVERRIDE 2 2
# 	RLC_SRM_MEM_DS_OVERRIDE 3 3
# 	RLC_SPM_MEM_LS_OVERRIDE 4 4
# 	RLC_SPM_MEM_DS_OVERRIDE 5 5
# 	RESERVED 6 6
# 	RLC_LS_DS_BUSY_OVERRIDE 7 7
# 	RLC_MEM_LS_ON_DELAY 8 15
# 	RLC_MEM_LS_OFF_DELAY 16 23
# 	RLC_SPP_MEM_LS_OVERRIDE 24 24
# 	RLC_SPP_MEM_DS_OVERRIDE 25 25
# 	RESERVED1 26 31
regRLC_MGCG_CTRL = 0x4c1a
# 	MGCG_EN 0 0
# 	SILICON_EN 1 1
# 	SIMULATION_EN 2 2
# 	ON_DELAY 3 6
# 	OFF_HYSTERESIS 7 14
# 	SPARE 15 31
regRLC_PACE_INT_CLEAR = 0x5b3e
# 	SMU_STRETCH_PCC_CLEAR 0 0
# 	SMU_PCC_CLEAR 1 1
regRLC_PACE_INT_DISABLE = 0x4ced
# 	DISABLE_INT 0 31
regRLC_PACE_INT_FORCE = 0x5b3d
# 	FORCE_INT 0 31
regRLC_PACE_INT_STAT = 0x4ccc
# 	STATUS 0 31
regRLC_PACE_SCRATCH_ADDR = 0x5b77
# 	ADDR 0 15
regRLC_PACE_SCRATCH_DATA = 0x5b78
# 	DATA 0 31
regRLC_PACE_SPARE_INT = 0x990
# 	INTERRUPT 0 0
# 	RESERVED 1 31
regRLC_PACE_SPARE_INT_1 = 0x991
# 	INTERRUPT 0 0
# 	RESERVED 1 31
regRLC_PACE_TIMER_CTRL = 0x4d06
# 	TIMER_0_EN 0 0
# 	TIMER_1_EN 1 1
# 	TIMER_0_AUTO_REARM 2 2
# 	TIMER_1_AUTO_REARM 3 3
# 	TIMER_0_INT_CLEAR 4 4
# 	TIMER_1_INT_CLEAR 5 5
# 	RESERVED 6 31
regRLC_PACE_TIMER_INT_0 = 0x4d04
# 	TIMER 0 31
regRLC_PACE_TIMER_INT_1 = 0x4d05
# 	TIMER 0 31
regRLC_PACE_TIMER_STAT = 0x5b33
# 	TIMER_0_STAT 0 0
# 	TIMER_1_STAT 1 1
# 	RESERVED 2 7
# 	TIMER_0_ENABLE_SYNC 8 8
# 	TIMER_1_ENABLE_SYNC 9 9
# 	TIMER_0_AUTO_REARM_SYNC 10 10
# 	TIMER_1_AUTO_REARM_SYNC 11 11
regRLC_PACE_UCODE_ADDR = 0x5b6c
# 	UCODE_ADDR 0 11
# 	RESERVED 12 31
regRLC_PACE_UCODE_DATA = 0x5b6d
# 	UCODE_DATA 0 31
regRLC_PCC_RESIDENCY_CNTR_CTRL = 0x4d4c
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	EVENT_SEL 5 8
# 	RESERVED 9 31
regRLC_PCC_RESIDENCY_EVENT_CNTR = 0x4d54
# 	DATA 0 31
regRLC_PCC_RESIDENCY_REF_CNTR = 0x4d5c
# 	DATA 0 31
regRLC_PERFCOUNTER0_HI = 0x3481
# 	PERFCOUNTER_HI 0 31
regRLC_PERFCOUNTER0_LO = 0x3480
# 	PERFCOUNTER_LO 0 31
regRLC_PERFCOUNTER0_SELECT = 0x3cc1
# 	PERFCOUNTER_SELECT 0 7
regRLC_PERFCOUNTER1_HI = 0x3483
# 	PERFCOUNTER_HI 0 31
regRLC_PERFCOUNTER1_LO = 0x3482
# 	PERFCOUNTER_LO 0 31
regRLC_PERFCOUNTER1_SELECT = 0x3cc2
# 	PERFCOUNTER_SELECT 0 7
regRLC_PERFMON_CNTL = 0x3cc0
# 	PERFMON_STATE 0 2
# 	PERFMON_SAMPLE_ENABLE 10 10
regRLC_PG_ALWAYS_ON_WGP_MASK = 0x4c53
# 	AON_WGP_MASK 0 31
regRLC_PG_CNTL = 0x4c43
# 	GFX_POWER_GATING_ENABLE 0 0
# 	GFX_POWER_GATING_SRC 1 1
# 	DYN_PER_WGP_PG_ENABLE 2 2
# 	STATIC_PER_WGP_PG_ENABLE 3 3
# 	GFX_PIPELINE_PG_ENABLE 4 4
# 	RESERVED 5 12
# 	MEM_DS_DISABLE 13 13
# 	PG_OVERRIDE 14 14
# 	CP_PG_DISABLE 15 15
# 	CHUB_HANDSHAKE_ENABLE 16 16
# 	SMU_CLK_SLOWDOWN_ON_PU_ENABLE 17 17
# 	SMU_CLK_SLOWDOWN_ON_PD_ENABLE 18 18
# 	RESERVED1 19 20
# 	Ultra_Low_Voltage_Enable 21 21
# 	RESERVED2 22 22
# 	SMU_HANDSHAKE_DISABLE 23 23
regRLC_PG_DELAY = 0x4c4d
# 	POWER_UP_DELAY 0 7
# 	POWER_DOWN_DELAY 8 15
# 	CMD_PROPAGATE_DELAY 16 23
# 	MEM_SLEEP_DELAY 24 31
regRLC_PG_DELAY_2 = 0x4c1f
# 	SERDES_TIMEOUT_VALUE 0 7
# 	SERDES_CMD_DELAY 8 15
# 	PERWGP_TIMEOUT_VALUE 16 31
regRLC_PG_DELAY_3 = 0x4c78
# 	CGCG_ACTIVE_BEFORE_CGPG 0 7
# 	RESERVED 8 31
regRLC_POWER_RESIDENCY_CNTR_CTRL = 0x4d48
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	RESERVED 5 31
regRLC_POWER_RESIDENCY_EVENT_CNTR = 0x4d50
# 	DATA 0 31
regRLC_POWER_RESIDENCY_REF_CNTR = 0x4d58
# 	DATA 0 31
regRLC_R2I_CNTL_0 = 0x4cd5
# 	Data 0 31
regRLC_R2I_CNTL_1 = 0x4cd6
# 	Data 0 31
regRLC_R2I_CNTL_2 = 0x4cd7
# 	Data 0 31
regRLC_R2I_CNTL_3 = 0x4cd8
# 	Data 0 31
regRLC_REFCLOCK_TIMESTAMP_LSB = 0x4c0c
# 	TIMESTAMP_LSB 0 31
regRLC_REFCLOCK_TIMESTAMP_MSB = 0x4c0d
# 	TIMESTAMP_MSB 0 31
regRLC_RLCG_DOORBELL_0_DATA_HI = 0x4c39
# 	DATA 0 31
regRLC_RLCG_DOORBELL_0_DATA_LO = 0x4c38
# 	DATA 0 31
regRLC_RLCG_DOORBELL_1_DATA_HI = 0x4c3b
# 	DATA 0 31
regRLC_RLCG_DOORBELL_1_DATA_LO = 0x4c3a
# 	DATA 0 31
regRLC_RLCG_DOORBELL_2_DATA_HI = 0x4c3d
# 	DATA 0 31
regRLC_RLCG_DOORBELL_2_DATA_LO = 0x4c3c
# 	DATA 0 31
regRLC_RLCG_DOORBELL_3_DATA_HI = 0x4c3f
# 	DATA 0 31
regRLC_RLCG_DOORBELL_3_DATA_LO = 0x4c3e
# 	DATA 0 31
regRLC_RLCG_DOORBELL_CNTL = 0x4c36
# 	DOORBELL_0_MODE 0 1
# 	DOORBELL_1_MODE 2 3
# 	DOORBELL_2_MODE 4 5
# 	DOORBELL_3_MODE 6 7
# 	DOORBELL_ID 16 20
# 	DOORBELL_ID_EN 21 21
# 	RESERVED 22 31
regRLC_RLCG_DOORBELL_RANGE = 0x4c47
# 	LOWER_ADDR_RESERVED 0 1
# 	LOWER_ADDR 2 11
# 	UPPER_ADDR_RESERVED 16 17
# 	UPPER_ADDR 18 27
regRLC_RLCG_DOORBELL_STAT = 0x4c37
# 	DOORBELL_0_VALID 0 0
# 	DOORBELL_1_VALID 1 1
# 	DOORBELL_2_VALID 2 2
# 	DOORBELL_3_VALID 3 3
regRLC_RLCP_DOORBELL_0_DATA_HI = 0x4d2a
# 	DATA 0 31
regRLC_RLCP_DOORBELL_0_DATA_LO = 0x4d29
# 	DATA 0 31
regRLC_RLCP_DOORBELL_1_DATA_HI = 0x4d2c
# 	DATA 0 31
regRLC_RLCP_DOORBELL_1_DATA_LO = 0x4d2b
# 	DATA 0 31
regRLC_RLCP_DOORBELL_2_DATA_HI = 0x4d2e
# 	DATA 0 31
regRLC_RLCP_DOORBELL_2_DATA_LO = 0x4d2d
# 	DATA 0 31
regRLC_RLCP_DOORBELL_3_DATA_HI = 0x4d30
# 	DATA 0 31
regRLC_RLCP_DOORBELL_3_DATA_LO = 0x4d2f
# 	DATA 0 31
regRLC_RLCP_DOORBELL_CNTL = 0x4d27
# 	DOORBELL_0_MODE 0 1
# 	DOORBELL_1_MODE 2 3
# 	DOORBELL_2_MODE 4 5
# 	DOORBELL_3_MODE 6 7
# 	DOORBELL_ID 16 20
# 	DOORBELL_ID_EN 21 21
regRLC_RLCP_DOORBELL_RANGE = 0x4d26
# 	LOWER_ADDR_RESERVED 0 1
# 	LOWER_ADDR 2 11
# 	UPPER_ADDR_RESERVED 16 17
# 	UPPER_ADDR 18 27
regRLC_RLCP_DOORBELL_STAT = 0x4d28
# 	DOORBELL_0_VALID 0 0
# 	DOORBELL_1_VALID 1 1
# 	DOORBELL_2_VALID 2 2
# 	DOORBELL_3_VALID 3 3
regRLC_RLCP_IRAM_ADDR = 0x5b64
# 	ADDR 0 31
regRLC_RLCP_IRAM_DATA = 0x5b65
# 	DATA 0 31
regRLC_RLCS_ABORTED_PD_SEQUENCE = 0x4e6c
# 	APS 0 15
# 	RESERVED 16 31
regRLC_RLCS_AUXILIARY_REG_1 = 0x4ec5
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_AUXILIARY_REG_2 = 0x4ec6
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_AUXILIARY_REG_3 = 0x4ec7
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_AUXILIARY_REG_4 = 0x4ec8
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_BOOTLOAD_ID_STATUS1 = 0x4ecb
# 	ID_0_LOADED 0 0
# 	ID_1_LOADED 1 1
# 	ID_2_LOADED 2 2
# 	ID_3_LOADED 3 3
# 	ID_4_LOADED 4 4
# 	ID_5_LOADED 5 5
# 	ID_6_LOADED 6 6
# 	ID_7_LOADED 7 7
# 	ID_8_LOADED 8 8
# 	ID_9_LOADED 9 9
# 	ID_10_LOADED 10 10
# 	ID_11_LOADED 11 11
# 	ID_12_LOADED 12 12
# 	ID_13_LOADED 13 13
# 	ID_14_LOADED 14 14
# 	ID_15_LOADED 15 15
# 	ID_16_LOADED 16 16
# 	ID_17_LOADED 17 17
# 	ID_18_LOADED 18 18
# 	ID_19_LOADED 19 19
# 	ID_20_LOADED 20 20
# 	ID_21_LOADED 21 21
# 	ID_22_LOADED 22 22
# 	ID_23_LOADED 23 23
# 	ID_24_LOADED 24 24
# 	ID_25_LOADED 25 25
# 	ID_26_LOADED 26 26
# 	ID_27_LOADED 27 27
# 	ID_28_LOADED 28 28
# 	ID_29_LOADED 29 29
# 	ID_30_LOADED 30 30
# 	ID_31_LOADED 31 31
regRLC_RLCS_BOOTLOAD_ID_STATUS2 = 0x4ecc
# 	ID_32_LOADED 0 0
# 	ID_33_LOADED 1 1
# 	ID_34_LOADED 2 2
# 	ID_35_LOADED 3 3
# 	ID_36_LOADED 4 4
# 	ID_37_LOADED 5 5
# 	ID_38_LOADED 6 6
# 	ID_39_LOADED 7 7
# 	ID_40_LOADED 8 8
# 	ID_41_LOADED 9 9
# 	ID_42_LOADED 10 10
# 	ID_43_LOADED 11 11
# 	ID_44_LOADED 12 12
# 	ID_45_LOADED 13 13
# 	ID_46_LOADED 14 14
# 	ID_47_LOADED 15 15
# 	ID_48_LOADED 16 16
# 	ID_49_LOADED 17 17
# 	ID_50_LOADED 18 18
# 	ID_51_LOADED 19 19
# 	ID_52_LOADED 20 20
# 	ID_53_LOADED 21 21
# 	ID_54_LOADED 22 22
# 	ID_55_LOADED 23 23
# 	ID_56_LOADED 24 24
# 	ID_57_LOADED 25 25
# 	ID_58_LOADED 26 26
# 	ID_59_LOADED 27 27
# 	ID_60_LOADED 28 28
# 	ID_61_LOADED 29 29
# 	ID_62_LOADED 30 30
# 	ID_63_LOADED 31 31
regRLC_RLCS_BOOTLOAD_STATUS = 0x4e82
# 	GFX_INIT_DONE 0 0
# 	RLC_GPM_IRAM_LOADED 3 3
# 	RLC_GPM_IRAM_DONE 4 4
# 	RESERVED 5 30
# 	BOOTLOAD_COMPLETE 31 31
regRLC_RLCS_CGCG_REQUEST = 0x4e66
# 	CGCG_REQUEST 0 0
# 	CGCG_REQUEST_3D 1 1
# 	RESERVED 2 31
regRLC_RLCS_CGCG_STATUS = 0x4e67
# 	CGCG_RAMP_STATUS 0 1
# 	GFX_CLK_STATUS 2 2
# 	CGCG_RAMP_STATUS_3D 3 4
# 	GFX_CLK_STATUS_3D 5 5
# 	RESERVED 6 31
regRLC_RLCS_CMP_IDLE_CNTL = 0x4e87
# 	INT_CLEAR 0 0
# 	CMP_IDLE_HYST 1 1
# 	CMP_IDLE 2 2
# 	MAX_HYSTERESIS 3 10
# 	HYSTERESIS_CNT 11 18
# 	RESERVED 19 31
regRLC_RLCS_CP_DMA_SRCID_OVER = 0x4eca
# 	SRCID_OVERRIDE 0 0
regRLC_RLCS_CP_INT_CTRL_1 = 0x4e7a
# 	INTERRUPT_ACK 0 0
# 	RESERVED 1 31
regRLC_RLCS_CP_INT_CTRL_2 = 0x4e7b
# 	IDLE_AUTO_ACK_EN 0 0
# 	BUSY_AUTO_ACK_EN 1 1
# 	IDLE_AUTO_ACK_ACTIVE 2 2
# 	BUSY_AUTO_ACK_ACTIVE 3 3
# 	INTERRUPT_PENDING 4 4
# 	RESERVED 5 31
regRLC_RLCS_CP_INT_INFO_1 = 0x4e7c
# 	INTERRUPT_INFO_1 0 31
regRLC_RLCS_CP_INT_INFO_2 = 0x4e7d
# 	INTERRUPT_INFO_2 0 15
# 	INTERRUPT_ID 16 24
# 	RESERVED 25 31
regRLC_RLCS_DEC_DUMP_ADDR = 0x4e61
regRLC_RLCS_DEC_END = 0x4fff
regRLC_RLCS_DEC_START = 0x4e60
regRLC_RLCS_DIDT_FORCE_STALL = 0x4e6d
# 	DFS 0 2
# 	VALID 3 3
# 	RESERVED 4 31
regRLC_RLCS_DSM_TRIG = 0x4e81
# 	START 0 0
# 	RESERVED 1 31
regRLC_RLCS_EDC_INT_CNTL = 0x4ece
# 	EDC_EVENT_INT_CLEAR 0 0
regRLC_RLCS_EXCEPTION_REG_1 = 0x4e62
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_EXCEPTION_REG_2 = 0x4e63
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_EXCEPTION_REG_3 = 0x4e64
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_EXCEPTION_REG_4 = 0x4e65
# 	ADDR 0 17
# 	RESERVED 18 31
regRLC_RLCS_GCR_DATA_0 = 0x4ed4
# 	PHASE_0 0 15
# 	PHASE_1 16 31
regRLC_RLCS_GCR_DATA_1 = 0x4ed5
# 	PHASE_2 0 15
# 	PHASE_3 16 31
regRLC_RLCS_GCR_DATA_2 = 0x4ed6
# 	PHASE_4 0 15
# 	PHASE_5 16 31
regRLC_RLCS_GCR_DATA_3 = 0x4ed7
# 	PHASE_6 0 15
# 	PHASE_7 16 31
regRLC_RLCS_GCR_STATUS = 0x4ed8
# 	GCR_BUSY 0 0
# 	GCR_OUT_COUNT 1 4
# 	RESERVED_2 5 7
# 	GCRIU_CLI_RSP_TAG 8 15
# 	RESERVED 16 31
regRLC_RLCS_GENERAL_0 = 0x4e88
# 	DATA 0 31
regRLC_RLCS_GENERAL_1 = 0x4e89
# 	DATA 0 31
regRLC_RLCS_GENERAL_10 = 0x4e92
# 	DATA 0 31
regRLC_RLCS_GENERAL_11 = 0x4e93
# 	DATA 0 31
regRLC_RLCS_GENERAL_12 = 0x4e94
# 	DATA 0 31
regRLC_RLCS_GENERAL_13 = 0x4e95
# 	DATA 0 31
regRLC_RLCS_GENERAL_14 = 0x4e96
# 	DATA 0 31
regRLC_RLCS_GENERAL_15 = 0x4e97
# 	DATA 0 31
regRLC_RLCS_GENERAL_16 = 0x4e98
# 	DATA 0 31
regRLC_RLCS_GENERAL_2 = 0x4e8a
# 	DATA 0 31
regRLC_RLCS_GENERAL_3 = 0x4e8b
# 	DATA 0 31
regRLC_RLCS_GENERAL_4 = 0x4e8c
# 	DATA 0 31
regRLC_RLCS_GENERAL_5 = 0x4e8d
# 	DATA 0 31
regRLC_RLCS_GENERAL_6 = 0x4e8e
# 	DATA 0 31
regRLC_RLCS_GENERAL_7 = 0x4e8f
# 	DATA 0 31
regRLC_RLCS_GENERAL_8 = 0x4e90
# 	DATA 0 31
regRLC_RLCS_GENERAL_9 = 0x4e91
# 	DATA 0 31
regRLC_RLCS_GFX_DS_ALLOW_MASK_CNTL = 0x4e6a
regRLC_RLCS_GFX_DS_CNTL = 0x4e69
# 	GFX_CLK_DS_ALLOW 0 0
# 	GFX_CLK_DS_RLC_BUSY_MASK 1 1
# 	GFX_CLK_DS_CP_BUSY_MASK 2 2
# 	GFX_CLK_DS_GFX_PWR_STALLED_MASK 6 6
# 	GFX_CLK_DS_NON3D_PWR_STALLED_MASK 7 7
# 	GFX_CLK_DS_IMU_DISABLE_MASK 8 8
# 	GFX_CLK_DS_SDMA_0_BUSY_MASK 16 16
# 	GFX_CLK_DS_SDMA_1_BUSY_MASK 17 17
# 	GFX_CLK_DS_SDMA_2_BUSY_MASK 18 18
# 	GFX_CLK_DS_SDMA_3_BUSY_MASK 19 19
# 	GFX_CLK_DS_SDMA_4_BUSY_MASK 20 20
# 	GFX_CLK_DS_SDMA_5_BUSY_MASK 21 21
# 	GFX_CLK_DS_SDMA_6_BUSY_MASK 22 22
# 	GFX_CLK_DS_SDMA_7_BUSY_MASK 23 23
regRLC_RLCS_GFX_MEM_POWER_CTRL_LO = 0x4ef8
# 	DATA 0 31
regRLC_RLCS_GFX_RM_CNTL = 0x4efa
# 	RLC_GFX_RM_VALID 0 0
# 	RESERVED 1 31
regRLC_RLCS_GPM_LEGACY_INT_DISABLE = 0x4ed2
# 	GC_CAC_EDC_EVENT_CHANGED 0 0
# 	GFX_POWER_BRAKE_CHANGED 1 1
regRLC_RLCS_GPM_LEGACY_INT_STAT = 0x4ed1
# 	GC_CAC_EDC_EVENT_CHANGED 0 0
# 	GFX_POWER_BRAKE_CHANGED 1 1
regRLC_RLCS_GPM_STAT = 0x4e6b
# 	RLC_BUSY 0 0
# 	GFX_POWER_STATUS 1 1
# 	GFX_CLOCK_STATUS 2 2
# 	GFX_LS_STATUS 3 3
# 	GFX_PIPELINE_POWER_STATUS 4 4
# 	CNTX_IDLE_BEING_PROCESSED 5 5
# 	CNTX_BUSY_BEING_PROCESSED 6 6
# 	GFX_IDLE_BEING_PROCESSED 7 7
# 	CMP_BUSY_BEING_PROCESSED 8 8
# 	SAVING_REGISTERS 9 9
# 	RESTORING_REGISTERS 10 10
# 	GFX3D_BLOCKS_CHANGING_POWER_STATE 11 11
# 	CMP_BLOCKS_CHANGING_POWER_STATE 12 12
# 	STATIC_WGP_POWERING_UP 13 13
# 	STATIC_WGP_POWERING_DOWN 14 14
# 	DYN_WGP_POWERING_UP 15 15
# 	DYN_WGP_POWERING_DOWN 16 16
# 	ABORTED_PD_SEQUENCE 17 17
# 	CMP_POWER_STATUS 18 18
# 	GFX_LS_STATUS_3D 19 19
# 	GFX_CLOCK_STATUS_3D 20 20
# 	MGCG_OVERRIDE_STATUS 21 21
# 	RLC_EXEC_ROM_CODE 22 22
# 	FGCG_OVERRIDE_STATUS 23 23
# 	PG_ERROR_STATUS 24 31
regRLC_RLCS_GPM_STAT_2 = 0x4e72
# 	TC_TRANS_ERROR 0 0
# 	RLC_PWR_NON3D_STALLED 1 1
# 	GFX_PWR_STALLED_STATUS 2 2
# 	GFX_ULV_STATUS 3 3
# 	GFX_GENERAL_STATUS 4 4
# 	RESERVED 5 31
regRLC_RLCS_GRBM_IDLE_BUSY_INT_CNTL = 0x4e86
# 	SDMA0_BUSY_INT_CLEAR 0 0
# 	SDMA1_BUSY_INT_CLEAR 1 1
# 	SDMA2_BUSY_INT_CLEAR 2 2
# 	SDMA3_BUSY_INT_CLEAR 3 3
# 	SDMA4_BUSY_INT_CLEAR 4 4
# 	SDMA5_BUSY_INT_CLEAR 5 5
# 	SDMA6_BUSY_INT_CLEAR 6 6
# 	SDMA7_BUSY_INT_CLEAR 7 7
regRLC_RLCS_GRBM_IDLE_BUSY_STAT = 0x4e85
# 	GRBM_RLC_GC_STAT_IDLE 0 1
# 	SDMA_0_BUSY 16 16
# 	SDMA_1_BUSY 17 17
# 	SDMA_2_BUSY 18 18
# 	SDMA_3_BUSY 19 19
# 	SDMA_4_BUSY 20 20
# 	SDMA_5_BUSY 21 21
# 	SDMA_6_BUSY 22 22
# 	SDMA_7_BUSY 23 23
# 	SDMA_0_BUSY_CHANGED 24 24
# 	SDMA_1_BUSY_CHANGED 25 25
# 	SDMA_2_BUSY_CHANGED 26 26
# 	SDMA_3_BUSY_CHANGED 27 27
# 	SDMA_4_BUSY_CHANGED 28 28
# 	SDMA_5_BUSY_CHANGED 29 29
# 	SDMA_6_BUSY_CHANGED 30 30
# 	SDMA_7_BUSY_CHANGED 31 31
regRLC_RLCS_GRBM_SOFT_RESET = 0x4e73
# 	RESET 0 0
# 	RESERVED 1 31
regRLC_RLCS_IH_COOKIE_SEMAPHORE = 0x4e77
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_RLCS_IH_SEMAPHORE = 0x4e76
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_RLCS_IMU_GFX_DOORBELL_FENCE = 0x4ef1
# 	ENABLE 0 0
# 	ACK 1 1
# 	RESERVED 2 31
regRLC_RLCS_IMU_RAM_ADDR_0_LSB = 0x4eee
# 	DATA 0 31
regRLC_RLCS_IMU_RAM_ADDR_0_MSB = 0x4eef
# 	DATA 0 15
# 	RESERVED 16 31
regRLC_RLCS_IMU_RAM_ADDR_1_LSB = 0x4eeb
# 	DATA 0 31
regRLC_RLCS_IMU_RAM_ADDR_1_MSB = 0x4eec
# 	DATA 0 15
# 	RESERVED 16 31
regRLC_RLCS_IMU_RAM_CNTL = 0x4ef0
# 	REQTOG 0 0
# 	ACKTOG 1 1
# 	RESERVED 2 31
regRLC_RLCS_IMU_RAM_DATA_0 = 0x4eed
# 	DATA 0 31
regRLC_RLCS_IMU_RAM_DATA_1 = 0x4eea
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_CNTL = 0x4ee1
# 	DONETOG 0 0
# 	CHGTOG 1 1
# 	RESERVED 2 31
regRLC_RLCS_IMU_RLC_MSG_CONTROL = 0x4ee0
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_DATA0 = 0x4edb
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_DATA1 = 0x4edc
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_DATA2 = 0x4edd
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_DATA3 = 0x4ede
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MSG_DATA4 = 0x4edf
# 	DATA 0 31
regRLC_RLCS_IMU_RLC_MUTEX_CNTL = 0x4ee7
# 	REQ 0 0
# 	ACQUIRE 1 1
# 	RESERVED 2 31
regRLC_RLCS_IMU_RLC_STATUS = 0x4ee8
# 	ALLOW_GFXOFF 0 0
# 	ALLOW_FA_DCS 1 1
# 	RESERVED_14_2 2 14
# 	DISABLE_GFXCLK_DS 15 15
# 	RESERVED 16 31
regRLC_RLCS_IMU_RLC_TELEMETRY_DATA_0 = 0x4ee5
# 	CURRENT 0 15
# 	VOLTAGE 16 31
regRLC_RLCS_IMU_RLC_TELEMETRY_DATA_1 = 0x4ee6
# 	TEMPERATURE1 0 15
# 	RESERVED 16 31
regRLC_RLCS_IMU_VIDCHG_CNTL = 0x4ecd
# 	REQ 0 0
# 	DATA 1 9
# 	PSIEN 10 10
# 	ACK 11 11
# 	RESERVED 12 31
regRLC_RLCS_IOV_CMD_STATUS = 0x4e6e
# 	DATA 0 31
regRLC_RLCS_IOV_CNTX_LOC_SIZE = 0x4e6f
# 	DATA 0 7
# 	RESERVED 8 31
regRLC_RLCS_IOV_SCH_BLOCK = 0x4e70
# 	DATA 0 31
regRLC_RLCS_IOV_VM_BUSY_STATUS = 0x4e71
# 	DATA 0 31
regRLC_RLCS_KMD_LOG_CNTL1 = 0x4ecf
# 	DATA 0 31
regRLC_RLCS_KMD_LOG_CNTL2 = 0x4ed0
# 	DATA 0 31
regRLC_RLCS_PERFMON_CLK_CNTL_UCODE = 0x4ed9
# 	PERFMON_CLOCK_STATE 0 0
regRLC_RLCS_PG_CHANGE_READ = 0x4e75
# 	RESERVED 0 0
# 	PG_REG_CHANGED 1 1
# 	DYN_PG_STATUS_CHANGED 2 2
# 	DYN_PG_REQ_CHANGED 3 3
regRLC_RLCS_PG_CHANGE_STATUS = 0x4e74
# 	PG_CNTL_CHANGED 0 0
# 	PG_REG_CHANGED 1 1
# 	DYN_PG_STATUS_CHANGED 2 2
# 	DYN_PG_REQ_CHANGED 3 3
# 	RESERVED 4 31
regRLC_RLCS_PMM_CGCG_CNTL = 0x4ef7
# 	VALID 0 0
# 	CLEAN 1 1
# 	RESERVED 2 31
regRLC_RLCS_POWER_BRAKE_CNTL = 0x4e83
# 	POWER_BRAKE 0 0
# 	INT_CLEAR 1 1
# 	MAX_HYSTERESIS 2 9
# 	HYSTERESIS_CNT 10 17
# 	RESERVED 18 31
regRLC_RLCS_POWER_BRAKE_CNTL_TH1 = 0x4e84
# 	POWER_BRAKE 0 0
# 	INT_CLEAR 1 1
# 	MAX_HYSTERESIS 2 9
# 	HYSTERESIS_CNT 10 17
# 	RESERVED 18 31
regRLC_RLCS_RLC_IMU_MSG_CNTL = 0x4ee4
# 	CHGTOG 0 0
# 	DONETOG 1 1
# 	RESERVED 2 31
regRLC_RLCS_RLC_IMU_MSG_CONTROL = 0x4ee3
# 	DATA 0 31
regRLC_RLCS_RLC_IMU_MSG_DATA0 = 0x4ee2
# 	DATA 0 31
regRLC_RLCS_RLC_IMU_STATUS = 0x4ee9
# 	PWR_DOWN_ACTIVE 0 0
# 	RLC_ALIVE 1 1
# 	RESERVED_3_2 2 3
# 	RESERVED 4 31
regRLC_RLCS_SDMA_INT_CNTL_1 = 0x4ef3
# 	INTERRUPT_ACK 0 0
# 	RESP_ID 1 1
# 	RESERVED 2 31
regRLC_RLCS_SDMA_INT_CNTL_2 = 0x4ef4
# 	AUTO_ACK_EN 0 0
# 	AUTO_ACK_ACTIVE 1 1
# 	RESERVED 2 31
regRLC_RLCS_SDMA_INT_INFO = 0x4ef6
# 	REQ_IDLE_TO_FW 0 7
# 	REQ_BUSY_TO_FW 8 15
# 	INTERRUPT_ID 16 16
# 	RESERVED 17 31
regRLC_RLCS_SDMA_INT_STAT = 0x4ef5
# 	REQ_IDLE_HIST 0 7
# 	REQ_BUSY_HIST 8 15
# 	LAST_SDMA_RLC_INT_ID 16 16
# 	SDMA_RLC_INT_PENDING 17 17
# 	RESERVED 18 31
regRLC_RLCS_SOC_DS_CNTL = 0x4e68
# 	SOC_CLK_DS_ALLOW 0 0
# 	SOC_CLK_DS_RLC_BUSY_MASK 1 1
# 	SOC_CLK_DS_CP_BUSY_MASK 2 2
# 	SOC_CLK_DS_GFX_PWR_STALLED_MASK 6 6
# 	SOC_CLK_DS_NON3D_PWR_STALLED_MASK 7 7
# 	SOC_CLK_DS_SDMA_0_BUSY_MASK 16 16
# 	SOC_CLK_DS_SDMA_1_BUSY_MASK 17 17
# 	SOC_CLK_DS_SDMA_2_BUSY_MASK 18 18
# 	SOC_CLK_DS_SDMA_3_BUSY_MASK 19 19
# 	SOC_CLK_DS_SDMA_4_BUSY_MASK 20 20
# 	SOC_CLK_DS_SDMA_5_BUSY_MASK 21 21
# 	SOC_CLK_DS_SDMA_6_BUSY_MASK 22 22
# 	SOC_CLK_DS_SDMA_7_BUSY_MASK 23 23
regRLC_RLCS_SPM_INT_CTRL = 0x4e7e
# 	INTERRUPT_ACK 0 0
# 	RESERVED 1 31
regRLC_RLCS_SPM_INT_INFO_1 = 0x4e7f
# 	INTERRUPT_INFO_1 0 31
regRLC_RLCS_SPM_INT_INFO_2 = 0x4e80
# 	INTERRUPT_INFO_2 0 15
# 	INTERRUPT_ID 16 24
# 	RESERVED 25 31
regRLC_RLCS_SPM_SQTT_MODE = 0x4ec9
# 	MODE 0 0
regRLC_RLCS_SRM_SRCID_CNTL = 0x4ed3
# 	SRCID 0 2
regRLC_RLCS_UTCL2_CNTL = 0x4eda
# 	MTYPE_NO_PTE_MODE 0 0
# 	GPA_OVERRIDE 1 1
# 	VF_OVERRIDE 2 2
# 	GPA_OVERRIDE_VALUE 3 4
# 	VF_OVERRIDE_VALUE 5 5
# 	IGNORE_PTE_PERMISSION 6 6
# 	RESERVED 7 31
regRLC_RLCS_WGP_READ = 0x4e79
# 	CS_WORK_ACTIVE 0 0
# 	STATIC_WGP_STATUS_CHANGED 1 1
# 	DYMANIC_WGP_STATUS_CHANGED 2 2
# 	RESERVED 3 31
regRLC_RLCS_WGP_STATUS = 0x4e78
# 	CS_WORK_ACTIVE 0 0
# 	STATIC_WGP_STATUS_CHANGED 1 1
# 	DYMANIC_WGP_STATUS_CHANGED 2 2
# 	STATIC_PERWGP_PD_INCOMPLETE 3 3
# 	RESERVED 4 31
regRLC_RLCV_COMMAND = 0x4e04
# 	CMD 0 3
# 	RESERVED 4 31
regRLC_RLCV_DOORBELL_0_DATA_HI = 0x4cf4
# 	DATA 0 31
regRLC_RLCV_DOORBELL_0_DATA_LO = 0x4cf3
# 	DATA 0 31
regRLC_RLCV_DOORBELL_1_DATA_HI = 0x4cf6
# 	DATA 0 31
regRLC_RLCV_DOORBELL_1_DATA_LO = 0x4cf5
# 	DATA 0 31
regRLC_RLCV_DOORBELL_2_DATA_HI = 0x4cf8
# 	DATA 0 31
regRLC_RLCV_DOORBELL_2_DATA_LO = 0x4cf7
# 	DATA 0 31
regRLC_RLCV_DOORBELL_3_DATA_HI = 0x4cfa
# 	DATA 0 31
regRLC_RLCV_DOORBELL_3_DATA_LO = 0x4cf9
# 	DATA 0 31
regRLC_RLCV_DOORBELL_CNTL = 0x4cf1
# 	DOORBELL_0_MODE 0 1
# 	DOORBELL_1_MODE 2 3
# 	DOORBELL_2_MODE 4 5
# 	DOORBELL_3_MODE 6 7
# 	DOORBELL_ID 16 20
# 	DOORBELL_ID_EN 21 21
regRLC_RLCV_DOORBELL_RANGE = 0x4cf0
# 	LOWER_ADDR_RESERVED 0 1
# 	LOWER_ADDR 2 11
# 	UPPER_ADDR_RESERVED 16 17
# 	UPPER_ADDR 18 27
regRLC_RLCV_DOORBELL_STAT = 0x4cf2
# 	DOORBELL_0_VALID 0 0
# 	DOORBELL_1_VALID 1 1
# 	DOORBELL_2_VALID 2 2
# 	DOORBELL_3_VALID 3 3
regRLC_RLCV_IRAM_ADDR = 0x5b66
# 	ADDR 0 31
regRLC_RLCV_IRAM_DATA = 0x5b67
# 	DATA 0 31
regRLC_RLCV_SAFE_MODE = 0x4e02
# 	CMD 0 0
# 	MESSAGE 1 4
# 	RESERVED1 5 7
# 	RESPONSE 8 11
# 	RESERVED 12 31
regRLC_RLCV_SPARE_INT = 0x4d00
# 	INTERRUPT 0 0
# 	RESERVED 1 31
regRLC_RLCV_SPARE_INT_1 = 0x992
# 	INTERRUPT 0 0
# 	RESERVED 1 31
regRLC_RLCV_TIMER_CTRL = 0x5b27
# 	TIMER_0_EN 0 0
# 	TIMER_1_EN 1 1
# 	TIMER_0_AUTO_REARM 2 2
# 	TIMER_1_AUTO_REARM 3 3
# 	TIMER_0_INT_CLEAR 4 4
# 	TIMER_1_INT_CLEAR 5 5
# 	RESERVED 6 31
regRLC_RLCV_TIMER_INT_0 = 0x5b25
# 	TIMER 0 31
regRLC_RLCV_TIMER_INT_1 = 0x5b26
# 	TIMER 0 31
regRLC_RLCV_TIMER_STAT = 0x5b28
# 	TIMER_0_STAT 0 0
# 	TIMER_1_STAT 1 1
# 	RESERVED 2 7
# 	TIMER_0_ENABLE_SYNC 8 8
# 	TIMER_1_ENABLE_SYNC 9 9
# 	TIMER_0_AUTO_REARM_SYNC 10 10
# 	TIMER_1_AUTO_REARM_SYNC 11 11
regRLC_SAFE_MODE = 0x980
# 	CMD 0 0
# 	MESSAGE 1 4
# 	RESERVED1 5 7
# 	RESPONSE 8 11
# 	RESERVED 12 31
regRLC_SDMA0_BUSY_STATUS = 0x5b1c
# 	BUSY_STATUS 0 31
regRLC_SDMA0_STATUS = 0x5b18
# 	STATUS 0 31
regRLC_SDMA1_BUSY_STATUS = 0x5b1d
# 	BUSY_STATUS 0 31
regRLC_SDMA1_STATUS = 0x5b19
# 	STATUS 0 31
regRLC_SDMA2_BUSY_STATUS = 0x5b1e
# 	BUSY_STATUS 0 31
regRLC_SDMA2_STATUS = 0x5b1a
# 	STATUS 0 31
regRLC_SDMA3_BUSY_STATUS = 0x5b1f
# 	BUSY_STATUS 0 31
regRLC_SDMA3_STATUS = 0x5b1b
# 	STATUS 0 31
regRLC_SEMAPHORE_0 = 0x4cc7
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_SEMAPHORE_1 = 0x4cc8
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_SEMAPHORE_2 = 0x4cc9
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_SEMAPHORE_3 = 0x4cca
# 	CLIENT_ID 0 4
# 	RESERVED 5 31
regRLC_SERDES_BUSY = 0x4c61
# 	GC_CENTER_HUB_0 0 0
# 	GC_CENTER_HUB_1 1 1
# 	RESERVED 2 15
# 	GC_SE_0 16 16
# 	GC_SE_1 17 17
# 	GC_SE_2 18 18
# 	GC_SE_3 19 19
# 	GC_SE_4 20 20
# 	GC_SE_5 21 21
# 	GC_SE_6 22 22
# 	GC_SE_7 23 23
# 	RESERVED_29_24 24 29
# 	RD_FIFO_NOT_EMPTY 30 30
# 	RD_PENDING 31 31
regRLC_SERDES_CTRL = 0x4c5f
# 	BPM_BROADCAST 0 0
# 	BPM_REG_WRITE 1 1
# 	BPM_LONG_CMD 2 2
# 	BPM_ADDR 3 15
# 	REG_ADDR 16 23
regRLC_SERDES_DATA = 0x4c60
# 	DATA 0 31
regRLC_SERDES_MASK = 0x4c5e
# 	GC_CENTER_HUB_0 0 0
# 	GC_CENTER_HUB_1 1 1
# 	RESERVED 2 15
# 	GC_SE_0 16 16
# 	GC_SE_1 17 17
# 	GC_SE_2 18 18
# 	GC_SE_3 19 19
# 	GC_SE_4 20 20
# 	GC_SE_5 21 21
# 	GC_SE_6 22 22
# 	GC_SE_7 23 23
# 	RESERVED_31_24 24 31
regRLC_SERDES_RD_DATA_0 = 0x4c5a
# 	DATA 0 31
regRLC_SERDES_RD_DATA_1 = 0x4c5b
# 	DATA 0 31
regRLC_SERDES_RD_DATA_2 = 0x4c5c
# 	DATA 0 31
regRLC_SERDES_RD_DATA_3 = 0x4c5d
# 	DATA 0 31
regRLC_SERDES_RD_INDEX = 0x4c59
# 	DATA_REG_ID 0 1
# 	SPARE 2 31
regRLC_SMU_ARGUMENT_1 = 0x4e0b
# 	ARG 0 31
regRLC_SMU_ARGUMENT_2 = 0x4e0c
# 	ARG 0 31
regRLC_SMU_ARGUMENT_3 = 0x4e0d
# 	ARG 0 31
regRLC_SMU_ARGUMENT_4 = 0x4e0e
# 	ARG 0 31
regRLC_SMU_ARGUMENT_5 = 0x4e0f
# 	ARG 0 31
regRLC_SMU_CLK_REQ = 0x4d08
# 	VALID 0 0
regRLC_SMU_COMMAND = 0x4e0a
# 	CMD 0 31
regRLC_SMU_MESSAGE = 0x4e05
# 	CMD 0 31
regRLC_SMU_MESSAGE_1 = 0x4e06
# 	CMD 0 31
regRLC_SMU_MESSAGE_2 = 0x4e07
# 	CMD 0 31
regRLC_SMU_SAFE_MODE = 0x4e03
# 	CMD 0 0
# 	MESSAGE 1 4
# 	RESERVED1 5 7
# 	RESPONSE 8 11
# 	RESERVED 12 31
regRLC_SPARE = 0x4d0b
# 	SPARE 0 31
regRLC_SPARE_INT_0 = 0x98d
# 	DATA 0 29
# 	PROCESSING 30 30
# 	COMPLETE 31 31
regRLC_SPARE_INT_1 = 0x98e
# 	DATA 0 29
# 	PROCESSING 30 30
# 	COMPLETE 31 31
regRLC_SPARE_INT_2 = 0x98f
# 	DATA 0 29
# 	PROCESSING 30 30
# 	COMPLETE 31 31
regRLC_SPM_ACCUM_CTRL = 0x3c9a
# 	StrobeResetPerfMonitors 0 0
# 	StrobeStartAccumulation 1 1
# 	StrobeRearmAccum 2 2
# 	StrobeResetSpmBlock 3 3
# 	StrobeStartSpm 4 7
# 	StrobeRearmSwaAccum 8 8
# 	StrobeStartSwa 9 9
# 	StrobePerfmonSampleWires 10 10
# 	RESERVED 11 31
regRLC_SPM_ACCUM_CTRLRAM_ADDR = 0x3c96
# 	addr 0 10
# 	RESERVED 11 31
regRLC_SPM_ACCUM_CTRLRAM_ADDR_OFFSET = 0x3c98
# 	global_offset 0 7
# 	spmwithaccum_se_offset 8 15
# 	spmwithaccum_global_offset 16 23
# 	RESERVED 24 31
regRLC_SPM_ACCUM_CTRLRAM_DATA = 0x3c97
# 	data 0 7
# 	RESERVED 8 31
regRLC_SPM_ACCUM_DATARAM_32BITCNTRS_REGIONS = 0x3c9f
# 	spp_addr_region 0 7
# 	swa_addr_region 8 15
# 	RESERVED 16 31
regRLC_SPM_ACCUM_DATARAM_ADDR = 0x3c92
# 	addr 0 6
# 	RESERVED 7 31
regRLC_SPM_ACCUM_DATARAM_DATA = 0x3c93
# 	data 0 31
regRLC_SPM_ACCUM_DATARAM_WRCOUNT = 0x3c9e
# 	DataRamWrCount 0 18
# 	RESERVED 19 31
regRLC_SPM_ACCUM_MODE = 0x3c9b
# 	EnableAccum 0 0
# 	EnableSpmWithAccumMode 1 1
# 	EnableSPPMode 2 2
# 	AutoResetPerfmonDisable 3 3
# 	AutoAccumEn 5 5
# 	SwaAutoAccumEn 6 6
# 	AutoSpmEn 7 7
# 	SwaAutoSpmEn 8 8
# 	Globals_LoadOverride 9 9
# 	Globals_SwaLoadOverride 10 10
# 	SE0_LoadOverride 11 11
# 	SE0_SwaLoadOverride 12 12
# 	SE1_LoadOverride 13 13
# 	SE1_SwaLoadOverride 14 14
# 	SE2_LoadOverride 15 15
# 	SE2_SwaLoadOverride 16 16
# 	SE3_LoadOverride 17 17
# 	SE3_SwaLoadOverride 18 18
# 	SE4_LoadOverride 19 19
# 	SE4_SwaLoadOverride 20 20
# 	SE5_LoadOverride 21 21
# 	SE5_SwaLoadOverride 22 22
regRLC_SPM_ACCUM_SAMPLES_REQUESTED = 0x3c9d
# 	SamplesRequested 0 7
regRLC_SPM_ACCUM_STATUS = 0x3c99
# 	NumbSamplesCompleted 0 7
# 	AccumDone 8 8
# 	SpmDone 9 9
# 	AccumOverflow 10 10
# 	AccumArmed 11 11
# 	SequenceInProgress 12 12
# 	FinalSequenceInProgress 13 13
# 	AllFifosEmpty 14 14
# 	FSMIsIdle 15 15
# 	SwaAccumDone 16 16
# 	SwaSpmDone 17 17
# 	SwaAccumOverflow 18 18
# 	SwaAccumArmed 19 19
# 	AllSegsDone 20 20
# 	RearmSwaPending 21 21
# 	RearmSppPending 22 22
# 	MultiSampleAborted 23 23
# 	RESERVED 24 31
regRLC_SPM_ACCUM_SWA_DATARAM_ADDR = 0x3c94
# 	addr 0 6
# 	RESERVED 7 31
regRLC_SPM_ACCUM_SWA_DATARAM_DATA = 0x3c95
# 	data 0 31
regRLC_SPM_ACCUM_THRESHOLD = 0x3c9c
# 	Threshold 0 15
regRLC_SPM_GFXCLOCK_HIGHCOUNT = 0x3ca5
# 	GFXCLOCK_HIGHCOUNT 0 31
regRLC_SPM_GFXCLOCK_LOWCOUNT = 0x3ca4
# 	GFXCLOCK_LOWCOUNT 0 31
regRLC_SPM_GLOBAL_DELAY_IND_ADDR = 0x4d64
# 	ADDR 0 11
regRLC_SPM_GLOBAL_DELAY_IND_DATA = 0x4d65
# 	DATA 0 5
regRLC_SPM_GLOBAL_MUXSEL_ADDR = 0x3c88
# 	ADDR 0 11
regRLC_SPM_GLOBAL_MUXSEL_DATA = 0x3c89
# 	SEL0 0 15
# 	SEL1 16 31
regRLC_SPM_INT_CNTL = 0x983
# 	RLC_SPM_INT_CNTL 0 0
# 	RESERVED 1 31
regRLC_SPM_INT_INFO_1 = 0x985
# 	INTERRUPT_INFO_1 0 31
regRLC_SPM_INT_INFO_2 = 0x986
# 	INTERRUPT_INFO_2 0 15
# 	INTERRUPT_ID 16 23
# 	RESERVED 24 31
regRLC_SPM_INT_STATUS = 0x984
# 	RLC_SPM_INT_STATUS 0 0
# 	RESERVED 1 31
regRLC_SPM_MC_CNTL = 0x982
# 	RLC_SPM_VMID 0 3
# 	RLC_SPM_POLICY 4 5
# 	RLC_SPM_PERF_CNTR 6 6
# 	RLC_SPM_FED 7 7
# 	RLC_SPM_MTYPE_OVER 8 8
# 	RLC_SPM_MTYPE 9 11
# 	RLC_SPM_BC 12 12
# 	RLC_SPM_RO 13 13
# 	RLC_SPM_VOL 14 14
# 	RLC_SPM_NOFILL 15 15
# 	RESERVED_3 16 17
# 	RLC_SPM_LLC_NOALLOC 18 18
# 	RLC_SPM_LLC_NOALLOC_OVER 19 19
# 	RESERVED 20 31
regRLC_SPM_MODE = 0x3cad
# 	MODE 0 0
regRLC_SPM_PAUSE = 0x3ca2
# 	PAUSE 0 0
# 	PAUSED 1 1
regRLC_SPM_PERFMON_CNTL = 0x3c80
# 	RESERVED1 0 11
# 	PERFMON_RING_MODE 12 13
# 	DISABLE_GFXCLOCK_COUNT 14 14
# 	RESERVED 15 15
# 	PERFMON_SAMPLE_INTERVAL 16 31
regRLC_SPM_PERFMON_RING_BASE_HI = 0x3c82
# 	RING_BASE_HI 0 15
# 	RESERVED 16 31
regRLC_SPM_PERFMON_RING_BASE_LO = 0x3c81
# 	RING_BASE_LO 0 31
regRLC_SPM_PERFMON_RING_SIZE = 0x3c83
# 	RING_BASE_SIZE 0 31
regRLC_SPM_PERFMON_SEGMENT_SIZE = 0x3c87
# 	TOTAL_NUM_SEGMENT 0 15
# 	GLOBAL_NUM_SEGMENT 16 23
# 	SE_NUM_SEGMENT 24 31
regRLC_SPM_RING_RDPTR = 0x3c85
# 	PERFMON_RING_RDPTR 0 31
regRLC_SPM_RING_WRPTR = 0x3c84
# 	RESERVED 0 4
# 	PERFMON_RING_WRPTR 5 31
regRLC_SPM_RSPM_CMD = 0x3cb8
# 	CMD 0 3
regRLC_SPM_RSPM_CMD_ACK = 0x3cb9
# 	SE0_ACK 0 0
# 	SE1_ACK 1 1
# 	SE2_ACK 2 2
# 	SE3_ACK 3 3
# 	SE4_ACK 4 4
# 	SE5_ACK 5 5
# 	SE6_ACK 6 6
# 	SE7_ACK 7 7
# 	SPM_ACK 8 8
regRLC_SPM_RSPM_REQ_DATA_HI = 0x3caf
# 	DATA 0 11
regRLC_SPM_RSPM_REQ_DATA_LO = 0x3cae
# 	DATA 0 31
regRLC_SPM_RSPM_REQ_OP = 0x3cb0
# 	OP 0 3
regRLC_SPM_RSPM_RET_DATA = 0x3cb1
# 	DATA 0 31
regRLC_SPM_RSPM_RET_OP = 0x3cb2
# 	OP 0 3
# 	VALID 8 8
regRLC_SPM_SAMPLE_CNT = 0x981
# 	COUNT 0 31
regRLC_SPM_SEGMENT_THRESHOLD = 0x3c86
# 	NUM_SEGMENT_THRESHOLD 0 7
# 	RESERVED 8 31
regRLC_SPM_SE_DELAY_IND_ADDR = 0x4d66
# 	ADDR 0 11
regRLC_SPM_SE_DELAY_IND_DATA = 0x4d67
# 	DATA 0 5
regRLC_SPM_SE_MUXSEL_ADDR = 0x3c8a
# 	ADDR 0 11
regRLC_SPM_SE_MUXSEL_DATA = 0x3c8b
# 	SEL0 0 15
# 	SEL1 16 31
regRLC_SPM_SE_RSPM_REQ_DATA_HI = 0x3cb4
# 	DATA 0 11
regRLC_SPM_SE_RSPM_REQ_DATA_LO = 0x3cb3
# 	DATA 0 31
regRLC_SPM_SE_RSPM_REQ_OP = 0x3cb5
# 	OP 0 3
regRLC_SPM_SE_RSPM_RET_DATA = 0x3cb6
# 	DATA 0 31
regRLC_SPM_SE_RSPM_RET_OP = 0x3cb7
# 	OP 0 3
# 	VALID 8 8
regRLC_SPM_SPARE = 0x3cbf
# 	SPARE 0 31
regRLC_SPM_STATUS = 0x3ca3
# 	CTL_BUSY 0 0
# 	RSPM_REG_BUSY 1 1
# 	SPM_RSPM_BUSY 2 2
# 	SPM_RSPM_IO_BUSY 3 3
# 	SE_RSPM_IO_BUSY 4 11
# 	ACCUM_BUSY 15 15
# 	FSM_MASTER_STATE 16 19
# 	FSM_MEMORY_STATE 20 23
# 	CTL_REQ_STATE 24 25
# 	CTL_RET_STATE 26 26
regRLC_SPM_THREAD_TRACE_CTRL = 0x4de6
# 	THREAD_TRACE_INT_EN 0 0
regRLC_SPM_UTCL1_CNTL = 0x4cb5
# 	XNACK_REDO_TIMER_CNT 0 19
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	RESERVED 30 31
regRLC_SPM_UTCL1_ERROR_1 = 0x4cbc
# 	Translated_ReqError 0 1
# 	Translated_ReqErrorVmid 2 5
# 	Translated_ReqErrorAddr_MSB 6 9
regRLC_SPM_UTCL1_ERROR_2 = 0x4cbd
# 	Translated_ReqErrorAddr_LSB 0 31
regRLC_SPP_CAM_ADDR = 0x4de8
# 	ADDR 0 7
regRLC_SPP_CAM_DATA = 0x4de9
# 	DATA 0 7
# 	TAG 8 31
regRLC_SPP_CAM_EXT_ADDR = 0x4dea
# 	ADDR 0 7
regRLC_SPP_CAM_EXT_DATA = 0x4deb
# 	VALID 0 0
# 	LOCK 1 1
regRLC_SPP_CTRL = 0x4d0c
# 	ENABLE 0 0
# 	ENABLE_PPROF 1 1
# 	ENABLE_PWR_OPT 2 2
# 	PAUSE 3 3
regRLC_SPP_GLOBAL_SH_ID = 0x4d1a
# 	SH_ID 0 31
regRLC_SPP_GLOBAL_SH_ID_VALID = 0x4d1b
# 	VALID 0 0
regRLC_SPP_INFLIGHT_RD_ADDR = 0x4d12
# 	ADDR 0 4
regRLC_SPP_INFLIGHT_RD_DATA = 0x4d13
# 	DATA 0 31
regRLC_SPP_PBB_INFO = 0x4d23
# 	PIPE0_OVERRIDE 0 0
# 	PIPE0_OVERRIDE_VALID 1 1
# 	PIPE1_OVERRIDE 2 2
# 	PIPE1_OVERRIDE_VALID 3 3
regRLC_SPP_PROF_INFO_1 = 0x4d18
# 	SH_ID 0 31
regRLC_SPP_PROF_INFO_2 = 0x4d19
# 	SH_TYPE 0 3
# 	CAM_HIT 4 4
# 	CAM_LOCK 5 5
# 	CAM_CONFLICT 6 6
regRLC_SPP_PVT_LEVEL_MAX = 0x4d21
# 	LEVEL 0 3
regRLC_SPP_PVT_STAT_0 = 0x4d1d
# 	LEVEL_0_COUNTER 0 5
# 	LEVEL_1_COUNTER 6 11
# 	LEVEL_2_COUNTER 12 17
# 	LEVEL_3_COUNTER 18 23
# 	LEVEL_4_COUNTER 24 30
regRLC_SPP_PVT_STAT_1 = 0x4d1e
# 	LEVEL_5_COUNTER 0 5
# 	LEVEL_6_COUNTER 6 11
# 	LEVEL_7_COUNTER 12 17
# 	LEVEL_8_COUNTER 18 23
# 	LEVEL_9_COUNTER 24 30
regRLC_SPP_PVT_STAT_2 = 0x4d1f
# 	LEVEL_10_COUNTER 0 5
# 	LEVEL_11_COUNTER 6 11
# 	LEVEL_12_COUNTER 12 17
# 	LEVEL_13_COUNTER 18 23
# 	LEVEL_14_COUNTER 24 30
regRLC_SPP_PVT_STAT_3 = 0x4d20
# 	LEVEL_15_COUNTER 0 5
regRLC_SPP_RESET = 0x4d24
# 	SSF_RESET 0 0
# 	EVENT_ARB_RESET 1 1
# 	CAM_RESET 2 2
# 	PVT_RESET 3 3
regRLC_SPP_SHADER_PROFILE_EN = 0x4d0d
# 	PS_ENABLE 0 0
# 	RESERVED_1 1 1
# 	GS_ENABLE 2 2
# 	HS_ENABLE 3 3
# 	CSG_ENABLE 4 4
# 	CS_ENABLE 5 5
# 	PS_STOP_CONDITION 6 6
# 	RESERVED_7 7 7
# 	GS_STOP_CONDITION 8 8
# 	HS_STOP_CONDITION 9 9
# 	CSG_STOP_CONDITION 10 10
# 	CS_STOP_CONDITION 11 11
# 	PS_START_CONDITION 12 12
# 	CS_START_CONDITION 13 13
# 	FORCE_MISS 14 14
# 	FORCE_UNLOCKED 15 15
# 	ENABLE_PROF_INFO_LOCK 16 16
regRLC_SPP_SSF_CAPTURE_EN = 0x4d0e
# 	PS_ENABLE 0 0
# 	RESERVED_1 1 1
# 	GS_ENABLE 2 2
# 	HS_ENABLE 3 3
# 	CSG_ENABLE 4 4
# 	CS_ENABLE 5 5
regRLC_SPP_SSF_THRESHOLD_0 = 0x4d0f
# 	PS_THRESHOLD 0 15
# 	RESERVED 16 31
regRLC_SPP_SSF_THRESHOLD_1 = 0x4d10
# 	GS_THRESHOLD 0 15
# 	HS_THRESHOLD 16 31
regRLC_SPP_SSF_THRESHOLD_2 = 0x4d11
# 	CSG_THRESHOLD 0 15
# 	CS_THRESHOLD 16 31
regRLC_SPP_STALL_STATE_UPDATE = 0x4d22
# 	STALL 0 0
# 	ENABLE 1 1
regRLC_SPP_STATUS = 0x4d1c
# 	RESERVED_0 0 0
# 	SSF_BUSY 1 1
# 	EVENT_ARB_BUSY 2 2
# 	SPP_BUSY 31 31
regRLC_SRM_ARAM_ADDR = 0x5b73
# 	ADDR 0 12
# 	RESERVED 13 31
regRLC_SRM_ARAM_DATA = 0x5b74
# 	DATA 0 31
regRLC_SRM_CNTL = 0x4c80
# 	SRM_ENABLE 0 0
# 	AUTO_INCR_ADDR 1 1
# 	RESERVED 2 31
regRLC_SRM_DRAM_ADDR = 0x5b71
# 	ADDR 0 12
# 	RESERVED 13 31
regRLC_SRM_DRAM_DATA = 0x5b72
# 	DATA 0 31
regRLC_SRM_GPM_ABORT = 0x4e09
# 	ABORT 0 0
# 	RESERVED 1 31
regRLC_SRM_GPM_COMMAND = 0x4e08
# 	OP 0 0
# 	INDEX_CNTL 1 1
# 	INDEX_CNTL_NUM 2 4
# 	SIZE 5 17
# 	START_OFFSET 18 30
# 	DEST_MEMORY 31 31
regRLC_SRM_GPM_COMMAND_STATUS = 0x4c88
# 	FIFO_EMPTY 0 0
# 	FIFO_FULL 1 1
# 	RESERVED 2 31
regRLC_SRM_INDEX_CNTL_ADDR_0 = 0x4c8b
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_1 = 0x4c8c
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_2 = 0x4c8d
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_3 = 0x4c8e
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_4 = 0x4c8f
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_5 = 0x4c90
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_6 = 0x4c91
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_ADDR_7 = 0x4c92
# 	ADDRESS 0 17
regRLC_SRM_INDEX_CNTL_DATA_0 = 0x4c93
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_1 = 0x4c94
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_2 = 0x4c95
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_3 = 0x4c96
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_4 = 0x4c97
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_5 = 0x4c98
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_6 = 0x4c99
# 	DATA 0 31
regRLC_SRM_INDEX_CNTL_DATA_7 = 0x4c9a
# 	DATA 0 31
regRLC_SRM_STAT = 0x4c9b
# 	SRM_BUSY 0 0
# 	SRM_BUSY_DELAY 1 1
# 	RESERVED 2 31
regRLC_STAT = 0x4c04
# 	RLC_BUSY 0 0
# 	RLC_SRM_BUSY 1 1
# 	RLC_GPM_BUSY 2 2
# 	RLC_SPM_BUSY 3 3
# 	MC_BUSY 4 4
# 	RLC_THREAD_0_BUSY 5 5
# 	RLC_THREAD_1_BUSY 6 6
# 	RLC_THREAD_2_BUSY 7 7
# 	RESERVED 8 31
regRLC_STATIC_PG_STATUS = 0x4c6e
# 	PG_STATUS_WGP_MASK 0 31
regRLC_UCODE_CNTL = 0x4c27
# 	RLC_UCODE_FLAGS 0 31
regRLC_ULV_RESIDENCY_CNTR_CTRL = 0x4d4b
# 	RESET 0 0
# 	ENABLE 1 1
# 	RESET_ACK 2 2
# 	ENABLE_ACK 3 3
# 	COUNTER_OVERFLOW 4 4
# 	RESERVED 5 31
regRLC_ULV_RESIDENCY_EVENT_CNTR = 0x4d53
# 	DATA 0 31
regRLC_ULV_RESIDENCY_REF_CNTR = 0x4d5b
# 	DATA 0 31
regRLC_UTCL1_STATUS = 0x4cd4
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	RESERVED 3 7
# 	FAULT_UTCL1ID 8 13
# 	RESERVED_1 14 15
# 	RETRY_UTCL1ID 16 21
# 	RESERVED_2 22 23
# 	PRT_UTCL1ID 24 29
# 	RESERVED_3 30 31
regRLC_UTCL1_STATUS_2 = 0x4cb6
# 	GPM_TH0_UTCL1_BUSY 0 0
# 	GPM_TH1_UTCL1_BUSY 1 1
# 	GPM_TH2_UTCL1_BUSY 2 2
# 	SPM_UTCL1_BUSY 3 3
# 	RESERVED_1 4 4
# 	GPM_TH0_UTCL1_StallOnTrans 5 5
# 	GPM_TH1_UTCL1_StallOnTrans 6 6
# 	GPM_TH2_UTCL1_StallOnTrans 7 7
# 	SPM_UTCL1_StallOnTrans 8 8
# 	RESERVED 9 31
regRLC_WGP_STATUS = 0x4c4e
# 	WORK_PENDING 0 31
regRLC_XT_CORE_ALT_RESET_VEC = 0x4dd7
# 	ALT_RESET_VEC 0 31
regRLC_XT_CORE_FAULT_INFO = 0x4dd6
# 	FAULT_INFO 0 31
regRLC_XT_CORE_INTERRUPT = 0x4dd5
# 	EXTINT1 0 25
# 	EXTINT2 26 26
# 	NMI 27 27
regRLC_XT_CORE_RESERVED = 0x4dd8
# 	RESERVED 0 31
regRLC_XT_CORE_STATUS = 0x4dd4
# 	P_WAIT_MODE 0 0
# 	P_FATAL_ERROR 1 1
# 	DOUBLE_EXCEPTION_ERROR 2 2
regRLC_XT_DOORBELL_0_DATA_HI = 0x4df9
# 	DATA 0 31
regRLC_XT_DOORBELL_0_DATA_LO = 0x4df8
# 	DATA 0 31
regRLC_XT_DOORBELL_1_DATA_HI = 0x4dfb
# 	DATA 0 31
regRLC_XT_DOORBELL_1_DATA_LO = 0x4dfa
# 	DATA 0 31
regRLC_XT_DOORBELL_2_DATA_HI = 0x4dfd
# 	DATA 0 31
regRLC_XT_DOORBELL_2_DATA_LO = 0x4dfc
# 	DATA 0 31
regRLC_XT_DOORBELL_3_DATA_HI = 0x4dff
# 	DATA 0 31
regRLC_XT_DOORBELL_3_DATA_LO = 0x4dfe
# 	DATA 0 31
regRLC_XT_DOORBELL_CNTL = 0x4df6
# 	DOORBELL_0_MODE 0 1
# 	DOORBELL_1_MODE 2 3
# 	DOORBELL_2_MODE 4 5
# 	DOORBELL_3_MODE 6 7
# 	DOORBELL_ID 16 20
# 	DOORBELL_ID_EN 21 21
regRLC_XT_DOORBELL_RANGE = 0x4df5
# 	LOWER_ADDR_RESERVED 0 1
# 	LOWER_ADDR 2 11
# 	UPPER_ADDR_RESERVED 16 17
# 	UPPER_ADDR 18 27
regRLC_XT_DOORBELL_STAT = 0x4df7
# 	DOORBELL_0_VALID 0 0
# 	DOORBELL_1_VALID 1 1
# 	DOORBELL_2_VALID 2 2
# 	DOORBELL_3_VALID 3 3
regRLC_XT_INT_VEC_CLEAR = 0x4dda
# 	NUM_0 0 0
# 	NUM_1 1 1
# 	NUM_2 2 2
# 	NUM_3 3 3
# 	NUM_4 4 4
# 	NUM_5 5 5
# 	NUM_6 6 6
# 	NUM_7 7 7
# 	NUM_8 8 8
# 	NUM_9 9 9
# 	NUM_10 10 10
# 	NUM_11 11 11
# 	NUM_12 12 12
# 	NUM_13 13 13
# 	NUM_14 14 14
# 	NUM_15 15 15
# 	NUM_16 16 16
# 	NUM_17 17 17
# 	NUM_18 18 18
# 	NUM_19 19 19
# 	NUM_20 20 20
# 	NUM_21 21 21
# 	NUM_22 22 22
# 	NUM_23 23 23
# 	NUM_24 24 24
# 	NUM_25 25 25
regRLC_XT_INT_VEC_FORCE = 0x4dd9
# 	NUM_0 0 0
# 	NUM_1 1 1
# 	NUM_2 2 2
# 	NUM_3 3 3
# 	NUM_4 4 4
# 	NUM_5 5 5
# 	NUM_6 6 6
# 	NUM_7 7 7
# 	NUM_8 8 8
# 	NUM_9 9 9
# 	NUM_10 10 10
# 	NUM_11 11 11
# 	NUM_12 12 12
# 	NUM_13 13 13
# 	NUM_14 14 14
# 	NUM_15 15 15
# 	NUM_16 16 16
# 	NUM_17 17 17
# 	NUM_18 18 18
# 	NUM_19 19 19
# 	NUM_20 20 20
# 	NUM_21 21 21
# 	NUM_22 22 22
# 	NUM_23 23 23
# 	NUM_24 24 24
# 	NUM_25 25 25
regRLC_XT_INT_VEC_MUX_INT_SEL = 0x4ddc
# 	INT_SEL 0 5
regRLC_XT_INT_VEC_MUX_SEL = 0x4ddb
# 	MUX_SEL 0 4
regRMI_CLOCK_CNTRL = 0x1896
# 	DYN_CLK_RB0_BUSY_MASK 0 4
# 	DYN_CLK_CMN_BUSY_MASK 5 9
# 	DYN_CLK_RB0_WAKEUP_MASK 10 14
# 	DYN_CLK_CMN_WAKEUP_MASK 15 19
regRMI_DEMUX_CNTL = 0x188a
# 	DEMUX_ARB0_MODE_OVERRIDE_EN 2 2
# 	DEMUX_ARB0_STALL_TIMER_START_VALUE 6 13
# 	DEMUX_ARB0_MODE 14 15
# 	DEMUX_ARB1_MODE_OVERRIDE_EN 18 18
# 	DEMUX_ARB1_STALL_TIMER_START_VALUE 22 29
# 	DEMUX_ARB1_MODE 30 31
regRMI_GENERAL_CNTL = 0x1880
# 	BURST_DISABLE 0 0
# 	VMID_BYPASS_ENABLE 1 16
# 	RB0_HARVEST_EN 19 19
# 	LOOPBACK_DIS_BY_REQ_TYPE 21 24
regRMI_GENERAL_CNTL1 = 0x1881
# 	EARLY_WRACK_ENABLE_PER_MTYPE 0 3
# 	TCIW0_64B_RD_STALL_MODE 4 5
# 	TCIW1_64B_RD_STALL_MODE 6 7
# 	EARLY_WRACK_DISABLE_FOR_LOOPBACK 8 8
# 	POLICY_OVERRIDE_VALUE 9 10
# 	POLICY_OVERRIDE 11 11
# 	ARBITER_ADDRESS_CHANGE_ENABLE 14 14
# 	LAST_OF_BURST_INSERTION_DISABLE 15 15
# 	TCIW0_PRODUCER_CREDITS 16 21
# 	TCIW1_PRODUCER_CREDITS 22 27
regRMI_GENERAL_STATUS = 0x1882
# 	GENERAL_RMI_ERRORS_COMBINED 0 0
# 	SKID_FIFO_0_OVERFLOW_ERROR 1 1
# 	SKID_FIFO_0_UNDERFLOW_ERROR 2 2
# 	SKID_FIFO_1_OVERFLOW_ERROR 3 3
# 	SKID_FIFO_1_UNDERFLOW_ERROR 4 4
# 	RMI_XBAR_BUSY 5 5
# 	RESERVED_BIT_6 6 6
# 	RMI_SCOREBOARD_BUSY 7 7
# 	TCIW0_PRT_FIFO_BUSY 8 8
# 	TCIW_FRMTR0_BUSY 9 9
# 	TCIW_RTN_FRMTR0_BUSY 10 10
# 	WRREQ_CONSUMER_FIFO_0_BUSY 11 11
# 	RDREQ_CONSUMER_FIFO_0_BUSY 12 12
# 	TCIW1_PRT_FIFO_BUSY 13 13
# 	TCIW_FRMTR1_BUSY 14 14
# 	TCIW_RTN_FRMTR1_BUSY 15 15
# 	RESERVED_BIT_18 18 18
# 	RESERVED_BIT_19 19 19
# 	RESERVED_BIT_20 20 20
# 	RESERVED_BITS_28_21 21 28
# 	RESERVED_BIT_29 29 29
# 	RESERVED_BIT_30 30 30
# 	SKID_FIFO_FREESPACE_IS_ZERO_ERROR 31 31
regRMI_PERFCOUNTER0_HI = 0x34c1
# 	PERFCOUNTER_HI 0 31
regRMI_PERFCOUNTER0_LO = 0x34c0
# 	PERFCOUNTER_LO 0 31
regRMI_PERFCOUNTER0_SELECT = 0x3d00
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regRMI_PERFCOUNTER0_SELECT1 = 0x3d01
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regRMI_PERFCOUNTER1_HI = 0x34c3
# 	PERFCOUNTER_HI 0 31
regRMI_PERFCOUNTER1_LO = 0x34c2
# 	PERFCOUNTER_LO 0 31
regRMI_PERFCOUNTER1_SELECT = 0x3d02
# 	PERF_SEL 0 9
# 	PERF_MODE 28 31
regRMI_PERFCOUNTER2_HI = 0x34c5
# 	PERFCOUNTER_HI 0 31
regRMI_PERFCOUNTER2_LO = 0x34c4
# 	PERFCOUNTER_LO 0 31
regRMI_PERFCOUNTER2_SELECT = 0x3d03
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regRMI_PERFCOUNTER2_SELECT1 = 0x3d04
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regRMI_PERFCOUNTER3_HI = 0x34c7
# 	PERFCOUNTER_HI 0 31
regRMI_PERFCOUNTER3_LO = 0x34c6
# 	PERFCOUNTER_LO 0 31
regRMI_PERFCOUNTER3_SELECT = 0x3d05
# 	PERF_SEL 0 9
# 	PERF_MODE 28 31
regRMI_PERF_COUNTER_CNTL = 0x3d06
# 	TRANS_BASED_PERF_EN_SEL 0 1
# 	EVENT_BASED_PERF_EN_SEL 2 3
# 	TC_PERF_EN_SEL 4 5
# 	PERF_EVENT_WINDOW_MASK0 6 7
# 	PERF_EVENT_WINDOW_MASK1 8 9
# 	PERF_COUNTER_CID 10 13
# 	PERF_COUNTER_VMID 14 18
# 	PERF_COUNTER_BURST_LENGTH_THRESHOLD 19 24
# 	PERF_SOFT_RESET 25 25
# 	PERF_CNTR_SPM_SEL 26 26
regRMI_PROBE_POP_LOGIC_CNTL = 0x1888
# 	EXT_LAT_FIFO_0_MAX_DEPTH 0 6
# 	XLAT_COMBINE0_DIS 7 7
# 	REDUCE_MAX_XLAT_CHAIN_SIZE_BY_2 8 9
# 	EXT_LAT_FIFO_1_MAX_DEPTH 10 16
# 	XLAT_COMBINE1_DIS 17 17
regRMI_RB_GLX_CID_MAP = 0x1898
# 	CB_COLOR_MAP 0 3
# 	CB_FMASK_MAP 4 7
# 	CB_CMASK_MAP 8 11
# 	CB_DCC_MAP 12 15
# 	DB_Z_MAP 16 19
# 	DB_S_MAP 20 23
# 	DB_TILE_MAP 24 27
# 	DB_ZPCPSD_MAP 28 31
regRMI_SCOREBOARD_CNTL = 0x1890
# 	COMPLETE_RB0_FLUSH 0 0
# 	REQ_IN_RE_EN_AFTER_FLUSH_RB0 1 1
# 	COMPLETE_RB1_FLUSH 2 2
# 	REQ_IN_RE_EN_AFTER_FLUSH_RB1 3 3
# 	VMID_INVAL_FLUSH_TYPE_OVERRIDE_EN 5 5
# 	VMID_INVAL_FLUSH_TYPE_OVERRIDE_VALUE 6 6
# 	FORCE_VMID_INVAL_DONE_TIMER_START_VALUE 9 20
regRMI_SCOREBOARD_STATUS0 = 0x1891
# 	CURRENT_SESSION_ID 0 0
# 	CP_VMID_INV_IN_PROG 1 1
# 	CP_VMID_INV_REQ_VMID 2 17
# 	CP_VMID_INV_UTC_DONE 18 18
# 	CP_VMID_INV_DONE 19 19
# 	CP_VMID_INV_FLUSH_TYPE 20 20
# 	FORCE_VMID_INV_DONE 21 21
# 	COUNTER_SELECT 22 26
regRMI_SCOREBOARD_STATUS1 = 0x1892
# 	RUNNING_CNT_RB0 0 11
# 	RUNNING_CNT_UNDERFLOW_RB0 12 12
# 	RUNNING_CNT_OVERFLOW_RB0 13 13
# 	MULTI_VMID_INVAL_FROM_CP_DETECTED 14 14
# 	RUNNING_CNT_RB1 15 26
# 	RUNNING_CNT_UNDERFLOW_RB1 27 27
# 	RUNNING_CNT_OVERFLOW_RB1 28 28
# 	COM_FLUSH_IN_PROG_RB1 29 29
# 	COM_FLUSH_IN_PROG_RB0 30 30
regRMI_SCOREBOARD_STATUS2 = 0x1893
# 	SNAPSHOT_CNT_RB0 0 11
# 	SNAPSHOT_CNT_UNDERFLOW_RB0 12 12
# 	SNAPSHOT_CNT_RB1 13 24
# 	SNAPSHOT_CNT_UNDERFLOW_RB1 25 25
# 	COM_FLUSH_DONE_RB1 26 26
# 	COM_FLUSH_DONE_RB0 27 27
# 	TIME_STAMP_FLUSH_IN_PROG_RB0 28 28
# 	TIME_STAMP_FLUSH_IN_PROG_RB1 29 29
# 	TIME_STAMP_FLUSH_DONE_RB0 30 30
# 	TIME_STAMP_FLUSH_DONE_RB1 31 31
regRMI_SPARE = 0x189f
# 	RMI_2_GL1_128B_READ_DISABLE 1 1
# 	RMI_2_GL1_REPEATER_FGCG_DISABLE 2 2
# 	RMI_2_RB_REPEATER_FGCG_DISABLE 3 3
# 	EARLY_WRITE_ACK_ENABLE_C_RW_NOA_RESOLVE_DIS 4 4
# 	RMI_REORDER_BYPASS_CHANNEL_DIS 5 5
# 	XNACK_RETURN_DATA_OVERRIDE 6 6
# 	SPARE_BIT_7 7 7
# 	NOFILL_RMI_CID_CC 8 8
# 	NOFILL_RMI_CID_FC 9 9
# 	NOFILL_RMI_CID_CM 10 10
# 	NOFILL_RMI_CID_DC 11 11
# 	NOFILL_RMI_CID_Z 12 12
# 	NOFILL_RMI_CID_S 13 13
# 	NOFILL_RMI_CID_TILE 14 14
# 	SPARE_BIT_15_0 15 15
# 	ARBITER_ADDRESS_MASK 16 31
regRMI_SPARE_1 = 0x18a0
# 	EARLY_WRACK_FIFO_DISABLE 0 0
# 	SPARE_BIT_9 1 1
# 	SPARE_BIT_10 2 2
# 	SPARE_BIT_11 3 3
# 	SPARE_BIT_12 4 4
# 	SPARE_BIT_13 5 5
# 	SPARE_BIT_14 6 6
# 	SPARE_BIT_15 7 7
# 	RMI_REORDER_DIS_BY_CID 8 15
# 	SPARE_BIT_16_1 16 31
regRMI_SPARE_2 = 0x18a1
# 	ERROR_ZERO_BYTE_MASK_CID 0 15
# 	SPARE_BIT_8_2 16 23
# 	SPARE_BIT_8_3 24 31
regRMI_SUBBLOCK_STATUS0 = 0x1883
# 	UTC_EXT_LAT_HID_FIFO_NUM_USED_PROBE0 0 6
# 	UTC_EXT_LAT_HID_FIFO_FULL_PROBE0 7 7
# 	UTC_EXT_LAT_HID_FIFO_EMPTY_PROBE0 8 8
# 	UTC_EXT_LAT_HID_FIFO_NUM_USED_PROBE1 9 15
# 	UTC_EXT_LAT_HID_FIFO_FULL_PROBE1 16 16
# 	UTC_EXT_LAT_HID_FIFO_EMPTY_PROBE1 17 17
# 	TCIW0_INFLIGHT_CNT 18 27
regRMI_SUBBLOCK_STATUS1 = 0x1884
# 	SKID_FIFO_0_FREE_SPACE 0 9
# 	SKID_FIFO_1_FREE_SPACE 10 19
# 	TCIW1_INFLIGHT_CNT 20 29
regRMI_SUBBLOCK_STATUS2 = 0x1885
# 	PRT_FIFO_0_NUM_USED 0 8
# 	PRT_FIFO_1_NUM_USED 9 17
regRMI_SUBBLOCK_STATUS3 = 0x1886
# 	SKID_FIFO_0_FREE_SPACE_TOTAL 0 9
# 	SKID_FIFO_1_FREE_SPACE_TOTAL 10 19
regRMI_TCIW_FORMATTER0_CNTL = 0x188e
# 	TCIW0_MAX_ALLOWED_INFLIGHT_REQ 9 18
# 	RMI_IN0_REORDER_DIS 29 29
# 	ALL_FAULT_RET0_DATA 31 31
regRMI_TCIW_FORMATTER1_CNTL = 0x188f
# 	WR_COMBINE1_DIS_OVERRIDE 0 0
# 	WR_COMBINE1_TIME_OUT_WINDOW 1 8
# 	TCIW1_MAX_ALLOWED_INFLIGHT_REQ 9 18
# 	RMI_IN1_REORDER_DIS 29 29
# 	WR_COMBINE1_DIS_AT_LAST_OF_BURST 30 30
# 	ALL_FAULT_RET1_DATA 31 31
regRMI_UTCL1_CNTL1 = 0x188b
# 	FORCE_4K_L2_RESP 0 0
# 	GPUVM_64K_DEF 1 1
# 	GPUVM_PERM_MODE 2 2
# 	RESP_MODE 3 4
# 	RESP_FAULT_MODE 5 6
# 	CLIENTID 7 15
# 	USERVM_DIS 16 16
# 	ENABLE_PUSH_LFIFO 17 17
# 	ENABLE_LFIFO_PRI_ARB 18 18
# 	REG_INV_VMID 19 22
# 	REG_INV_ALL_VMID 23 23
# 	REG_INV_TOGGLE 24 24
# 	CLIENT_INVALIDATE_ALL_VMID 25 25
# 	FORCE_MISS 26 26
# 	FORCE_IN_ORDER 27 27
# 	REDUCE_FIFO_DEPTH_BY_2 28 29
# 	REDUCE_CACHE_SIZE_BY_2 30 31
regRMI_UTCL1_CNTL2 = 0x188c
# 	UTC_SPARE 0 7
# 	MTYPE_OVRD_DIS 9 9
# 	LINE_VALID 10 10
# 	DIS_EDC 11 11
# 	GPUVM_INV_MODE 12 12
# 	SHOOTDOWN_OPT 13 13
# 	FORCE_SNOOP 14 14
# 	FORCE_GPUVM_INV_ACK 15 15
# 	UTCL1_ARB_BURST_MODE 16 17
# 	UTCL1_ENABLE_PERF_EVENT_RD_WR 18 18
# 	UTCL1_PERF_EVENT_RD_WR 19 19
# 	UTCL1_ENABLE_PERF_EVENT_VMID 20 20
# 	UTCL1_PERF_EVENT_VMID 21 24
# 	UTCL1_DIS_DUAL_L2_REQ 25 25
# 	UTCL1_FORCE_FRAG_2M_TO_64K 26 26
# 	PERM_MODE_OVRD 27 27
# 	LINE_INVALIDATE_OPT 28 28
# 	GPUVM_16K_DEFAULT 29 29
# 	FGCG_DISABLE 30 30
# 	RESERVED 31 31
regRMI_UTCL1_STATUS = 0x1897
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
regRMI_UTC_UNIT_CONFIG = 0x188d
# 	TMZ_REQ_EN 0 15
regRMI_UTC_XNACK_N_MISC_CNTL = 0x1889
# 	MASTER_XNACK_TIMER_INC 0 7
# 	IND_XNACK_TIMER_START_VALUE 8 11
# 	UTCL1_PERM_MODE 12 12
# 	CP_VMID_RESET_REQUEST_DISABLE 13 13
regRMI_XBAR_ARBITER_CONFIG = 0x1894
# 	XBAR_ARB0_MODE 0 1
# 	XBAR_ARB0_BREAK_LOB_ON_WEIGHTEDRR 2 2
# 	XBAR_ARB0_STALL 3 3
# 	XBAR_ARB0_BREAK_LOB_ON_IDLEIN 4 4
# 	XBAR_ARB0_MODE_OVERRIDE_EN 5 5
# 	XBAR_ARB0_STALL_TIMER_OVERRIDE 6 7
# 	XBAR_ARB0_STALL_TIMER_START_VALUE 8 15
# 	XBAR_ARB1_MODE 16 17
# 	XBAR_ARB1_BREAK_LOB_ON_WEIGHTEDRR 18 18
# 	XBAR_ARB1_STALL 19 19
# 	XBAR_ARB1_BREAK_LOB_ON_IDLEIN 20 20
# 	XBAR_ARB1_MODE_OVERRIDE_EN 21 21
# 	XBAR_ARB1_STALL_TIMER_OVERRIDE 22 23
# 	XBAR_ARB1_STALL_TIMER_START_VALUE 24 31
regRMI_XBAR_ARBITER_CONFIG_1 = 0x1895
# 	XBAR_ARB_ROUND_ROBIN_WEIGHT_RB0_RD 0 7
# 	XBAR_ARB_ROUND_ROBIN_WEIGHT_RB0_WR 8 15
regRMI_XBAR_CONFIG = 0x1887
# 	XBAR_MUX_CONFIG_OVERRIDE 0 1
# 	XBAR_MUX_CONFIG_REQ_TYPE_OVERRIDE 2 5
# 	XBAR_MUX_CONFIG_CB_DB_OVERRIDE 6 6
# 	ARBITER_DIS 7 7
# 	XBAR_EN_IN_REQ 8 11
# 	XBAR_EN_IN_REQ_OVERRIDE 12 12
# 	XBAR_EN_IN_RB0 13 13
regRTAVFS_RTAVFS_REG_ADDR = 0x4b00
# 	RTAVFSADDR 0 9
regRTAVFS_RTAVFS_WR_DATA = 0x4b01
# 	RTAVFSDATA 0 31
regSCRATCH_REG0 = 0x2040
# 	SCRATCH_REG0 0 31
regSCRATCH_REG1 = 0x2041
# 	SCRATCH_REG1 0 31
regSCRATCH_REG2 = 0x2042
# 	SCRATCH_REG2 0 31
regSCRATCH_REG3 = 0x2043
# 	SCRATCH_REG3 0 31
regSCRATCH_REG4 = 0x2044
# 	SCRATCH_REG4 0 31
regSCRATCH_REG5 = 0x2045
# 	SCRATCH_REG5 0 31
regSCRATCH_REG6 = 0x2046
# 	SCRATCH_REG6 0 31
regSCRATCH_REG7 = 0x2047
# 	SCRATCH_REG7 0 31
regSCRATCH_REG_ATOMIC = 0x2048
# 	IMMED 0 23
# 	ID 24 26
# 	reserved27 27 27
# 	OP 28 30
# 	reserved31 31 31
regSCRATCH_REG_CMPSWAP_ATOMIC = 0x2048
# 	IMMED_COMPARE 0 11
# 	IMMED_REPLACE 12 23
# 	ID 24 26
# 	reserved27 27 27
# 	OP 28 30
# 	reserved31 31 31
regSDMA0_AQL_STATUS = 0x5f
# 	COMPLETE_SIGNAL_EMPTY 0 0
# 	INVALID_CMD_EMPTY 1 1
regSDMA0_ATOMIC_CNTL = 0x39
# 	LOOP_TIMER 0 30
# 	ATOMIC_RTN_INT_ENABLE 31 31
regSDMA0_ATOMIC_PREOP_HI = 0x3b
# 	DATA 0 31
regSDMA0_ATOMIC_PREOP_LO = 0x3a
# 	DATA 0 31
regSDMA0_BA_THRESHOLD = 0x33
# 	READ_THRES 0 9
# 	WRITE_THRES 16 25
regSDMA0_BROADCAST_UCODE_ADDR = 0x5886
# 	VALUE 0 12
# 	THID 15 15
regSDMA0_BROADCAST_UCODE_DATA = 0x5887
# 	VALUE 0 31
regSDMA0_CE_CTRL = 0x7e
# 	RD_LUT_WATERMARK 0 2
# 	RD_LUT_DEPTH 3 4
# 	WR_AFIFO_WATERMARK 5 7
# 	CE_DCC_READ_128B_ENABLE 8 8
# 	RESERVED 9 31
regSDMA0_CHICKEN_BITS = 0x1d
# 	STALL_ON_TRANS_FULL_ENABLE 1 1
# 	STALL_ON_NO_FREE_DATA_BUFFER_ENABLE 2 2
# 	SRBM_POLL_RETRYING 5 5
# 	RD_BURST 6 7
# 	WR_BURST 8 9
# 	COMBINE_256B_WAIT_CYCLE 10 13
# 	WR_COMBINE_256B_ENABLE 14 14
# 	RD_COMBINE_256B_ENABLE 15 15
# 	COPY_OVERLAP_ENABLE 16 16
# 	RAW_CHECK_ENABLE 17 17
# 	T2L_256B_ENABLE 18 18
# 	SOFT_OVERRIDE_GCR_FGCG 19 19
# 	SOFT_OVERRIDE_GRBM_FGCG 20 20
# 	SOFT_OVERRIDE_CH_FGCG 21 21
# 	SOFT_OVERRIDE_UTCL2_INVREQ_FGCG 22 22
# 	SOFT_OVERRIDE_UTCL1_FGCG 23 23
# 	CG_STATUS_OUTPUT 24 24
# 	SW_FREEZE_ENABLE 25 25
# 	RESERVED 26 31
regSDMA0_CHICKEN_BITS_2 = 0x4b
# 	F32_CMD_PROC_DELAY 0 3
# 	F32_SEND_POSTCODE_EN 4 4
# 	UCODE_BUF_DS_EN 6 6
# 	UCODE_SELFLOAD_THREAD_OVERLAP 7 7
# 	WPTR_POLL_OUTSTANDING 8 11
# 	RESERVED_14_12 12 14
# 	RESERVED_15 15 15
# 	RB_FIFO_WATERMARK 16 17
# 	IB_FIFO_WATERMARK 18 19
# 	RESERVED_22_20 20 22
# 	CH_RD_WATERMARK 23 24
# 	CH_WR_WATERMARK 25 29
# 	CH_WR_WATERMARK_LSB 30 30
# 	PIO_VFID_SOURCE 31 31
regSDMA0_CLOCK_GATING_STATUS = 0x75
# 	DYN_CLK_GATE_STATUS 0 0
# 	CE_CLK_GATE_STATUS 2 2
# 	CE_BC_CLK_GATE_STATUS 3 3
# 	CE_NBC_CLK_GATE_STATUS 4 4
# 	REG_CLK_GATE_STATUS 5 5
# 	F32_CLK_GATE_STATUS 6 6
regSDMA0_CNTL = 0x1c
# 	TRAP_ENABLE 0 0
# 	SEM_WAIT_INT_ENABLE 2 2
# 	DATA_SWAP_ENABLE 3 3
# 	FENCE_SWAP_ENABLE 4 4
# 	MIDCMD_PREEMPT_ENABLE 5 5
# 	PIO_DONE_ACK_ENABLE 6 6
# 	TMZ_MIDCMD_PREEMPT_ENABLE 8 8
# 	MIDCMD_EXPIRE_ENABLE 9 9
# 	CP_MES_INT_ENABLE 10 10
# 	PAGE_RETRY_TIMEOUT_INT_ENABLE 11 11
# 	PAGE_NULL_INT_ENABLE 12 12
# 	PAGE_FAULT_INT_ENABLE 13 13
# 	CH_PERFCNT_ENABLE 16 16
# 	MIDCMD_WORLDSWITCH_ENABLE 17 17
# 	CTXEMPTY_INT_ENABLE 28 28
# 	FROZEN_INT_ENABLE 29 29
# 	IB_PREEMPT_INT_ENABLE 30 30
# 	RB_PREEMPT_INT_ENABLE 31 31
regSDMA0_CNTL1 = 0x27
# 	WPTR_POLL_FREQUENCY 2 15
regSDMA0_CRD_CNTL = 0x5b
# 	MC_WRREQ_CREDIT 7 12
# 	MC_RDREQ_CREDIT 13 18
# 	CH_WRREQ_CREDIT 19 24
# 	CH_RDREQ_CREDIT 25 30
regSDMA0_DEC_START = 0x0
# 	START 0 31
regSDMA0_EA_DBIT_ADDR_DATA = 0x60
# 	VALUE 0 31
regSDMA0_EA_DBIT_ADDR_INDEX = 0x61
# 	VALUE 0 2
regSDMA0_EDC_CONFIG = 0x32
# 	DIS_EDC 1 1
# 	ECC_INT_ENABLE 2 2
regSDMA0_EDC_COUNTER = 0x36
# 	SDMA_UCODE_BUF_DED 0 0
# 	SDMA_UCODE_BUF_SEC 1 1
# 	SDMA_RB_CMD_BUF_SED 2 2
# 	SDMA_IB_CMD_BUF_SED 3 3
# 	SDMA_UTCL1_RD_FIFO_SED 4 4
# 	SDMA_UTCL1_RDBST_FIFO_SED 5 5
# 	SDMA_DATA_LUT_FIFO_SED 6 6
# 	SDMA_MBANK_DATA_BUF0_SED 7 7
# 	SDMA_MBANK_DATA_BUF1_SED 8 8
# 	SDMA_MBANK_DATA_BUF2_SED 9 9
# 	SDMA_MBANK_DATA_BUF3_SED 10 10
# 	SDMA_MBANK_DATA_BUF4_SED 11 11
# 	SDMA_MBANK_DATA_BUF5_SED 12 12
# 	SDMA_MBANK_DATA_BUF6_SED 13 13
# 	SDMA_MBANK_DATA_BUF7_SED 14 14
# 	SDMA_SPLIT_DAT_BUF_SED 15 15
# 	SDMA_MC_WR_ADDR_FIFO_SED 16 16
regSDMA0_EDC_COUNTER_CLEAR = 0x37
# 	DUMMY 0 0
regSDMA0_ERROR_LOG = 0x50
# 	OVERRIDE 0 15
# 	STATUS 16 31
regSDMA0_F32_CNTL = 0x589a
# 	HALT 0 0
# 	TH0_CHECKSUM_CLR 8 8
# 	TH0_RESET 9 9
# 	TH0_ENABLE 10 10
# 	TH1_CHECKSUM_CLR 12 12
# 	TH1_RESET 13 13
# 	TH1_ENABLE 14 14
# 	TH0_PRIORITY 16 23
# 	TH1_PRIORITY 24 31
regSDMA0_F32_COUNTER = 0x55
# 	VALUE 0 31
regSDMA0_F32_MISC_CNTL = 0xb
# 	F32_WAKEUP 0 0
regSDMA0_FED_STATUS = 0x7f
# 	RB_FETCH_ECC 0 0
# 	IB_FETCH_ECC 1 1
# 	F32_DATA_ECC 2 2
# 	WPTR_ATOMIC_ECC 3 3
# 	COPY_DATA_ECC 4 4
# 	COPY_METADATA_ECC 5 5
# 	SELFLOAD_UCODE_ECC 6 6
regSDMA0_FREEZE = 0x2b
# 	PREEMPT 0 0
# 	FREEZE 4 4
# 	FROZEN 5 5
# 	F32_FREEZE 6 6
regSDMA0_GB_ADDR_CONFIG = 0x1e
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regSDMA0_GB_ADDR_CONFIG_READ = 0x1f
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regSDMA0_GLOBAL_QUANTUM = 0x4f
# 	GLOBAL_FOCUS_QUANTUM 0 7
# 	GLOBAL_NORMAL_QUANTUM 8 15
regSDMA0_GLOBAL_TIMESTAMP_HI = 0x10
# 	DATA 0 31
regSDMA0_GLOBAL_TIMESTAMP_LO = 0xf
# 	DATA 0 31
regSDMA0_HBM_PAGE_CONFIG = 0x28
# 	PAGE_SIZE_EXPONENT 0 1
regSDMA0_HOLE_ADDR_HI = 0x73
# 	VALUE 0 31
regSDMA0_HOLE_ADDR_LO = 0x72
# 	VALUE 0 31
regSDMA0_IB_OFFSET_FETCH = 0x23
# 	OFFSET 2 21
regSDMA0_ID = 0x34
# 	DEVICE_ID 0 7
regSDMA0_INT_STATUS = 0x70
# 	DATA 0 31
regSDMA0_PERFCNT_MISC_CNTL = 0x3e23
# 	CMD_OP 0 15
regSDMA0_PERFCNT_PERFCOUNTER0_CFG = 0x3e20
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regSDMA0_PERFCNT_PERFCOUNTER1_CFG = 0x3e21
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regSDMA0_PERFCNT_PERFCOUNTER_HI = 0x3661
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regSDMA0_PERFCNT_PERFCOUNTER_LO = 0x3660
# 	COUNTER_LO 0 31
regSDMA0_PERFCNT_PERFCOUNTER_RSLT_CNTL = 0x3e22
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regSDMA0_PERFCOUNTER0_HI = 0x3663
# 	PERFCOUNTER_HI 0 31
regSDMA0_PERFCOUNTER0_LO = 0x3662
# 	PERFCOUNTER_LO 0 31
regSDMA0_PERFCOUNTER0_SELECT = 0x3e24
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSDMA0_PERFCOUNTER0_SELECT1 = 0x3e25
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSDMA0_PERFCOUNTER1_HI = 0x3665
# 	PERFCOUNTER_HI 0 31
regSDMA0_PERFCOUNTER1_LO = 0x3664
# 	PERFCOUNTER_LO 0 31
regSDMA0_PERFCOUNTER1_SELECT = 0x3e26
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSDMA0_PERFCOUNTER1_SELECT1 = 0x3e27
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSDMA0_PHYSICAL_ADDR_HI = 0x4e
# 	ADDR 0 15
regSDMA0_PHYSICAL_ADDR_LO = 0x4d
# 	D_VALID 0 0
# 	DIRTY 1 1
# 	PHY_VALID 2 2
# 	ADDR 12 31
regSDMA0_POWER_CNTL = 0x1a
# 	LS_ENABLE 8 8
regSDMA0_PROCESS_QUANTUM0 = 0x2c
# 	PROCESS0_QUANTUM 0 7
# 	PROCESS1_QUANTUM 8 15
# 	PROCESS2_QUANTUM 16 23
# 	PROCESS3_QUANTUM 24 31
regSDMA0_PROCESS_QUANTUM1 = 0x2d
# 	PROCESS4_QUANTUM 0 7
# 	PROCESS5_QUANTUM 8 15
# 	PROCESS6_QUANTUM 16 23
# 	PROCESS7_QUANTUM 24 31
regSDMA0_PROGRAM = 0x24
# 	STREAM 0 31
regSDMA0_PUB_DUMMY_REG0 = 0x51
# 	VALUE 0 31
regSDMA0_PUB_DUMMY_REG1 = 0x52
# 	VALUE 0 31
regSDMA0_PUB_DUMMY_REG2 = 0x53
# 	VALUE 0 31
regSDMA0_PUB_DUMMY_REG3 = 0x54
# 	VALUE 0 31
regSDMA0_QUEUE0_CONTEXT_STATUS = 0x91
# 	SELECTED 0 0
# 	USE_IB 1 1
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE0_CSA_ADDR_HI = 0xad
# 	ADDR 0 31
regSDMA0_QUEUE0_CSA_ADDR_LO = 0xac
# 	ADDR 2 31
regSDMA0_QUEUE0_DOORBELL = 0x92
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE0_DOORBELL_LOG = 0xa9
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE0_DOORBELL_OFFSET = 0xab
# 	OFFSET 2 27
regSDMA0_QUEUE0_DUMMY_REG = 0xb1
# 	DUMMY 0 31
regSDMA0_QUEUE0_IB_BASE_HI = 0x8e
# 	ADDR 0 31
regSDMA0_QUEUE0_IB_BASE_LO = 0x8d
# 	ADDR 5 31
regSDMA0_QUEUE0_IB_CNTL = 0x8a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE0_IB_OFFSET = 0x8c
# 	OFFSET 2 21
regSDMA0_QUEUE0_IB_RPTR = 0x8b
# 	OFFSET 2 21
regSDMA0_QUEUE0_IB_SIZE = 0x8f
# 	SIZE 0 19
regSDMA0_QUEUE0_IB_SUB_REMAIN = 0xaf
# 	SIZE 0 13
regSDMA0_QUEUE0_MIDCMD_CNTL = 0xcb
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE0_MIDCMD_DATA0 = 0xc0
# 	DATA0 0 31
regSDMA0_QUEUE0_MIDCMD_DATA1 = 0xc1
# 	DATA1 0 31
regSDMA0_QUEUE0_MIDCMD_DATA10 = 0xca
# 	DATA10 0 31
regSDMA0_QUEUE0_MIDCMD_DATA2 = 0xc2
# 	DATA2 0 31
regSDMA0_QUEUE0_MIDCMD_DATA3 = 0xc3
# 	DATA3 0 31
regSDMA0_QUEUE0_MIDCMD_DATA4 = 0xc4
# 	DATA4 0 31
regSDMA0_QUEUE0_MIDCMD_DATA5 = 0xc5
# 	DATA5 0 31
regSDMA0_QUEUE0_MIDCMD_DATA6 = 0xc6
# 	DATA6 0 31
regSDMA0_QUEUE0_MIDCMD_DATA7 = 0xc7
# 	DATA7 0 31
regSDMA0_QUEUE0_MIDCMD_DATA8 = 0xc8
# 	DATA8 0 31
regSDMA0_QUEUE0_MIDCMD_DATA9 = 0xc9
# 	DATA9 0 31
regSDMA0_QUEUE0_MINOR_PTR_UPDATE = 0xb5
# 	ENABLE 0 0
regSDMA0_QUEUE0_PREEMPT = 0xb0
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE0_RB_AQL_CNTL = 0xb4
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE0_RB_BASE = 0x81
# 	ADDR 0 31
regSDMA0_QUEUE0_RB_BASE_HI = 0x82
# 	ADDR 0 23
regSDMA0_QUEUE0_RB_CNTL = 0x80
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE0_RB_PREEMPT = 0xb6
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE0_RB_RPTR = 0x83
# 	OFFSET 0 31
regSDMA0_QUEUE0_RB_RPTR_ADDR_HI = 0x88
# 	ADDR 0 31
regSDMA0_QUEUE0_RB_RPTR_ADDR_LO = 0x89
# 	ADDR 2 31
regSDMA0_QUEUE0_RB_RPTR_HI = 0x84
# 	OFFSET 0 31
regSDMA0_QUEUE0_RB_WPTR = 0x85
# 	OFFSET 0 31
regSDMA0_QUEUE0_RB_WPTR_HI = 0x86
# 	OFFSET 0 31
regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_HI = 0xb2
# 	ADDR 0 31
regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_LO = 0xb3
# 	ADDR 2 31
regSDMA0_QUEUE0_SCHEDULE_CNTL = 0xae
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE0_SKIP_CNTL = 0x90
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE1_CONTEXT_STATUS = 0xe9
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE1_CSA_ADDR_HI = 0x105
# 	ADDR 0 31
regSDMA0_QUEUE1_CSA_ADDR_LO = 0x104
# 	ADDR 2 31
regSDMA0_QUEUE1_DOORBELL = 0xea
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE1_DOORBELL_LOG = 0x101
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE1_DOORBELL_OFFSET = 0x103
# 	OFFSET 2 27
regSDMA0_QUEUE1_DUMMY_REG = 0x109
# 	DUMMY 0 31
regSDMA0_QUEUE1_IB_BASE_HI = 0xe6
# 	ADDR 0 31
regSDMA0_QUEUE1_IB_BASE_LO = 0xe5
# 	ADDR 5 31
regSDMA0_QUEUE1_IB_CNTL = 0xe2
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE1_IB_OFFSET = 0xe4
# 	OFFSET 2 21
regSDMA0_QUEUE1_IB_RPTR = 0xe3
# 	OFFSET 2 21
regSDMA0_QUEUE1_IB_SIZE = 0xe7
# 	SIZE 0 19
regSDMA0_QUEUE1_IB_SUB_REMAIN = 0x107
# 	SIZE 0 13
regSDMA0_QUEUE1_MIDCMD_CNTL = 0x123
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE1_MIDCMD_DATA0 = 0x118
# 	DATA0 0 31
regSDMA0_QUEUE1_MIDCMD_DATA1 = 0x119
# 	DATA1 0 31
regSDMA0_QUEUE1_MIDCMD_DATA10 = 0x122
# 	DATA10 0 31
regSDMA0_QUEUE1_MIDCMD_DATA2 = 0x11a
# 	DATA2 0 31
regSDMA0_QUEUE1_MIDCMD_DATA3 = 0x11b
# 	DATA3 0 31
regSDMA0_QUEUE1_MIDCMD_DATA4 = 0x11c
# 	DATA4 0 31
regSDMA0_QUEUE1_MIDCMD_DATA5 = 0x11d
# 	DATA5 0 31
regSDMA0_QUEUE1_MIDCMD_DATA6 = 0x11e
# 	DATA6 0 31
regSDMA0_QUEUE1_MIDCMD_DATA7 = 0x11f
# 	DATA7 0 31
regSDMA0_QUEUE1_MIDCMD_DATA8 = 0x120
# 	DATA8 0 31
regSDMA0_QUEUE1_MIDCMD_DATA9 = 0x121
# 	DATA9 0 31
regSDMA0_QUEUE1_MINOR_PTR_UPDATE = 0x10d
# 	ENABLE 0 0
regSDMA0_QUEUE1_PREEMPT = 0x108
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE1_RB_AQL_CNTL = 0x10c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE1_RB_BASE = 0xd9
# 	ADDR 0 31
regSDMA0_QUEUE1_RB_BASE_HI = 0xda
# 	ADDR 0 23
regSDMA0_QUEUE1_RB_CNTL = 0xd8
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE1_RB_PREEMPT = 0x10e
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE1_RB_RPTR = 0xdb
# 	OFFSET 0 31
regSDMA0_QUEUE1_RB_RPTR_ADDR_HI = 0xe0
# 	ADDR 0 31
regSDMA0_QUEUE1_RB_RPTR_ADDR_LO = 0xe1
# 	ADDR 2 31
regSDMA0_QUEUE1_RB_RPTR_HI = 0xdc
# 	OFFSET 0 31
regSDMA0_QUEUE1_RB_WPTR = 0xdd
# 	OFFSET 0 31
regSDMA0_QUEUE1_RB_WPTR_HI = 0xde
# 	OFFSET 0 31
regSDMA0_QUEUE1_RB_WPTR_POLL_ADDR_HI = 0x10a
# 	ADDR 0 31
regSDMA0_QUEUE1_RB_WPTR_POLL_ADDR_LO = 0x10b
# 	ADDR 2 31
regSDMA0_QUEUE1_SCHEDULE_CNTL = 0x106
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE1_SKIP_CNTL = 0xe8
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE2_CONTEXT_STATUS = 0x141
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE2_CSA_ADDR_HI = 0x15d
# 	ADDR 0 31
regSDMA0_QUEUE2_CSA_ADDR_LO = 0x15c
# 	ADDR 2 31
regSDMA0_QUEUE2_DOORBELL = 0x142
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE2_DOORBELL_LOG = 0x159
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE2_DOORBELL_OFFSET = 0x15b
# 	OFFSET 2 27
regSDMA0_QUEUE2_DUMMY_REG = 0x161
# 	DUMMY 0 31
regSDMA0_QUEUE2_IB_BASE_HI = 0x13e
# 	ADDR 0 31
regSDMA0_QUEUE2_IB_BASE_LO = 0x13d
# 	ADDR 5 31
regSDMA0_QUEUE2_IB_CNTL = 0x13a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE2_IB_OFFSET = 0x13c
# 	OFFSET 2 21
regSDMA0_QUEUE2_IB_RPTR = 0x13b
# 	OFFSET 2 21
regSDMA0_QUEUE2_IB_SIZE = 0x13f
# 	SIZE 0 19
regSDMA0_QUEUE2_IB_SUB_REMAIN = 0x15f
# 	SIZE 0 13
regSDMA0_QUEUE2_MIDCMD_CNTL = 0x17b
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE2_MIDCMD_DATA0 = 0x170
# 	DATA0 0 31
regSDMA0_QUEUE2_MIDCMD_DATA1 = 0x171
# 	DATA1 0 31
regSDMA0_QUEUE2_MIDCMD_DATA10 = 0x17a
# 	DATA10 0 31
regSDMA0_QUEUE2_MIDCMD_DATA2 = 0x172
# 	DATA2 0 31
regSDMA0_QUEUE2_MIDCMD_DATA3 = 0x173
# 	DATA3 0 31
regSDMA0_QUEUE2_MIDCMD_DATA4 = 0x174
# 	DATA4 0 31
regSDMA0_QUEUE2_MIDCMD_DATA5 = 0x175
# 	DATA5 0 31
regSDMA0_QUEUE2_MIDCMD_DATA6 = 0x176
# 	DATA6 0 31
regSDMA0_QUEUE2_MIDCMD_DATA7 = 0x177
# 	DATA7 0 31
regSDMA0_QUEUE2_MIDCMD_DATA8 = 0x178
# 	DATA8 0 31
regSDMA0_QUEUE2_MIDCMD_DATA9 = 0x179
# 	DATA9 0 31
regSDMA0_QUEUE2_MINOR_PTR_UPDATE = 0x165
# 	ENABLE 0 0
regSDMA0_QUEUE2_PREEMPT = 0x160
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE2_RB_AQL_CNTL = 0x164
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE2_RB_BASE = 0x131
# 	ADDR 0 31
regSDMA0_QUEUE2_RB_BASE_HI = 0x132
# 	ADDR 0 23
regSDMA0_QUEUE2_RB_CNTL = 0x130
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE2_RB_PREEMPT = 0x166
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE2_RB_RPTR = 0x133
# 	OFFSET 0 31
regSDMA0_QUEUE2_RB_RPTR_ADDR_HI = 0x138
# 	ADDR 0 31
regSDMA0_QUEUE2_RB_RPTR_ADDR_LO = 0x139
# 	ADDR 2 31
regSDMA0_QUEUE2_RB_RPTR_HI = 0x134
# 	OFFSET 0 31
regSDMA0_QUEUE2_RB_WPTR = 0x135
# 	OFFSET 0 31
regSDMA0_QUEUE2_RB_WPTR_HI = 0x136
# 	OFFSET 0 31
regSDMA0_QUEUE2_RB_WPTR_POLL_ADDR_HI = 0x162
# 	ADDR 0 31
regSDMA0_QUEUE2_RB_WPTR_POLL_ADDR_LO = 0x163
# 	ADDR 2 31
regSDMA0_QUEUE2_SCHEDULE_CNTL = 0x15e
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE2_SKIP_CNTL = 0x140
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE3_CONTEXT_STATUS = 0x199
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE3_CSA_ADDR_HI = 0x1b5
# 	ADDR 0 31
regSDMA0_QUEUE3_CSA_ADDR_LO = 0x1b4
# 	ADDR 2 31
regSDMA0_QUEUE3_DOORBELL = 0x19a
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE3_DOORBELL_LOG = 0x1b1
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE3_DOORBELL_OFFSET = 0x1b3
# 	OFFSET 2 27
regSDMA0_QUEUE3_DUMMY_REG = 0x1b9
# 	DUMMY 0 31
regSDMA0_QUEUE3_IB_BASE_HI = 0x196
# 	ADDR 0 31
regSDMA0_QUEUE3_IB_BASE_LO = 0x195
# 	ADDR 5 31
regSDMA0_QUEUE3_IB_CNTL = 0x192
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE3_IB_OFFSET = 0x194
# 	OFFSET 2 21
regSDMA0_QUEUE3_IB_RPTR = 0x193
# 	OFFSET 2 21
regSDMA0_QUEUE3_IB_SIZE = 0x197
# 	SIZE 0 19
regSDMA0_QUEUE3_IB_SUB_REMAIN = 0x1b7
# 	SIZE 0 13
regSDMA0_QUEUE3_MIDCMD_CNTL = 0x1d3
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE3_MIDCMD_DATA0 = 0x1c8
# 	DATA0 0 31
regSDMA0_QUEUE3_MIDCMD_DATA1 = 0x1c9
# 	DATA1 0 31
regSDMA0_QUEUE3_MIDCMD_DATA10 = 0x1d2
# 	DATA10 0 31
regSDMA0_QUEUE3_MIDCMD_DATA2 = 0x1ca
# 	DATA2 0 31
regSDMA0_QUEUE3_MIDCMD_DATA3 = 0x1cb
# 	DATA3 0 31
regSDMA0_QUEUE3_MIDCMD_DATA4 = 0x1cc
# 	DATA4 0 31
regSDMA0_QUEUE3_MIDCMD_DATA5 = 0x1cd
# 	DATA5 0 31
regSDMA0_QUEUE3_MIDCMD_DATA6 = 0x1ce
# 	DATA6 0 31
regSDMA0_QUEUE3_MIDCMD_DATA7 = 0x1cf
# 	DATA7 0 31
regSDMA0_QUEUE3_MIDCMD_DATA8 = 0x1d0
# 	DATA8 0 31
regSDMA0_QUEUE3_MIDCMD_DATA9 = 0x1d1
# 	DATA9 0 31
regSDMA0_QUEUE3_MINOR_PTR_UPDATE = 0x1bd
# 	ENABLE 0 0
regSDMA0_QUEUE3_PREEMPT = 0x1b8
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE3_RB_AQL_CNTL = 0x1bc
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE3_RB_BASE = 0x189
# 	ADDR 0 31
regSDMA0_QUEUE3_RB_BASE_HI = 0x18a
# 	ADDR 0 23
regSDMA0_QUEUE3_RB_CNTL = 0x188
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE3_RB_PREEMPT = 0x1be
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE3_RB_RPTR = 0x18b
# 	OFFSET 0 31
regSDMA0_QUEUE3_RB_RPTR_ADDR_HI = 0x190
# 	ADDR 0 31
regSDMA0_QUEUE3_RB_RPTR_ADDR_LO = 0x191
# 	ADDR 2 31
regSDMA0_QUEUE3_RB_RPTR_HI = 0x18c
# 	OFFSET 0 31
regSDMA0_QUEUE3_RB_WPTR = 0x18d
# 	OFFSET 0 31
regSDMA0_QUEUE3_RB_WPTR_HI = 0x18e
# 	OFFSET 0 31
regSDMA0_QUEUE3_RB_WPTR_POLL_ADDR_HI = 0x1ba
# 	ADDR 0 31
regSDMA0_QUEUE3_RB_WPTR_POLL_ADDR_LO = 0x1bb
# 	ADDR 2 31
regSDMA0_QUEUE3_SCHEDULE_CNTL = 0x1b6
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE3_SKIP_CNTL = 0x198
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE4_CONTEXT_STATUS = 0x1f1
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE4_CSA_ADDR_HI = 0x20d
# 	ADDR 0 31
regSDMA0_QUEUE4_CSA_ADDR_LO = 0x20c
# 	ADDR 2 31
regSDMA0_QUEUE4_DOORBELL = 0x1f2
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE4_DOORBELL_LOG = 0x209
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE4_DOORBELL_OFFSET = 0x20b
# 	OFFSET 2 27
regSDMA0_QUEUE4_DUMMY_REG = 0x211
# 	DUMMY 0 31
regSDMA0_QUEUE4_IB_BASE_HI = 0x1ee
# 	ADDR 0 31
regSDMA0_QUEUE4_IB_BASE_LO = 0x1ed
# 	ADDR 5 31
regSDMA0_QUEUE4_IB_CNTL = 0x1ea
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE4_IB_OFFSET = 0x1ec
# 	OFFSET 2 21
regSDMA0_QUEUE4_IB_RPTR = 0x1eb
# 	OFFSET 2 21
regSDMA0_QUEUE4_IB_SIZE = 0x1ef
# 	SIZE 0 19
regSDMA0_QUEUE4_IB_SUB_REMAIN = 0x20f
# 	SIZE 0 13
regSDMA0_QUEUE4_MIDCMD_CNTL = 0x22b
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE4_MIDCMD_DATA0 = 0x220
# 	DATA0 0 31
regSDMA0_QUEUE4_MIDCMD_DATA1 = 0x221
# 	DATA1 0 31
regSDMA0_QUEUE4_MIDCMD_DATA10 = 0x22a
# 	DATA10 0 31
regSDMA0_QUEUE4_MIDCMD_DATA2 = 0x222
# 	DATA2 0 31
regSDMA0_QUEUE4_MIDCMD_DATA3 = 0x223
# 	DATA3 0 31
regSDMA0_QUEUE4_MIDCMD_DATA4 = 0x224
# 	DATA4 0 31
regSDMA0_QUEUE4_MIDCMD_DATA5 = 0x225
# 	DATA5 0 31
regSDMA0_QUEUE4_MIDCMD_DATA6 = 0x226
# 	DATA6 0 31
regSDMA0_QUEUE4_MIDCMD_DATA7 = 0x227
# 	DATA7 0 31
regSDMA0_QUEUE4_MIDCMD_DATA8 = 0x228
# 	DATA8 0 31
regSDMA0_QUEUE4_MIDCMD_DATA9 = 0x229
# 	DATA9 0 31
regSDMA0_QUEUE4_MINOR_PTR_UPDATE = 0x215
# 	ENABLE 0 0
regSDMA0_QUEUE4_PREEMPT = 0x210
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE4_RB_AQL_CNTL = 0x214
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE4_RB_BASE = 0x1e1
# 	ADDR 0 31
regSDMA0_QUEUE4_RB_BASE_HI = 0x1e2
# 	ADDR 0 23
regSDMA0_QUEUE4_RB_CNTL = 0x1e0
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE4_RB_PREEMPT = 0x216
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE4_RB_RPTR = 0x1e3
# 	OFFSET 0 31
regSDMA0_QUEUE4_RB_RPTR_ADDR_HI = 0x1e8
# 	ADDR 0 31
regSDMA0_QUEUE4_RB_RPTR_ADDR_LO = 0x1e9
# 	ADDR 2 31
regSDMA0_QUEUE4_RB_RPTR_HI = 0x1e4
# 	OFFSET 0 31
regSDMA0_QUEUE4_RB_WPTR = 0x1e5
# 	OFFSET 0 31
regSDMA0_QUEUE4_RB_WPTR_HI = 0x1e6
# 	OFFSET 0 31
regSDMA0_QUEUE4_RB_WPTR_POLL_ADDR_HI = 0x212
# 	ADDR 0 31
regSDMA0_QUEUE4_RB_WPTR_POLL_ADDR_LO = 0x213
# 	ADDR 2 31
regSDMA0_QUEUE4_SCHEDULE_CNTL = 0x20e
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE4_SKIP_CNTL = 0x1f0
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE5_CONTEXT_STATUS = 0x249
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE5_CSA_ADDR_HI = 0x265
# 	ADDR 0 31
regSDMA0_QUEUE5_CSA_ADDR_LO = 0x264
# 	ADDR 2 31
regSDMA0_QUEUE5_DOORBELL = 0x24a
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE5_DOORBELL_LOG = 0x261
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE5_DOORBELL_OFFSET = 0x263
# 	OFFSET 2 27
regSDMA0_QUEUE5_DUMMY_REG = 0x269
# 	DUMMY 0 31
regSDMA0_QUEUE5_IB_BASE_HI = 0x246
# 	ADDR 0 31
regSDMA0_QUEUE5_IB_BASE_LO = 0x245
# 	ADDR 5 31
regSDMA0_QUEUE5_IB_CNTL = 0x242
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE5_IB_OFFSET = 0x244
# 	OFFSET 2 21
regSDMA0_QUEUE5_IB_RPTR = 0x243
# 	OFFSET 2 21
regSDMA0_QUEUE5_IB_SIZE = 0x247
# 	SIZE 0 19
regSDMA0_QUEUE5_IB_SUB_REMAIN = 0x267
# 	SIZE 0 13
regSDMA0_QUEUE5_MIDCMD_CNTL = 0x283
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE5_MIDCMD_DATA0 = 0x278
# 	DATA0 0 31
regSDMA0_QUEUE5_MIDCMD_DATA1 = 0x279
# 	DATA1 0 31
regSDMA0_QUEUE5_MIDCMD_DATA10 = 0x282
# 	DATA10 0 31
regSDMA0_QUEUE5_MIDCMD_DATA2 = 0x27a
# 	DATA2 0 31
regSDMA0_QUEUE5_MIDCMD_DATA3 = 0x27b
# 	DATA3 0 31
regSDMA0_QUEUE5_MIDCMD_DATA4 = 0x27c
# 	DATA4 0 31
regSDMA0_QUEUE5_MIDCMD_DATA5 = 0x27d
# 	DATA5 0 31
regSDMA0_QUEUE5_MIDCMD_DATA6 = 0x27e
# 	DATA6 0 31
regSDMA0_QUEUE5_MIDCMD_DATA7 = 0x27f
# 	DATA7 0 31
regSDMA0_QUEUE5_MIDCMD_DATA8 = 0x280
# 	DATA8 0 31
regSDMA0_QUEUE5_MIDCMD_DATA9 = 0x281
# 	DATA9 0 31
regSDMA0_QUEUE5_MINOR_PTR_UPDATE = 0x26d
# 	ENABLE 0 0
regSDMA0_QUEUE5_PREEMPT = 0x268
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE5_RB_AQL_CNTL = 0x26c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE5_RB_BASE = 0x239
# 	ADDR 0 31
regSDMA0_QUEUE5_RB_BASE_HI = 0x23a
# 	ADDR 0 23
regSDMA0_QUEUE5_RB_CNTL = 0x238
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE5_RB_PREEMPT = 0x26e
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE5_RB_RPTR = 0x23b
# 	OFFSET 0 31
regSDMA0_QUEUE5_RB_RPTR_ADDR_HI = 0x240
# 	ADDR 0 31
regSDMA0_QUEUE5_RB_RPTR_ADDR_LO = 0x241
# 	ADDR 2 31
regSDMA0_QUEUE5_RB_RPTR_HI = 0x23c
# 	OFFSET 0 31
regSDMA0_QUEUE5_RB_WPTR = 0x23d
# 	OFFSET 0 31
regSDMA0_QUEUE5_RB_WPTR_HI = 0x23e
# 	OFFSET 0 31
regSDMA0_QUEUE5_RB_WPTR_POLL_ADDR_HI = 0x26a
# 	ADDR 0 31
regSDMA0_QUEUE5_RB_WPTR_POLL_ADDR_LO = 0x26b
# 	ADDR 2 31
regSDMA0_QUEUE5_SCHEDULE_CNTL = 0x266
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE5_SKIP_CNTL = 0x248
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE6_CONTEXT_STATUS = 0x2a1
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE6_CSA_ADDR_HI = 0x2bd
# 	ADDR 0 31
regSDMA0_QUEUE6_CSA_ADDR_LO = 0x2bc
# 	ADDR 2 31
regSDMA0_QUEUE6_DOORBELL = 0x2a2
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE6_DOORBELL_LOG = 0x2b9
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE6_DOORBELL_OFFSET = 0x2bb
# 	OFFSET 2 27
regSDMA0_QUEUE6_DUMMY_REG = 0x2c1
# 	DUMMY 0 31
regSDMA0_QUEUE6_IB_BASE_HI = 0x29e
# 	ADDR 0 31
regSDMA0_QUEUE6_IB_BASE_LO = 0x29d
# 	ADDR 5 31
regSDMA0_QUEUE6_IB_CNTL = 0x29a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE6_IB_OFFSET = 0x29c
# 	OFFSET 2 21
regSDMA0_QUEUE6_IB_RPTR = 0x29b
# 	OFFSET 2 21
regSDMA0_QUEUE6_IB_SIZE = 0x29f
# 	SIZE 0 19
regSDMA0_QUEUE6_IB_SUB_REMAIN = 0x2bf
# 	SIZE 0 13
regSDMA0_QUEUE6_MIDCMD_CNTL = 0x2db
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE6_MIDCMD_DATA0 = 0x2d0
# 	DATA0 0 31
regSDMA0_QUEUE6_MIDCMD_DATA1 = 0x2d1
# 	DATA1 0 31
regSDMA0_QUEUE6_MIDCMD_DATA10 = 0x2da
# 	DATA10 0 31
regSDMA0_QUEUE6_MIDCMD_DATA2 = 0x2d2
# 	DATA2 0 31
regSDMA0_QUEUE6_MIDCMD_DATA3 = 0x2d3
# 	DATA3 0 31
regSDMA0_QUEUE6_MIDCMD_DATA4 = 0x2d4
# 	DATA4 0 31
regSDMA0_QUEUE6_MIDCMD_DATA5 = 0x2d5
# 	DATA5 0 31
regSDMA0_QUEUE6_MIDCMD_DATA6 = 0x2d6
# 	DATA6 0 31
regSDMA0_QUEUE6_MIDCMD_DATA7 = 0x2d7
# 	DATA7 0 31
regSDMA0_QUEUE6_MIDCMD_DATA8 = 0x2d8
# 	DATA8 0 31
regSDMA0_QUEUE6_MIDCMD_DATA9 = 0x2d9
# 	DATA9 0 31
regSDMA0_QUEUE6_MINOR_PTR_UPDATE = 0x2c5
# 	ENABLE 0 0
regSDMA0_QUEUE6_PREEMPT = 0x2c0
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE6_RB_AQL_CNTL = 0x2c4
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE6_RB_BASE = 0x291
# 	ADDR 0 31
regSDMA0_QUEUE6_RB_BASE_HI = 0x292
# 	ADDR 0 23
regSDMA0_QUEUE6_RB_CNTL = 0x290
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE6_RB_PREEMPT = 0x2c6
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE6_RB_RPTR = 0x293
# 	OFFSET 0 31
regSDMA0_QUEUE6_RB_RPTR_ADDR_HI = 0x298
# 	ADDR 0 31
regSDMA0_QUEUE6_RB_RPTR_ADDR_LO = 0x299
# 	ADDR 2 31
regSDMA0_QUEUE6_RB_RPTR_HI = 0x294
# 	OFFSET 0 31
regSDMA0_QUEUE6_RB_WPTR = 0x295
# 	OFFSET 0 31
regSDMA0_QUEUE6_RB_WPTR_HI = 0x296
# 	OFFSET 0 31
regSDMA0_QUEUE6_RB_WPTR_POLL_ADDR_HI = 0x2c2
# 	ADDR 0 31
regSDMA0_QUEUE6_RB_WPTR_POLL_ADDR_LO = 0x2c3
# 	ADDR 2 31
regSDMA0_QUEUE6_SCHEDULE_CNTL = 0x2be
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE6_SKIP_CNTL = 0x2a0
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE7_CONTEXT_STATUS = 0x2f9
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA0_QUEUE7_CSA_ADDR_HI = 0x315
# 	ADDR 0 31
regSDMA0_QUEUE7_CSA_ADDR_LO = 0x314
# 	ADDR 2 31
regSDMA0_QUEUE7_DOORBELL = 0x2fa
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA0_QUEUE7_DOORBELL_LOG = 0x311
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA0_QUEUE7_DOORBELL_OFFSET = 0x313
# 	OFFSET 2 27
regSDMA0_QUEUE7_DUMMY_REG = 0x319
# 	DUMMY 0 31
regSDMA0_QUEUE7_IB_BASE_HI = 0x2f6
# 	ADDR 0 31
regSDMA0_QUEUE7_IB_BASE_LO = 0x2f5
# 	ADDR 5 31
regSDMA0_QUEUE7_IB_CNTL = 0x2f2
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA0_QUEUE7_IB_OFFSET = 0x2f4
# 	OFFSET 2 21
regSDMA0_QUEUE7_IB_RPTR = 0x2f3
# 	OFFSET 2 21
regSDMA0_QUEUE7_IB_SIZE = 0x2f7
# 	SIZE 0 19
regSDMA0_QUEUE7_IB_SUB_REMAIN = 0x317
# 	SIZE 0 13
regSDMA0_QUEUE7_MIDCMD_CNTL = 0x333
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA0_QUEUE7_MIDCMD_DATA0 = 0x328
# 	DATA0 0 31
regSDMA0_QUEUE7_MIDCMD_DATA1 = 0x329
# 	DATA1 0 31
regSDMA0_QUEUE7_MIDCMD_DATA10 = 0x332
# 	DATA10 0 31
regSDMA0_QUEUE7_MIDCMD_DATA2 = 0x32a
# 	DATA2 0 31
regSDMA0_QUEUE7_MIDCMD_DATA3 = 0x32b
# 	DATA3 0 31
regSDMA0_QUEUE7_MIDCMD_DATA4 = 0x32c
# 	DATA4 0 31
regSDMA0_QUEUE7_MIDCMD_DATA5 = 0x32d
# 	DATA5 0 31
regSDMA0_QUEUE7_MIDCMD_DATA6 = 0x32e
# 	DATA6 0 31
regSDMA0_QUEUE7_MIDCMD_DATA7 = 0x32f
# 	DATA7 0 31
regSDMA0_QUEUE7_MIDCMD_DATA8 = 0x330
# 	DATA8 0 31
regSDMA0_QUEUE7_MIDCMD_DATA9 = 0x331
# 	DATA9 0 31
regSDMA0_QUEUE7_MINOR_PTR_UPDATE = 0x31d
# 	ENABLE 0 0
regSDMA0_QUEUE7_PREEMPT = 0x318
# 	IB_PREEMPT 0 0
regSDMA0_QUEUE7_RB_AQL_CNTL = 0x31c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA0_QUEUE7_RB_BASE = 0x2e9
# 	ADDR 0 31
regSDMA0_QUEUE7_RB_BASE_HI = 0x2ea
# 	ADDR 0 23
regSDMA0_QUEUE7_RB_CNTL = 0x2e8
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA0_QUEUE7_RB_PREEMPT = 0x31e
# 	PREEMPT_REQ 0 0
regSDMA0_QUEUE7_RB_RPTR = 0x2eb
# 	OFFSET 0 31
regSDMA0_QUEUE7_RB_RPTR_ADDR_HI = 0x2f0
# 	ADDR 0 31
regSDMA0_QUEUE7_RB_RPTR_ADDR_LO = 0x2f1
# 	ADDR 2 31
regSDMA0_QUEUE7_RB_RPTR_HI = 0x2ec
# 	OFFSET 0 31
regSDMA0_QUEUE7_RB_WPTR = 0x2ed
# 	OFFSET 0 31
regSDMA0_QUEUE7_RB_WPTR_HI = 0x2ee
# 	OFFSET 0 31
regSDMA0_QUEUE7_RB_WPTR_POLL_ADDR_HI = 0x31a
# 	ADDR 0 31
regSDMA0_QUEUE7_RB_WPTR_POLL_ADDR_LO = 0x31b
# 	ADDR 2 31
regSDMA0_QUEUE7_SCHEDULE_CNTL = 0x316
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA0_QUEUE7_SKIP_CNTL = 0x2f8
# 	SKIP_COUNT 0 19
regSDMA0_QUEUE_RESET_REQ = 0x7b
# 	QUEUE0_RESET 0 0
# 	QUEUE1_RESET 1 1
# 	QUEUE2_RESET 2 2
# 	QUEUE3_RESET 3 3
# 	QUEUE4_RESET 4 4
# 	QUEUE5_RESET 5 5
# 	QUEUE6_RESET 6 6
# 	QUEUE7_RESET 7 7
# 	RESERVED 8 31
regSDMA0_QUEUE_STATUS0 = 0x2f
# 	QUEUE0_STATUS 0 3
# 	QUEUE1_STATUS 4 7
# 	QUEUE2_STATUS 8 11
# 	QUEUE3_STATUS 12 15
# 	QUEUE4_STATUS 16 19
# 	QUEUE5_STATUS 20 23
# 	QUEUE6_STATUS 24 27
# 	QUEUE7_STATUS 28 31
regSDMA0_RB_RPTR_FETCH = 0x20
# 	OFFSET 2 31
regSDMA0_RB_RPTR_FETCH_HI = 0x21
# 	OFFSET 0 31
regSDMA0_RELAX_ORDERING_LUT = 0x4a
# 	RESERVED0 0 0
# 	COPY 1 1
# 	WRITE 2 2
# 	RESERVED3 3 3
# 	RESERVED4 4 4
# 	FENCE 5 5
# 	RESERVED76 6 7
# 	POLL_MEM 8 8
# 	COND_EXE 9 9
# 	ATOMIC 10 10
# 	CONST_FILL 11 11
# 	PTEPDE 12 12
# 	TIMESTAMP 13 13
# 	RESERVED 14 26
# 	WORLD_SWITCH 27 27
# 	RPTR_WRB 28 28
# 	WPTR_POLL 29 29
# 	IB_FETCH 30 30
# 	RB_FETCH 31 31
regSDMA0_RLC_CGCG_CTRL = 0x5c
# 	CGCG_INT_ENABLE 1 1
# 	CGCG_IDLE_HYSTERESIS 16 31
regSDMA0_SCRATCH_RAM_ADDR = 0x78
# 	ADDR 0 6
regSDMA0_SCRATCH_RAM_DATA = 0x77
# 	DATA 0 31
regSDMA0_SEM_WAIT_FAIL_TIMER_CNTL = 0x22
# 	TIMER 0 31
regSDMA0_STATUS1_REG = 0x26
# 	CE_WREQ_IDLE 0 0
# 	CE_WR_IDLE 1 1
# 	CE_SPLIT_IDLE 2 2
# 	CE_RREQ_IDLE 3 3
# 	CE_OUT_IDLE 4 4
# 	CE_IN_IDLE 5 5
# 	CE_DST_IDLE 6 6
# 	CE_CMD_IDLE 9 9
# 	CE_AFIFO_FULL 10 10
# 	CE_INFO_FULL 11 11
# 	CE_INFO1_FULL 12 12
# 	EX_START 13 13
# 	CE_RD_STALL 15 15
# 	CE_WR_STALL 16 16
# 	SEC_INTR_STATUS 17 17
# 	WPTR_POLL_IDLE 18 18
# 	SDMA_IDLE 19 19
regSDMA0_STATUS2_REG = 0x38
# 	ID 0 1
# 	TH0F32_INSTR_PTR 2 15
# 	CMD_OP 16 31
regSDMA0_STATUS3_REG = 0x4c
# 	CMD_OP_STATUS 0 15
# 	PREV_VM_CMD 16 19
# 	EXCEPTION_IDLE 20 20
# 	AQL_PREV_CMD_IDLE 21 21
# 	TLBI_IDLE 22 22
# 	GCR_IDLE 23 23
# 	INVREQ_IDLE 24 24
# 	QUEUE_ID_MATCH 25 25
# 	INT_QUEUE_ID 26 29
# 	TMZ_MTYPE_STATUS 30 31
regSDMA0_STATUS4_REG = 0x76
# 	IDLE 0 0
# 	IH_OUTSTANDING 2 2
# 	SEM_OUTSTANDING 3 3
# 	CH_RD_OUTSTANDING 4 4
# 	CH_WR_OUTSTANDING 5 5
# 	GCR_OUTSTANDING 6 6
# 	TLBI_OUTSTANDING 7 7
# 	UTCL2_RD_OUTSTANDING 8 8
# 	UTCL2_WR_OUTSTANDING 9 9
# 	REG_POLLING 10 10
# 	MEM_POLLING 11 11
# 	RESERVED_13_12 12 13
# 	RESERVED_15_14 14 15
# 	ACTIVE_QUEUE_ID 16 19
# 	SRIOV_WATING_RLCV_CMD 20 20
# 	SRIOV_SDMA_EXECUTING_CMD 21 21
# 	UTCL2_RD_XNACK_FAULT 22 22
# 	UTCL2_RD_XNACK_NULL 23 23
# 	UTCL2_RD_XNACK_TIMEOUT 24 24
# 	UTCL2_WR_XNACK_FAULT 25 25
# 	UTCL2_WR_XNACK_NULL 26 26
# 	UTCL2_WR_XNACK_TIMEOUT 27 27
regSDMA0_STATUS5_REG = 0x7a
# 	QUEUE0_RB_ENABLE_STATUS 0 0
# 	QUEUE1_RB_ENABLE_STATUS 1 1
# 	QUEUE2_RB_ENABLE_STATUS 2 2
# 	QUEUE3_RB_ENABLE_STATUS 3 3
# 	QUEUE4_RB_ENABLE_STATUS 4 4
# 	QUEUE5_RB_ENABLE_STATUS 5 5
# 	QUEUE6_RB_ENABLE_STATUS 6 6
# 	QUEUE7_RB_ENABLE_STATUS 7 7
# 	ACTIVE_QUEUE_ID 16 19
# 	QUEUE0_WPTR_POLL_PAGE_EXCEPTION 20 20
# 	QUEUE1_WPTR_POLL_PAGE_EXCEPTION 21 21
# 	QUEUE2_WPTR_POLL_PAGE_EXCEPTION 22 22
# 	QUEUE3_WPTR_POLL_PAGE_EXCEPTION 23 23
# 	QUEUE4_WPTR_POLL_PAGE_EXCEPTION 24 24
# 	QUEUE5_WPTR_POLL_PAGE_EXCEPTION 25 25
# 	QUEUE6_WPTR_POLL_PAGE_EXCEPTION 26 26
# 	QUEUE7_WPTR_POLL_PAGE_EXCEPTION 27 27
regSDMA0_STATUS6_REG = 0x7c
# 	ID 0 1
# 	TH1F32_INSTR_PTR 2 15
# 	TH1_EXCEPTION 16 31
regSDMA0_STATUS_REG = 0x25
# 	IDLE 0 0
# 	REG_IDLE 1 1
# 	RB_EMPTY 2 2
# 	RB_FULL 3 3
# 	RB_CMD_IDLE 4 4
# 	RB_CMD_FULL 5 5
# 	IB_CMD_IDLE 6 6
# 	IB_CMD_FULL 7 7
# 	BLOCK_IDLE 8 8
# 	INSIDE_IB 9 9
# 	EX_IDLE 10 10
# 	CGCG_FENCE 11 11
# 	PACKET_READY 12 12
# 	MC_WR_IDLE 13 13
# 	SRBM_IDLE 14 14
# 	CONTEXT_EMPTY 15 15
# 	DELTA_RPTR_FULL 16 16
# 	RB_MC_RREQ_IDLE 17 17
# 	IB_MC_RREQ_IDLE 18 18
# 	MC_RD_IDLE 19 19
# 	DELTA_RPTR_EMPTY 20 20
# 	MC_RD_RET_STALL 21 21
# 	MC_RD_NO_POLL_IDLE 22 22
# 	PREV_CMD_IDLE 25 25
# 	SEM_IDLE 26 26
# 	SEM_REQ_STALL 27 27
# 	SEM_RESP_STATE 28 29
# 	INT_IDLE 30 30
# 	INT_REQ_STALL 31 31
regSDMA0_TILING_CONFIG = 0x63
# 	PIPE_INTERLEAVE_SIZE 4 6
regSDMA0_TIMESTAMP_CNTL = 0x79
# 	CAPTURE 0 0
regSDMA0_TLBI_GCR_CNTL = 0x62
# 	TLBI_CMD_DW 0 3
# 	GCR_CMD_DW 4 7
# 	GCR_CLKEN_CYCLE 8 11
# 	TLBI_CREDIT 16 23
# 	GCR_CREDIT 24 31
regSDMA0_UCODE1_CHECKSUM = 0x7d
# 	DATA 0 31
regSDMA0_UCODE_ADDR = 0x5880
# 	VALUE 0 12
# 	THID 15 15
regSDMA0_UCODE_CHECKSUM = 0x29
# 	DATA 0 31
regSDMA0_UCODE_DATA = 0x5881
# 	VALUE 0 31
regSDMA0_UCODE_SELFLOAD_CONTROL = 0x5882
regSDMA0_UTCL1_CNTL = 0x3c
# 	REDO_DELAY 0 4
# 	PAGE_WAIT_DELAY 5 8
# 	RESP_MODE 9 10
# 	FORCE_INVALIDATION 14 14
# 	FORCE_INVREQ_HEAVY 15 15
# 	WR_EXE_PERMS_CTRL 16 16
# 	RD_EXE_PERMS_CTRL 17 17
# 	INVACK_DELAY 18 21
# 	REQL2_CREDIT 24 29
regSDMA0_UTCL1_INV0 = 0x42
# 	INV_PROC_BUSY 0 0
# 	GPUVM_FRAG_SIZE 1 6
# 	GPUVM_VMID 7 10
# 	GPUVM_MODE 11 12
# 	GPUVM_HIGH 13 13
# 	GPUVM_TAG 14 17
# 	GPUVM_VMID_HIGH 18 21
# 	GPUVM_VMID_LOW 22 25
# 	INV_TYPE 26 27
regSDMA0_UTCL1_INV1 = 0x43
# 	INV_ADDR_LO 0 31
regSDMA0_UTCL1_INV2 = 0x44
# 	CPF_VMID 0 15
# 	CPF_FLUSH_TYPE 16 16
# 	CPF_FRAG_SIZE 17 22
regSDMA0_UTCL1_PAGE = 0x3f
# 	VM_HOLE 0 0
# 	REQ_TYPE 1 4
# 	USE_MTYPE 6 9
# 	USE_PT_SNOOP 10 10
# 	USE_IO 11 11
# 	RD_L2_POLICY 12 13
# 	WR_L2_POLICY 14 15
# 	DMA_PAGE_SIZE 16 21
# 	USE_BC 22 22
# 	ADDR_IS_PA 23 23
# 	LLC_NOALLOC 24 24
regSDMA0_UTCL1_RD_STATUS = 0x40
# 	RD_VA_FIFO_EMPTY 0 0
# 	RD_REG_ENTRY_EMPTY 1 1
# 	RD_PAGE_FIFO_EMPTY 2 2
# 	RD_REQ_FIFO_EMPTY 3 3
# 	RD_VA_REQ_FIFO_EMPTY 4 4
# 	RESERVED0 5 5
# 	RESERVED1 6 6
# 	META_Q_EMPTY 7 7
# 	RD_VA_FIFO_FULL 8 8
# 	RD_REG_ENTRY_FULL 9 9
# 	RD_PAGE_FIFO_FULL 10 10
# 	RD_REQ_FIFO_FULL 11 11
# 	RD_VA_REQ_FIFO_FULL 12 12
# 	RESERVED2 13 13
# 	RESERVED3 14 14
# 	META_Q_FULL 15 15
# 	RD_L2_INTF_IDLE 16 16
# 	RD_REQRET_IDLE 17 17
# 	RD_REQ_IDLE 18 18
# 	RD_MERGE_TYPE 19 20
# 	RD_MERGE_DATA_PA_READY 21 21
# 	RD_MERGE_META_PA_READY 22 22
# 	RD_MERGE_REG_READY 23 23
# 	RD_MERGE_PAGE_FIFO_READY 24 24
# 	RD_MERGE_REQ_FIFO_READY 25 25
# 	RESERVED4 26 26
# 	RD_MERGE_OUT_RTR 27 27
# 	RDREQ_IN_RTR 28 28
# 	RDREQ_OUT_RTR 29 29
# 	INV_BUSY 30 30
# 	DBIT_REQ_IDLE 31 31
regSDMA0_UTCL1_RD_XNACK0 = 0x45
# 	XNACK_FAULT_ADDR_LO 0 31
regSDMA0_UTCL1_RD_XNACK1 = 0x46
# 	XNACK_FAULT_ADDR_HI 0 3
# 	XNACK_FAULT_VMID 4 7
# 	XNACK_FAULT_VECTOR 8 9
# 	XNACK_NULL_VECTOR 10 11
# 	XNACK_TIMEOUT_VECTOR 12 13
# 	XNACK_FAULT_FLAG 14 14
# 	XNACK_NULL_FLAG 15 15
# 	XNACK_TIMEOUT_FLAG 16 16
regSDMA0_UTCL1_TIMEOUT = 0x3e
# 	XNACK_LIMIT 0 15
regSDMA0_UTCL1_WATERMK = 0x3d
# 	WR_REQ_FIFO_WATERMK 0 3
# 	WR_REQ_FIFO_DEPTH_STEP 4 5
# 	RD_REQ_FIFO_WATERMK 6 9
# 	RD_REQ_FIFO_DEPTH_STEP 10 11
# 	WR_PAGE_FIFO_WATERMK 12 15
# 	WR_PAGE_FIFO_DEPTH_STEP 16 17
# 	RD_PAGE_FIFO_WATERMK 18 21
# 	RD_PAGE_FIFO_DEPTH_STEP 22 23
regSDMA0_UTCL1_WR_STATUS = 0x41
# 	WR_VA_FIFO_EMPTY 0 0
# 	WR_REG_ENTRY_EMPTY 1 1
# 	WR_PAGE_FIFO_EMPTY 2 2
# 	WR_REQ_FIFO_EMPTY 3 3
# 	WR_VA_REQ_FIFO_EMPTY 4 4
# 	WR_DATA2_EMPTY 5 5
# 	WR_DATA1_EMPTY 6 6
# 	RESERVED0 7 7
# 	WR_VA_FIFO_FULL 8 8
# 	WR_REG_ENTRY_FULL 9 9
# 	WR_PAGE_FIFO_FULL 10 10
# 	WR_REQ_FIFO_FULL 11 11
# 	WR_VA_REQ_FIFO_FULL 12 12
# 	WR_DATA2_FULL 13 13
# 	WR_DATA1_FULL 14 14
# 	F32_WR_RTR 15 15
# 	WR_L2_INTF_IDLE 16 16
# 	WR_REQRET_IDLE 17 17
# 	WR_REQ_IDLE 18 18
# 	WR_MERGE_TYPE 19 20
# 	WR_MERGE_DATA_PA_READY 21 21
# 	WR_MERGE_META_PA_READY 22 22
# 	WR_MERGE_REG_READY 23 23
# 	WR_MERGE_PAGE_FIFO_READY 24 24
# 	WR_MERGE_REQ_FIFO_READY 25 25
# 	WR_MERGE_DATA_SEL 26 26
# 	WR_MERGE_OUT_RTR 27 27
# 	WRREQ_IN_RTR 28 28
# 	WRREQ_OUT_RTR 29 29
# 	WRREQ_IN_DATA1_RTR 30 30
# 	WRREQ_IN_DATA2_RTR 31 31
regSDMA0_UTCL1_WR_XNACK0 = 0x47
# 	XNACK_FAULT_ADDR_LO 0 31
regSDMA0_UTCL1_WR_XNACK1 = 0x48
# 	XNACK_FAULT_ADDR_HI 0 3
# 	XNACK_FAULT_VMID 4 7
# 	XNACK_FAULT_VECTOR 8 9
# 	XNACK_NULL_VECTOR 10 11
# 	XNACK_TIMEOUT_VECTOR 12 13
# 	XNACK_FAULT_FLAG 14 14
# 	XNACK_NULL_FLAG 15 15
# 	XNACK_TIMEOUT_FLAG 16 16
regSDMA0_VERSION = 0x35
# 	MINVER 0 6
# 	MAJVER 8 14
# 	REV 16 21
regSDMA0_WATCHDOG_CNTL = 0x2e
# 	QUEUE_HANG_COUNT 0 7
# 	CMD_TIMEOUT_COUNT 8 15
regSDMA1_AQL_STATUS = 0x65f
# 	COMPLETE_SIGNAL_EMPTY 0 0
# 	INVALID_CMD_EMPTY 1 1
regSDMA1_ATOMIC_CNTL = 0x639
# 	LOOP_TIMER 0 30
# 	ATOMIC_RTN_INT_ENABLE 31 31
regSDMA1_ATOMIC_PREOP_HI = 0x63b
# 	DATA 0 31
regSDMA1_ATOMIC_PREOP_LO = 0x63a
# 	DATA 0 31
regSDMA1_BA_THRESHOLD = 0x633
# 	READ_THRES 0 9
# 	WRITE_THRES 16 25
regSDMA1_BROADCAST_UCODE_ADDR = 0x58a6
# 	VALUE 0 12
# 	THID 15 15
regSDMA1_BROADCAST_UCODE_DATA = 0x58a7
# 	VALUE 0 31
regSDMA1_CE_CTRL = 0x67e
# 	RD_LUT_WATERMARK 0 2
# 	RD_LUT_DEPTH 3 4
# 	WR_AFIFO_WATERMARK 5 7
# 	CE_DCC_READ_128B_ENABLE 8 8
# 	RESERVED 9 31
regSDMA1_CHICKEN_BITS = 0x61d
# 	STALL_ON_TRANS_FULL_ENABLE 1 1
# 	STALL_ON_NO_FREE_DATA_BUFFER_ENABLE 2 2
# 	SRBM_POLL_RETRYING 5 5
# 	RD_BURST 6 7
# 	WR_BURST 8 9
# 	COMBINE_256B_WAIT_CYCLE 10 13
# 	WR_COMBINE_256B_ENABLE 14 14
# 	RD_COMBINE_256B_ENABLE 15 15
# 	COPY_OVERLAP_ENABLE 16 16
# 	RAW_CHECK_ENABLE 17 17
# 	T2L_256B_ENABLE 18 18
# 	SOFT_OVERRIDE_GCR_FGCG 19 19
# 	SOFT_OVERRIDE_GRBM_FGCG 20 20
# 	SOFT_OVERRIDE_CH_FGCG 21 21
# 	SOFT_OVERRIDE_UTCL2_INVREQ_FGCG 22 22
# 	SOFT_OVERRIDE_UTCL1_FGCG 23 23
# 	CG_STATUS_OUTPUT 24 24
# 	SW_FREEZE_ENABLE 25 25
# 	RESERVED 26 31
regSDMA1_CHICKEN_BITS_2 = 0x64b
# 	F32_CMD_PROC_DELAY 0 3
# 	F32_SEND_POSTCODE_EN 4 4
# 	UCODE_BUF_DS_EN 6 6
# 	UCODE_SELFLOAD_THREAD_OVERLAP 7 7
# 	WPTR_POLL_OUTSTANDING 8 11
# 	RESERVED_14_12 12 14
# 	RESERVED_15 15 15
# 	RB_FIFO_WATERMARK 16 17
# 	IB_FIFO_WATERMARK 18 19
# 	RESERVED_22_20 20 22
# 	CH_RD_WATERMARK 23 24
# 	CH_WR_WATERMARK 25 29
# 	CH_WR_WATERMARK_LSB 30 30
# 	PIO_VFID_SOURCE 31 31
regSDMA1_CLOCK_GATING_STATUS = 0x675
# 	DYN_CLK_GATE_STATUS 0 0
# 	CE_CLK_GATE_STATUS 2 2
# 	CE_BC_CLK_GATE_STATUS 3 3
# 	CE_NBC_CLK_GATE_STATUS 4 4
# 	REG_CLK_GATE_STATUS 5 5
# 	F32_CLK_GATE_STATUS 6 6
regSDMA1_CNTL = 0x61c
# 	TRAP_ENABLE 0 0
# 	SEM_WAIT_INT_ENABLE 2 2
# 	DATA_SWAP_ENABLE 3 3
# 	FENCE_SWAP_ENABLE 4 4
# 	MIDCMD_PREEMPT_ENABLE 5 5
# 	PIO_DONE_ACK_ENABLE 6 6
# 	TMZ_MIDCMD_PREEMPT_ENABLE 8 8
# 	MIDCMD_EXPIRE_ENABLE 9 9
# 	CP_MES_INT_ENABLE 10 10
# 	PAGE_RETRY_TIMEOUT_INT_ENABLE 11 11
# 	PAGE_NULL_INT_ENABLE 12 12
# 	PAGE_FAULT_INT_ENABLE 13 13
# 	CH_PERFCNT_ENABLE 16 16
# 	MIDCMD_WORLDSWITCH_ENABLE 17 17
# 	CTXEMPTY_INT_ENABLE 28 28
# 	FROZEN_INT_ENABLE 29 29
# 	IB_PREEMPT_INT_ENABLE 30 30
# 	RB_PREEMPT_INT_ENABLE 31 31
regSDMA1_CNTL1 = 0x627
# 	WPTR_POLL_FREQUENCY 2 15
regSDMA1_CRD_CNTL = 0x65b
# 	MC_WRREQ_CREDIT 7 12
# 	MC_RDREQ_CREDIT 13 18
# 	CH_WRREQ_CREDIT 19 24
# 	CH_RDREQ_CREDIT 25 30
regSDMA1_DEC_START = 0x600
# 	START 0 31
regSDMA1_EA_DBIT_ADDR_DATA = 0x660
# 	VALUE 0 31
regSDMA1_EA_DBIT_ADDR_INDEX = 0x661
# 	VALUE 0 2
regSDMA1_EDC_CONFIG = 0x632
# 	DIS_EDC 1 1
# 	ECC_INT_ENABLE 2 2
regSDMA1_EDC_COUNTER = 0x636
# 	SDMA_UCODE_BUF_DED 0 0
# 	SDMA_UCODE_BUF_SEC 1 1
# 	SDMA_RB_CMD_BUF_SED 2 2
# 	SDMA_IB_CMD_BUF_SED 3 3
# 	SDMA_UTCL1_RD_FIFO_SED 4 4
# 	SDMA_UTCL1_RDBST_FIFO_SED 5 5
# 	SDMA_DATA_LUT_FIFO_SED 6 6
# 	SDMA_MBANK_DATA_BUF0_SED 7 7
# 	SDMA_MBANK_DATA_BUF1_SED 8 8
# 	SDMA_MBANK_DATA_BUF2_SED 9 9
# 	SDMA_MBANK_DATA_BUF3_SED 10 10
# 	SDMA_MBANK_DATA_BUF4_SED 11 11
# 	SDMA_MBANK_DATA_BUF5_SED 12 12
# 	SDMA_MBANK_DATA_BUF6_SED 13 13
# 	SDMA_MBANK_DATA_BUF7_SED 14 14
# 	SDMA_SPLIT_DAT_BUF_SED 15 15
# 	SDMA_MC_WR_ADDR_FIFO_SED 16 16
regSDMA1_EDC_COUNTER_CLEAR = 0x637
# 	DUMMY 0 0
regSDMA1_ERROR_LOG = 0x650
# 	OVERRIDE 0 15
# 	STATUS 16 31
regSDMA1_F32_CNTL = 0x58ba
# 	HALT 0 0
# 	TH0_CHECKSUM_CLR 8 8
# 	TH0_RESET 9 9
# 	TH0_ENABLE 10 10
# 	TH1_CHECKSUM_CLR 12 12
# 	TH1_RESET 13 13
# 	TH1_ENABLE 14 14
# 	TH0_PRIORITY 16 23
# 	TH1_PRIORITY 24 31
regSDMA1_F32_COUNTER = 0x655
# 	VALUE 0 31
regSDMA1_F32_MISC_CNTL = 0x60b
# 	F32_WAKEUP 0 0
regSDMA1_FED_STATUS = 0x67f
# 	RB_FETCH_ECC 0 0
# 	IB_FETCH_ECC 1 1
# 	F32_DATA_ECC 2 2
# 	WPTR_ATOMIC_ECC 3 3
# 	COPY_DATA_ECC 4 4
# 	COPY_METADATA_ECC 5 5
# 	SELFLOAD_UCODE_ECC 6 6
regSDMA1_FREEZE = 0x62b
# 	PREEMPT 0 0
# 	FREEZE 4 4
# 	FROZEN 5 5
# 	F32_FREEZE 6 6
regSDMA1_GB_ADDR_CONFIG = 0x61e
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regSDMA1_GB_ADDR_CONFIG_READ = 0x61f
# 	NUM_PIPES 0 2
# 	PIPE_INTERLEAVE_SIZE 3 5
# 	MAX_COMPRESSED_FRAGS 6 7
# 	NUM_PKRS 8 10
# 	NUM_SHADER_ENGINES 19 20
# 	NUM_RB_PER_SE 26 27
regSDMA1_GLOBAL_QUANTUM = 0x64f
# 	GLOBAL_FOCUS_QUANTUM 0 7
# 	GLOBAL_NORMAL_QUANTUM 8 15
regSDMA1_GLOBAL_TIMESTAMP_HI = 0x610
# 	DATA 0 31
regSDMA1_GLOBAL_TIMESTAMP_LO = 0x60f
# 	DATA 0 31
regSDMA1_HBM_PAGE_CONFIG = 0x628
# 	PAGE_SIZE_EXPONENT 0 1
regSDMA1_HOLE_ADDR_HI = 0x673
# 	VALUE 0 31
regSDMA1_HOLE_ADDR_LO = 0x672
# 	VALUE 0 31
regSDMA1_IB_OFFSET_FETCH = 0x623
# 	OFFSET 2 21
regSDMA1_ID = 0x634
# 	DEVICE_ID 0 7
regSDMA1_INT_STATUS = 0x670
# 	DATA 0 31
regSDMA1_PERFCNT_MISC_CNTL = 0x3e2f
# 	CMD_OP 0 15
regSDMA1_PERFCNT_PERFCOUNTER0_CFG = 0x3e2c
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regSDMA1_PERFCNT_PERFCOUNTER1_CFG = 0x3e2d
# 	PERF_SEL 0 7
# 	PERF_SEL_END 8 15
# 	PERF_MODE 24 27
# 	ENABLE 28 28
# 	CLEAR 29 29
regSDMA1_PERFCNT_PERFCOUNTER_HI = 0x366d
# 	COUNTER_HI 0 15
# 	COMPARE_VALUE 16 31
regSDMA1_PERFCNT_PERFCOUNTER_LO = 0x366c
# 	COUNTER_LO 0 31
regSDMA1_PERFCNT_PERFCOUNTER_RSLT_CNTL = 0x3e2e
# 	PERF_COUNTER_SELECT 0 3
# 	START_TRIGGER 8 15
# 	STOP_TRIGGER 16 23
# 	ENABLE_ANY 24 24
# 	CLEAR_ALL 25 25
# 	STOP_ALL_ON_SATURATE 26 26
regSDMA1_PERFCOUNTER0_HI = 0x366f
# 	PERFCOUNTER_HI 0 31
regSDMA1_PERFCOUNTER0_LO = 0x366e
# 	PERFCOUNTER_LO 0 31
regSDMA1_PERFCOUNTER0_SELECT = 0x3e30
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSDMA1_PERFCOUNTER0_SELECT1 = 0x3e31
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSDMA1_PERFCOUNTER1_HI = 0x3671
# 	PERFCOUNTER_HI 0 31
regSDMA1_PERFCOUNTER1_LO = 0x3670
# 	PERFCOUNTER_LO 0 31
regSDMA1_PERFCOUNTER1_SELECT = 0x3e32
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSDMA1_PERFCOUNTER1_SELECT1 = 0x3e33
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSDMA1_PHYSICAL_ADDR_HI = 0x64e
# 	ADDR 0 15
regSDMA1_PHYSICAL_ADDR_LO = 0x64d
# 	D_VALID 0 0
# 	DIRTY 1 1
# 	PHY_VALID 2 2
# 	ADDR 12 31
regSDMA1_POWER_CNTL = 0x61a
# 	LS_ENABLE 8 8
regSDMA1_PROCESS_QUANTUM0 = 0x62c
# 	PROCESS0_QUANTUM 0 7
# 	PROCESS1_QUANTUM 8 15
# 	PROCESS2_QUANTUM 16 23
# 	PROCESS3_QUANTUM 24 31
regSDMA1_PROCESS_QUANTUM1 = 0x62d
# 	PROCESS4_QUANTUM 0 7
# 	PROCESS5_QUANTUM 8 15
# 	PROCESS6_QUANTUM 16 23
# 	PROCESS7_QUANTUM 24 31
regSDMA1_PROGRAM = 0x624
# 	STREAM 0 31
regSDMA1_PUB_DUMMY_REG0 = 0x651
# 	VALUE 0 31
regSDMA1_PUB_DUMMY_REG1 = 0x652
# 	VALUE 0 31
regSDMA1_PUB_DUMMY_REG2 = 0x653
# 	VALUE 0 31
regSDMA1_PUB_DUMMY_REG3 = 0x654
# 	VALUE 0 31
regSDMA1_QUEUE0_CONTEXT_STATUS = 0x691
# 	SELECTED 0 0
# 	USE_IB 1 1
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE0_CSA_ADDR_HI = 0x6ad
# 	ADDR 0 31
regSDMA1_QUEUE0_CSA_ADDR_LO = 0x6ac
# 	ADDR 2 31
regSDMA1_QUEUE0_DOORBELL = 0x692
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE0_DOORBELL_LOG = 0x6a9
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE0_DOORBELL_OFFSET = 0x6ab
# 	OFFSET 2 27
regSDMA1_QUEUE0_DUMMY_REG = 0x6b1
# 	DUMMY 0 31
regSDMA1_QUEUE0_IB_BASE_HI = 0x68e
# 	ADDR 0 31
regSDMA1_QUEUE0_IB_BASE_LO = 0x68d
# 	ADDR 5 31
regSDMA1_QUEUE0_IB_CNTL = 0x68a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE0_IB_OFFSET = 0x68c
# 	OFFSET 2 21
regSDMA1_QUEUE0_IB_RPTR = 0x68b
# 	OFFSET 2 21
regSDMA1_QUEUE0_IB_SIZE = 0x68f
# 	SIZE 0 19
regSDMA1_QUEUE0_IB_SUB_REMAIN = 0x6af
# 	SIZE 0 13
regSDMA1_QUEUE0_MIDCMD_CNTL = 0x6cb
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE0_MIDCMD_DATA0 = 0x6c0
# 	DATA0 0 31
regSDMA1_QUEUE0_MIDCMD_DATA1 = 0x6c1
# 	DATA1 0 31
regSDMA1_QUEUE0_MIDCMD_DATA10 = 0x6ca
# 	DATA10 0 31
regSDMA1_QUEUE0_MIDCMD_DATA2 = 0x6c2
# 	DATA2 0 31
regSDMA1_QUEUE0_MIDCMD_DATA3 = 0x6c3
# 	DATA3 0 31
regSDMA1_QUEUE0_MIDCMD_DATA4 = 0x6c4
# 	DATA4 0 31
regSDMA1_QUEUE0_MIDCMD_DATA5 = 0x6c5
# 	DATA5 0 31
regSDMA1_QUEUE0_MIDCMD_DATA6 = 0x6c6
# 	DATA6 0 31
regSDMA1_QUEUE0_MIDCMD_DATA7 = 0x6c7
# 	DATA7 0 31
regSDMA1_QUEUE0_MIDCMD_DATA8 = 0x6c8
# 	DATA8 0 31
regSDMA1_QUEUE0_MIDCMD_DATA9 = 0x6c9
# 	DATA9 0 31
regSDMA1_QUEUE0_MINOR_PTR_UPDATE = 0x6b5
# 	ENABLE 0 0
regSDMA1_QUEUE0_PREEMPT = 0x6b0
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE0_RB_AQL_CNTL = 0x6b4
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE0_RB_BASE = 0x681
# 	ADDR 0 31
regSDMA1_QUEUE0_RB_BASE_HI = 0x682
# 	ADDR 0 23
regSDMA1_QUEUE0_RB_CNTL = 0x680
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE0_RB_PREEMPT = 0x6b6
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE0_RB_RPTR = 0x683
# 	OFFSET 0 31
regSDMA1_QUEUE0_RB_RPTR_ADDR_HI = 0x688
# 	ADDR 0 31
regSDMA1_QUEUE0_RB_RPTR_ADDR_LO = 0x689
# 	ADDR 2 31
regSDMA1_QUEUE0_RB_RPTR_HI = 0x684
# 	OFFSET 0 31
regSDMA1_QUEUE0_RB_WPTR = 0x685
# 	OFFSET 0 31
regSDMA1_QUEUE0_RB_WPTR_HI = 0x686
# 	OFFSET 0 31
regSDMA1_QUEUE0_RB_WPTR_POLL_ADDR_HI = 0x6b2
# 	ADDR 0 31
regSDMA1_QUEUE0_RB_WPTR_POLL_ADDR_LO = 0x6b3
# 	ADDR 2 31
regSDMA1_QUEUE0_SCHEDULE_CNTL = 0x6ae
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE0_SKIP_CNTL = 0x690
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE1_CONTEXT_STATUS = 0x6e9
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE1_CSA_ADDR_HI = 0x705
# 	ADDR 0 31
regSDMA1_QUEUE1_CSA_ADDR_LO = 0x704
# 	ADDR 2 31
regSDMA1_QUEUE1_DOORBELL = 0x6ea
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE1_DOORBELL_LOG = 0x701
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE1_DOORBELL_OFFSET = 0x703
# 	OFFSET 2 27
regSDMA1_QUEUE1_DUMMY_REG = 0x709
# 	DUMMY 0 31
regSDMA1_QUEUE1_IB_BASE_HI = 0x6e6
# 	ADDR 0 31
regSDMA1_QUEUE1_IB_BASE_LO = 0x6e5
# 	ADDR 5 31
regSDMA1_QUEUE1_IB_CNTL = 0x6e2
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE1_IB_OFFSET = 0x6e4
# 	OFFSET 2 21
regSDMA1_QUEUE1_IB_RPTR = 0x6e3
# 	OFFSET 2 21
regSDMA1_QUEUE1_IB_SIZE = 0x6e7
# 	SIZE 0 19
regSDMA1_QUEUE1_IB_SUB_REMAIN = 0x707
# 	SIZE 0 13
regSDMA1_QUEUE1_MIDCMD_CNTL = 0x723
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE1_MIDCMD_DATA0 = 0x718
# 	DATA0 0 31
regSDMA1_QUEUE1_MIDCMD_DATA1 = 0x719
# 	DATA1 0 31
regSDMA1_QUEUE1_MIDCMD_DATA10 = 0x722
# 	DATA10 0 31
regSDMA1_QUEUE1_MIDCMD_DATA2 = 0x71a
# 	DATA2 0 31
regSDMA1_QUEUE1_MIDCMD_DATA3 = 0x71b
# 	DATA3 0 31
regSDMA1_QUEUE1_MIDCMD_DATA4 = 0x71c
# 	DATA4 0 31
regSDMA1_QUEUE1_MIDCMD_DATA5 = 0x71d
# 	DATA5 0 31
regSDMA1_QUEUE1_MIDCMD_DATA6 = 0x71e
# 	DATA6 0 31
regSDMA1_QUEUE1_MIDCMD_DATA7 = 0x71f
# 	DATA7 0 31
regSDMA1_QUEUE1_MIDCMD_DATA8 = 0x720
# 	DATA8 0 31
regSDMA1_QUEUE1_MIDCMD_DATA9 = 0x721
# 	DATA9 0 31
regSDMA1_QUEUE1_MINOR_PTR_UPDATE = 0x70d
# 	ENABLE 0 0
regSDMA1_QUEUE1_PREEMPT = 0x708
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE1_RB_AQL_CNTL = 0x70c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE1_RB_BASE = 0x6d9
# 	ADDR 0 31
regSDMA1_QUEUE1_RB_BASE_HI = 0x6da
# 	ADDR 0 23
regSDMA1_QUEUE1_RB_CNTL = 0x6d8
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE1_RB_PREEMPT = 0x70e
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE1_RB_RPTR = 0x6db
# 	OFFSET 0 31
regSDMA1_QUEUE1_RB_RPTR_ADDR_HI = 0x6e0
# 	ADDR 0 31
regSDMA1_QUEUE1_RB_RPTR_ADDR_LO = 0x6e1
# 	ADDR 2 31
regSDMA1_QUEUE1_RB_RPTR_HI = 0x6dc
# 	OFFSET 0 31
regSDMA1_QUEUE1_RB_WPTR = 0x6dd
# 	OFFSET 0 31
regSDMA1_QUEUE1_RB_WPTR_HI = 0x6de
# 	OFFSET 0 31
regSDMA1_QUEUE1_RB_WPTR_POLL_ADDR_HI = 0x70a
# 	ADDR 0 31
regSDMA1_QUEUE1_RB_WPTR_POLL_ADDR_LO = 0x70b
# 	ADDR 2 31
regSDMA1_QUEUE1_SCHEDULE_CNTL = 0x706
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE1_SKIP_CNTL = 0x6e8
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE2_CONTEXT_STATUS = 0x741
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE2_CSA_ADDR_HI = 0x75d
# 	ADDR 0 31
regSDMA1_QUEUE2_CSA_ADDR_LO = 0x75c
# 	ADDR 2 31
regSDMA1_QUEUE2_DOORBELL = 0x742
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE2_DOORBELL_LOG = 0x759
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE2_DOORBELL_OFFSET = 0x75b
# 	OFFSET 2 27
regSDMA1_QUEUE2_DUMMY_REG = 0x761
# 	DUMMY 0 31
regSDMA1_QUEUE2_IB_BASE_HI = 0x73e
# 	ADDR 0 31
regSDMA1_QUEUE2_IB_BASE_LO = 0x73d
# 	ADDR 5 31
regSDMA1_QUEUE2_IB_CNTL = 0x73a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE2_IB_OFFSET = 0x73c
# 	OFFSET 2 21
regSDMA1_QUEUE2_IB_RPTR = 0x73b
# 	OFFSET 2 21
regSDMA1_QUEUE2_IB_SIZE = 0x73f
# 	SIZE 0 19
regSDMA1_QUEUE2_IB_SUB_REMAIN = 0x75f
# 	SIZE 0 13
regSDMA1_QUEUE2_MIDCMD_CNTL = 0x77b
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE2_MIDCMD_DATA0 = 0x770
# 	DATA0 0 31
regSDMA1_QUEUE2_MIDCMD_DATA1 = 0x771
# 	DATA1 0 31
regSDMA1_QUEUE2_MIDCMD_DATA10 = 0x77a
# 	DATA10 0 31
regSDMA1_QUEUE2_MIDCMD_DATA2 = 0x772
# 	DATA2 0 31
regSDMA1_QUEUE2_MIDCMD_DATA3 = 0x773
# 	DATA3 0 31
regSDMA1_QUEUE2_MIDCMD_DATA4 = 0x774
# 	DATA4 0 31
regSDMA1_QUEUE2_MIDCMD_DATA5 = 0x775
# 	DATA5 0 31
regSDMA1_QUEUE2_MIDCMD_DATA6 = 0x776
# 	DATA6 0 31
regSDMA1_QUEUE2_MIDCMD_DATA7 = 0x777
# 	DATA7 0 31
regSDMA1_QUEUE2_MIDCMD_DATA8 = 0x778
# 	DATA8 0 31
regSDMA1_QUEUE2_MIDCMD_DATA9 = 0x779
# 	DATA9 0 31
regSDMA1_QUEUE2_MINOR_PTR_UPDATE = 0x765
# 	ENABLE 0 0
regSDMA1_QUEUE2_PREEMPT = 0x760
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE2_RB_AQL_CNTL = 0x764
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE2_RB_BASE = 0x731
# 	ADDR 0 31
regSDMA1_QUEUE2_RB_BASE_HI = 0x732
# 	ADDR 0 23
regSDMA1_QUEUE2_RB_CNTL = 0x730
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE2_RB_PREEMPT = 0x766
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE2_RB_RPTR = 0x733
# 	OFFSET 0 31
regSDMA1_QUEUE2_RB_RPTR_ADDR_HI = 0x738
# 	ADDR 0 31
regSDMA1_QUEUE2_RB_RPTR_ADDR_LO = 0x739
# 	ADDR 2 31
regSDMA1_QUEUE2_RB_RPTR_HI = 0x734
# 	OFFSET 0 31
regSDMA1_QUEUE2_RB_WPTR = 0x735
# 	OFFSET 0 31
regSDMA1_QUEUE2_RB_WPTR_HI = 0x736
# 	OFFSET 0 31
regSDMA1_QUEUE2_RB_WPTR_POLL_ADDR_HI = 0x762
# 	ADDR 0 31
regSDMA1_QUEUE2_RB_WPTR_POLL_ADDR_LO = 0x763
# 	ADDR 2 31
regSDMA1_QUEUE2_SCHEDULE_CNTL = 0x75e
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE2_SKIP_CNTL = 0x740
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE3_CONTEXT_STATUS = 0x799
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE3_CSA_ADDR_HI = 0x7b5
# 	ADDR 0 31
regSDMA1_QUEUE3_CSA_ADDR_LO = 0x7b4
# 	ADDR 2 31
regSDMA1_QUEUE3_DOORBELL = 0x79a
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE3_DOORBELL_LOG = 0x7b1
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE3_DOORBELL_OFFSET = 0x7b3
# 	OFFSET 2 27
regSDMA1_QUEUE3_DUMMY_REG = 0x7b9
# 	DUMMY 0 31
regSDMA1_QUEUE3_IB_BASE_HI = 0x796
# 	ADDR 0 31
regSDMA1_QUEUE3_IB_BASE_LO = 0x795
# 	ADDR 5 31
regSDMA1_QUEUE3_IB_CNTL = 0x792
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE3_IB_OFFSET = 0x794
# 	OFFSET 2 21
regSDMA1_QUEUE3_IB_RPTR = 0x793
# 	OFFSET 2 21
regSDMA1_QUEUE3_IB_SIZE = 0x797
# 	SIZE 0 19
regSDMA1_QUEUE3_IB_SUB_REMAIN = 0x7b7
# 	SIZE 0 13
regSDMA1_QUEUE3_MIDCMD_CNTL = 0x7d3
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE3_MIDCMD_DATA0 = 0x7c8
# 	DATA0 0 31
regSDMA1_QUEUE3_MIDCMD_DATA1 = 0x7c9
# 	DATA1 0 31
regSDMA1_QUEUE3_MIDCMD_DATA10 = 0x7d2
# 	DATA10 0 31
regSDMA1_QUEUE3_MIDCMD_DATA2 = 0x7ca
# 	DATA2 0 31
regSDMA1_QUEUE3_MIDCMD_DATA3 = 0x7cb
# 	DATA3 0 31
regSDMA1_QUEUE3_MIDCMD_DATA4 = 0x7cc
# 	DATA4 0 31
regSDMA1_QUEUE3_MIDCMD_DATA5 = 0x7cd
# 	DATA5 0 31
regSDMA1_QUEUE3_MIDCMD_DATA6 = 0x7ce
# 	DATA6 0 31
regSDMA1_QUEUE3_MIDCMD_DATA7 = 0x7cf
# 	DATA7 0 31
regSDMA1_QUEUE3_MIDCMD_DATA8 = 0x7d0
# 	DATA8 0 31
regSDMA1_QUEUE3_MIDCMD_DATA9 = 0x7d1
# 	DATA9 0 31
regSDMA1_QUEUE3_MINOR_PTR_UPDATE = 0x7bd
# 	ENABLE 0 0
regSDMA1_QUEUE3_PREEMPT = 0x7b8
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE3_RB_AQL_CNTL = 0x7bc
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE3_RB_BASE = 0x789
# 	ADDR 0 31
regSDMA1_QUEUE3_RB_BASE_HI = 0x78a
# 	ADDR 0 23
regSDMA1_QUEUE3_RB_CNTL = 0x788
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE3_RB_PREEMPT = 0x7be
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE3_RB_RPTR = 0x78b
# 	OFFSET 0 31
regSDMA1_QUEUE3_RB_RPTR_ADDR_HI = 0x790
# 	ADDR 0 31
regSDMA1_QUEUE3_RB_RPTR_ADDR_LO = 0x791
# 	ADDR 2 31
regSDMA1_QUEUE3_RB_RPTR_HI = 0x78c
# 	OFFSET 0 31
regSDMA1_QUEUE3_RB_WPTR = 0x78d
# 	OFFSET 0 31
regSDMA1_QUEUE3_RB_WPTR_HI = 0x78e
# 	OFFSET 0 31
regSDMA1_QUEUE3_RB_WPTR_POLL_ADDR_HI = 0x7ba
# 	ADDR 0 31
regSDMA1_QUEUE3_RB_WPTR_POLL_ADDR_LO = 0x7bb
# 	ADDR 2 31
regSDMA1_QUEUE3_SCHEDULE_CNTL = 0x7b6
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE3_SKIP_CNTL = 0x798
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE4_CONTEXT_STATUS = 0x7f1
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE4_CSA_ADDR_HI = 0x80d
# 	ADDR 0 31
regSDMA1_QUEUE4_CSA_ADDR_LO = 0x80c
# 	ADDR 2 31
regSDMA1_QUEUE4_DOORBELL = 0x7f2
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE4_DOORBELL_LOG = 0x809
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE4_DOORBELL_OFFSET = 0x80b
# 	OFFSET 2 27
regSDMA1_QUEUE4_DUMMY_REG = 0x811
# 	DUMMY 0 31
regSDMA1_QUEUE4_IB_BASE_HI = 0x7ee
# 	ADDR 0 31
regSDMA1_QUEUE4_IB_BASE_LO = 0x7ed
# 	ADDR 5 31
regSDMA1_QUEUE4_IB_CNTL = 0x7ea
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE4_IB_OFFSET = 0x7ec
# 	OFFSET 2 21
regSDMA1_QUEUE4_IB_RPTR = 0x7eb
# 	OFFSET 2 21
regSDMA1_QUEUE4_IB_SIZE = 0x7ef
# 	SIZE 0 19
regSDMA1_QUEUE4_IB_SUB_REMAIN = 0x80f
# 	SIZE 0 13
regSDMA1_QUEUE4_MIDCMD_CNTL = 0x82b
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE4_MIDCMD_DATA0 = 0x820
# 	DATA0 0 31
regSDMA1_QUEUE4_MIDCMD_DATA1 = 0x821
# 	DATA1 0 31
regSDMA1_QUEUE4_MIDCMD_DATA10 = 0x82a
# 	DATA10 0 31
regSDMA1_QUEUE4_MIDCMD_DATA2 = 0x822
# 	DATA2 0 31
regSDMA1_QUEUE4_MIDCMD_DATA3 = 0x823
# 	DATA3 0 31
regSDMA1_QUEUE4_MIDCMD_DATA4 = 0x824
# 	DATA4 0 31
regSDMA1_QUEUE4_MIDCMD_DATA5 = 0x825
# 	DATA5 0 31
regSDMA1_QUEUE4_MIDCMD_DATA6 = 0x826
# 	DATA6 0 31
regSDMA1_QUEUE4_MIDCMD_DATA7 = 0x827
# 	DATA7 0 31
regSDMA1_QUEUE4_MIDCMD_DATA8 = 0x828
# 	DATA8 0 31
regSDMA1_QUEUE4_MIDCMD_DATA9 = 0x829
# 	DATA9 0 31
regSDMA1_QUEUE4_MINOR_PTR_UPDATE = 0x815
# 	ENABLE 0 0
regSDMA1_QUEUE4_PREEMPT = 0x810
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE4_RB_AQL_CNTL = 0x814
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE4_RB_BASE = 0x7e1
# 	ADDR 0 31
regSDMA1_QUEUE4_RB_BASE_HI = 0x7e2
# 	ADDR 0 23
regSDMA1_QUEUE4_RB_CNTL = 0x7e0
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE4_RB_PREEMPT = 0x816
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE4_RB_RPTR = 0x7e3
# 	OFFSET 0 31
regSDMA1_QUEUE4_RB_RPTR_ADDR_HI = 0x7e8
# 	ADDR 0 31
regSDMA1_QUEUE4_RB_RPTR_ADDR_LO = 0x7e9
# 	ADDR 2 31
regSDMA1_QUEUE4_RB_RPTR_HI = 0x7e4
# 	OFFSET 0 31
regSDMA1_QUEUE4_RB_WPTR = 0x7e5
# 	OFFSET 0 31
regSDMA1_QUEUE4_RB_WPTR_HI = 0x7e6
# 	OFFSET 0 31
regSDMA1_QUEUE4_RB_WPTR_POLL_ADDR_HI = 0x812
# 	ADDR 0 31
regSDMA1_QUEUE4_RB_WPTR_POLL_ADDR_LO = 0x813
# 	ADDR 2 31
regSDMA1_QUEUE4_SCHEDULE_CNTL = 0x80e
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE4_SKIP_CNTL = 0x7f0
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE5_CONTEXT_STATUS = 0x849
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE5_CSA_ADDR_HI = 0x865
# 	ADDR 0 31
regSDMA1_QUEUE5_CSA_ADDR_LO = 0x864
# 	ADDR 2 31
regSDMA1_QUEUE5_DOORBELL = 0x84a
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE5_DOORBELL_LOG = 0x861
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE5_DOORBELL_OFFSET = 0x863
# 	OFFSET 2 27
regSDMA1_QUEUE5_DUMMY_REG = 0x869
# 	DUMMY 0 31
regSDMA1_QUEUE5_IB_BASE_HI = 0x846
# 	ADDR 0 31
regSDMA1_QUEUE5_IB_BASE_LO = 0x845
# 	ADDR 5 31
regSDMA1_QUEUE5_IB_CNTL = 0x842
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE5_IB_OFFSET = 0x844
# 	OFFSET 2 21
regSDMA1_QUEUE5_IB_RPTR = 0x843
# 	OFFSET 2 21
regSDMA1_QUEUE5_IB_SIZE = 0x847
# 	SIZE 0 19
regSDMA1_QUEUE5_IB_SUB_REMAIN = 0x867
# 	SIZE 0 13
regSDMA1_QUEUE5_MIDCMD_CNTL = 0x883
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE5_MIDCMD_DATA0 = 0x878
# 	DATA0 0 31
regSDMA1_QUEUE5_MIDCMD_DATA1 = 0x879
# 	DATA1 0 31
regSDMA1_QUEUE5_MIDCMD_DATA10 = 0x882
# 	DATA10 0 31
regSDMA1_QUEUE5_MIDCMD_DATA2 = 0x87a
# 	DATA2 0 31
regSDMA1_QUEUE5_MIDCMD_DATA3 = 0x87b
# 	DATA3 0 31
regSDMA1_QUEUE5_MIDCMD_DATA4 = 0x87c
# 	DATA4 0 31
regSDMA1_QUEUE5_MIDCMD_DATA5 = 0x87d
# 	DATA5 0 31
regSDMA1_QUEUE5_MIDCMD_DATA6 = 0x87e
# 	DATA6 0 31
regSDMA1_QUEUE5_MIDCMD_DATA7 = 0x87f
# 	DATA7 0 31
regSDMA1_QUEUE5_MIDCMD_DATA8 = 0x880
# 	DATA8 0 31
regSDMA1_QUEUE5_MIDCMD_DATA9 = 0x881
# 	DATA9 0 31
regSDMA1_QUEUE5_MINOR_PTR_UPDATE = 0x86d
# 	ENABLE 0 0
regSDMA1_QUEUE5_PREEMPT = 0x868
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE5_RB_AQL_CNTL = 0x86c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE5_RB_BASE = 0x839
# 	ADDR 0 31
regSDMA1_QUEUE5_RB_BASE_HI = 0x83a
# 	ADDR 0 23
regSDMA1_QUEUE5_RB_CNTL = 0x838
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE5_RB_PREEMPT = 0x86e
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE5_RB_RPTR = 0x83b
# 	OFFSET 0 31
regSDMA1_QUEUE5_RB_RPTR_ADDR_HI = 0x840
# 	ADDR 0 31
regSDMA1_QUEUE5_RB_RPTR_ADDR_LO = 0x841
# 	ADDR 2 31
regSDMA1_QUEUE5_RB_RPTR_HI = 0x83c
# 	OFFSET 0 31
regSDMA1_QUEUE5_RB_WPTR = 0x83d
# 	OFFSET 0 31
regSDMA1_QUEUE5_RB_WPTR_HI = 0x83e
# 	OFFSET 0 31
regSDMA1_QUEUE5_RB_WPTR_POLL_ADDR_HI = 0x86a
# 	ADDR 0 31
regSDMA1_QUEUE5_RB_WPTR_POLL_ADDR_LO = 0x86b
# 	ADDR 2 31
regSDMA1_QUEUE5_SCHEDULE_CNTL = 0x866
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE5_SKIP_CNTL = 0x848
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE6_CONTEXT_STATUS = 0x8a1
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE6_CSA_ADDR_HI = 0x8bd
# 	ADDR 0 31
regSDMA1_QUEUE6_CSA_ADDR_LO = 0x8bc
# 	ADDR 2 31
regSDMA1_QUEUE6_DOORBELL = 0x8a2
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE6_DOORBELL_LOG = 0x8b9
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE6_DOORBELL_OFFSET = 0x8bb
# 	OFFSET 2 27
regSDMA1_QUEUE6_DUMMY_REG = 0x8c1
# 	DUMMY 0 31
regSDMA1_QUEUE6_IB_BASE_HI = 0x89e
# 	ADDR 0 31
regSDMA1_QUEUE6_IB_BASE_LO = 0x89d
# 	ADDR 5 31
regSDMA1_QUEUE6_IB_CNTL = 0x89a
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE6_IB_OFFSET = 0x89c
# 	OFFSET 2 21
regSDMA1_QUEUE6_IB_RPTR = 0x89b
# 	OFFSET 2 21
regSDMA1_QUEUE6_IB_SIZE = 0x89f
# 	SIZE 0 19
regSDMA1_QUEUE6_IB_SUB_REMAIN = 0x8bf
# 	SIZE 0 13
regSDMA1_QUEUE6_MIDCMD_CNTL = 0x8db
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE6_MIDCMD_DATA0 = 0x8d0
# 	DATA0 0 31
regSDMA1_QUEUE6_MIDCMD_DATA1 = 0x8d1
# 	DATA1 0 31
regSDMA1_QUEUE6_MIDCMD_DATA10 = 0x8da
# 	DATA10 0 31
regSDMA1_QUEUE6_MIDCMD_DATA2 = 0x8d2
# 	DATA2 0 31
regSDMA1_QUEUE6_MIDCMD_DATA3 = 0x8d3
# 	DATA3 0 31
regSDMA1_QUEUE6_MIDCMD_DATA4 = 0x8d4
# 	DATA4 0 31
regSDMA1_QUEUE6_MIDCMD_DATA5 = 0x8d5
# 	DATA5 0 31
regSDMA1_QUEUE6_MIDCMD_DATA6 = 0x8d6
# 	DATA6 0 31
regSDMA1_QUEUE6_MIDCMD_DATA7 = 0x8d7
# 	DATA7 0 31
regSDMA1_QUEUE6_MIDCMD_DATA8 = 0x8d8
# 	DATA8 0 31
regSDMA1_QUEUE6_MIDCMD_DATA9 = 0x8d9
# 	DATA9 0 31
regSDMA1_QUEUE6_MINOR_PTR_UPDATE = 0x8c5
# 	ENABLE 0 0
regSDMA1_QUEUE6_PREEMPT = 0x8c0
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE6_RB_AQL_CNTL = 0x8c4
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE6_RB_BASE = 0x891
# 	ADDR 0 31
regSDMA1_QUEUE6_RB_BASE_HI = 0x892
# 	ADDR 0 23
regSDMA1_QUEUE6_RB_CNTL = 0x890
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE6_RB_PREEMPT = 0x8c6
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE6_RB_RPTR = 0x893
# 	OFFSET 0 31
regSDMA1_QUEUE6_RB_RPTR_ADDR_HI = 0x898
# 	ADDR 0 31
regSDMA1_QUEUE6_RB_RPTR_ADDR_LO = 0x899
# 	ADDR 2 31
regSDMA1_QUEUE6_RB_RPTR_HI = 0x894
# 	OFFSET 0 31
regSDMA1_QUEUE6_RB_WPTR = 0x895
# 	OFFSET 0 31
regSDMA1_QUEUE6_RB_WPTR_HI = 0x896
# 	OFFSET 0 31
regSDMA1_QUEUE6_RB_WPTR_POLL_ADDR_HI = 0x8c2
# 	ADDR 0 31
regSDMA1_QUEUE6_RB_WPTR_POLL_ADDR_LO = 0x8c3
# 	ADDR 2 31
regSDMA1_QUEUE6_SCHEDULE_CNTL = 0x8be
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE6_SKIP_CNTL = 0x8a0
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE7_CONTEXT_STATUS = 0x8f9
# 	SELECTED 0 0
# 	IDLE 2 2
# 	EXPIRED 3 3
# 	EXCEPTION 4 6
# 	CTXSW_ABLE 7 7
# 	PREEMPT_DISABLE 10 10
# 	RPTR_WB_IDLE 11 11
# 	WPTR_UPDATE_PENDING 12 12
# 	WPTR_UPDATE_FAIL_COUNT 16 23
regSDMA1_QUEUE7_CSA_ADDR_HI = 0x915
# 	ADDR 0 31
regSDMA1_QUEUE7_CSA_ADDR_LO = 0x914
# 	ADDR 2 31
regSDMA1_QUEUE7_DOORBELL = 0x8fa
# 	ENABLE 28 28
# 	CAPTURED 30 30
regSDMA1_QUEUE7_DOORBELL_LOG = 0x911
# 	BE_ERROR 0 0
# 	DATA 2 31
regSDMA1_QUEUE7_DOORBELL_OFFSET = 0x913
# 	OFFSET 2 27
regSDMA1_QUEUE7_DUMMY_REG = 0x919
# 	DUMMY 0 31
regSDMA1_QUEUE7_IB_BASE_HI = 0x8f6
# 	ADDR 0 31
regSDMA1_QUEUE7_IB_BASE_LO = 0x8f5
# 	ADDR 5 31
regSDMA1_QUEUE7_IB_CNTL = 0x8f2
# 	IB_ENABLE 0 0
# 	IB_SWAP_ENABLE 4 4
# 	SWITCH_INSIDE_IB 8 8
# 	CMD_VMID 16 19
regSDMA1_QUEUE7_IB_OFFSET = 0x8f4
# 	OFFSET 2 21
regSDMA1_QUEUE7_IB_RPTR = 0x8f3
# 	OFFSET 2 21
regSDMA1_QUEUE7_IB_SIZE = 0x8f7
# 	SIZE 0 19
regSDMA1_QUEUE7_IB_SUB_REMAIN = 0x917
# 	SIZE 0 13
regSDMA1_QUEUE7_MIDCMD_CNTL = 0x933
# 	DATA_VALID 0 0
# 	COPY_MODE 1 1
# 	SPLIT_STATE 4 7
# 	ALLOW_PREEMPT 8 8
regSDMA1_QUEUE7_MIDCMD_DATA0 = 0x928
# 	DATA0 0 31
regSDMA1_QUEUE7_MIDCMD_DATA1 = 0x929
# 	DATA1 0 31
regSDMA1_QUEUE7_MIDCMD_DATA10 = 0x932
# 	DATA10 0 31
regSDMA1_QUEUE7_MIDCMD_DATA2 = 0x92a
# 	DATA2 0 31
regSDMA1_QUEUE7_MIDCMD_DATA3 = 0x92b
# 	DATA3 0 31
regSDMA1_QUEUE7_MIDCMD_DATA4 = 0x92c
# 	DATA4 0 31
regSDMA1_QUEUE7_MIDCMD_DATA5 = 0x92d
# 	DATA5 0 31
regSDMA1_QUEUE7_MIDCMD_DATA6 = 0x92e
# 	DATA6 0 31
regSDMA1_QUEUE7_MIDCMD_DATA7 = 0x92f
# 	DATA7 0 31
regSDMA1_QUEUE7_MIDCMD_DATA8 = 0x930
# 	DATA8 0 31
regSDMA1_QUEUE7_MIDCMD_DATA9 = 0x931
# 	DATA9 0 31
regSDMA1_QUEUE7_MINOR_PTR_UPDATE = 0x91d
# 	ENABLE 0 0
regSDMA1_QUEUE7_PREEMPT = 0x918
# 	IB_PREEMPT 0 0
regSDMA1_QUEUE7_RB_AQL_CNTL = 0x91c
# 	AQL_ENABLE 0 0
# 	AQL_PACKET_SIZE 1 7
# 	PACKET_STEP 8 15
# 	MIDCMD_PREEMPT_ENABLE 16 16
# 	MIDCMD_PREEMPT_DATA_RESTORE 17 17
# 	OVERLAP_ENABLE 18 18
regSDMA1_QUEUE7_RB_BASE = 0x8e9
# 	ADDR 0 31
regSDMA1_QUEUE7_RB_BASE_HI = 0x8ea
# 	ADDR 0 23
regSDMA1_QUEUE7_RB_CNTL = 0x8e8
# 	RB_ENABLE 0 0
# 	RB_SIZE 1 5
# 	WPTR_POLL_ENABLE 8 8
# 	RB_SWAP_ENABLE 9 9
# 	WPTR_POLL_SWAP_ENABLE 10 10
# 	F32_WPTR_POLL_ENABLE 11 11
# 	RPTR_WRITEBACK_ENABLE 12 12
# 	RPTR_WRITEBACK_SWAP_ENABLE 13 13
# 	RPTR_WRITEBACK_TIMER 16 20
# 	RB_PRIV 23 23
# 	RB_VMID 24 27
regSDMA1_QUEUE7_RB_PREEMPT = 0x91e
# 	PREEMPT_REQ 0 0
regSDMA1_QUEUE7_RB_RPTR = 0x8eb
# 	OFFSET 0 31
regSDMA1_QUEUE7_RB_RPTR_ADDR_HI = 0x8f0
# 	ADDR 0 31
regSDMA1_QUEUE7_RB_RPTR_ADDR_LO = 0x8f1
# 	ADDR 2 31
regSDMA1_QUEUE7_RB_RPTR_HI = 0x8ec
# 	OFFSET 0 31
regSDMA1_QUEUE7_RB_WPTR = 0x8ed
# 	OFFSET 0 31
regSDMA1_QUEUE7_RB_WPTR_HI = 0x8ee
# 	OFFSET 0 31
regSDMA1_QUEUE7_RB_WPTR_POLL_ADDR_HI = 0x91a
# 	ADDR 0 31
regSDMA1_QUEUE7_RB_WPTR_POLL_ADDR_LO = 0x91b
# 	ADDR 2 31
regSDMA1_QUEUE7_SCHEDULE_CNTL = 0x916
# 	GLOBAL_ID 0 1
# 	PROCESS_ID 2 4
# 	LOCAL_ID 6 7
# 	CONTEXT_QUANTUM 8 15
regSDMA1_QUEUE7_SKIP_CNTL = 0x8f8
# 	SKIP_COUNT 0 19
regSDMA1_QUEUE_RESET_REQ = 0x67b
# 	QUEUE0_RESET 0 0
# 	QUEUE1_RESET 1 1
# 	QUEUE2_RESET 2 2
# 	QUEUE3_RESET 3 3
# 	QUEUE4_RESET 4 4
# 	QUEUE5_RESET 5 5
# 	QUEUE6_RESET 6 6
# 	QUEUE7_RESET 7 7
# 	RESERVED 8 31
regSDMA1_QUEUE_STATUS0 = 0x62f
# 	QUEUE0_STATUS 0 3
# 	QUEUE1_STATUS 4 7
# 	QUEUE2_STATUS 8 11
# 	QUEUE3_STATUS 12 15
# 	QUEUE4_STATUS 16 19
# 	QUEUE5_STATUS 20 23
# 	QUEUE6_STATUS 24 27
# 	QUEUE7_STATUS 28 31
regSDMA1_RB_RPTR_FETCH = 0x620
# 	OFFSET 2 31
regSDMA1_RB_RPTR_FETCH_HI = 0x621
# 	OFFSET 0 31
regSDMA1_RELAX_ORDERING_LUT = 0x64a
# 	RESERVED0 0 0
# 	COPY 1 1
# 	WRITE 2 2
# 	RESERVED3 3 3
# 	RESERVED4 4 4
# 	FENCE 5 5
# 	RESERVED76 6 7
# 	POLL_MEM 8 8
# 	COND_EXE 9 9
# 	ATOMIC 10 10
# 	CONST_FILL 11 11
# 	PTEPDE 12 12
# 	TIMESTAMP 13 13
# 	RESERVED 14 26
# 	WORLD_SWITCH 27 27
# 	RPTR_WRB 28 28
# 	WPTR_POLL 29 29
# 	IB_FETCH 30 30
# 	RB_FETCH 31 31
regSDMA1_RLC_CGCG_CTRL = 0x65c
# 	CGCG_INT_ENABLE 1 1
# 	CGCG_IDLE_HYSTERESIS 16 31
regSDMA1_SCRATCH_RAM_ADDR = 0x678
# 	ADDR 0 6
regSDMA1_SCRATCH_RAM_DATA = 0x677
# 	DATA 0 31
regSDMA1_SEM_WAIT_FAIL_TIMER_CNTL = 0x622
# 	TIMER 0 31
regSDMA1_STATUS1_REG = 0x626
# 	CE_WREQ_IDLE 0 0
# 	CE_WR_IDLE 1 1
# 	CE_SPLIT_IDLE 2 2
# 	CE_RREQ_IDLE 3 3
# 	CE_OUT_IDLE 4 4
# 	CE_IN_IDLE 5 5
# 	CE_DST_IDLE 6 6
# 	CE_CMD_IDLE 9 9
# 	CE_AFIFO_FULL 10 10
# 	CE_INFO_FULL 11 11
# 	CE_INFO1_FULL 12 12
# 	EX_START 13 13
# 	CE_RD_STALL 15 15
# 	CE_WR_STALL 16 16
# 	SEC_INTR_STATUS 17 17
# 	WPTR_POLL_IDLE 18 18
# 	SDMA_IDLE 19 19
regSDMA1_STATUS2_REG = 0x638
# 	ID 0 1
# 	TH0F32_INSTR_PTR 2 15
# 	CMD_OP 16 31
regSDMA1_STATUS3_REG = 0x64c
# 	CMD_OP_STATUS 0 15
# 	PREV_VM_CMD 16 19
# 	EXCEPTION_IDLE 20 20
# 	AQL_PREV_CMD_IDLE 21 21
# 	TLBI_IDLE 22 22
# 	GCR_IDLE 23 23
# 	INVREQ_IDLE 24 24
# 	QUEUE_ID_MATCH 25 25
# 	INT_QUEUE_ID 26 29
# 	TMZ_MTYPE_STATUS 30 31
regSDMA1_STATUS4_REG = 0x676
# 	IDLE 0 0
# 	IH_OUTSTANDING 2 2
# 	SEM_OUTSTANDING 3 3
# 	CH_RD_OUTSTANDING 4 4
# 	CH_WR_OUTSTANDING 5 5
# 	GCR_OUTSTANDING 6 6
# 	TLBI_OUTSTANDING 7 7
# 	UTCL2_RD_OUTSTANDING 8 8
# 	UTCL2_WR_OUTSTANDING 9 9
# 	REG_POLLING 10 10
# 	MEM_POLLING 11 11
# 	RESERVED_13_12 12 13
# 	RESERVED_15_14 14 15
# 	ACTIVE_QUEUE_ID 16 19
# 	SRIOV_WATING_RLCV_CMD 20 20
# 	SRIOV_SDMA_EXECUTING_CMD 21 21
# 	UTCL2_RD_XNACK_FAULT 22 22
# 	UTCL2_RD_XNACK_NULL 23 23
# 	UTCL2_RD_XNACK_TIMEOUT 24 24
# 	UTCL2_WR_XNACK_FAULT 25 25
# 	UTCL2_WR_XNACK_NULL 26 26
# 	UTCL2_WR_XNACK_TIMEOUT 27 27
regSDMA1_STATUS5_REG = 0x67a
# 	QUEUE0_RB_ENABLE_STATUS 0 0
# 	QUEUE1_RB_ENABLE_STATUS 1 1
# 	QUEUE2_RB_ENABLE_STATUS 2 2
# 	QUEUE3_RB_ENABLE_STATUS 3 3
# 	QUEUE4_RB_ENABLE_STATUS 4 4
# 	QUEUE5_RB_ENABLE_STATUS 5 5
# 	QUEUE6_RB_ENABLE_STATUS 6 6
# 	QUEUE7_RB_ENABLE_STATUS 7 7
# 	ACTIVE_QUEUE_ID 16 19
# 	QUEUE0_WPTR_POLL_PAGE_EXCEPTION 20 20
# 	QUEUE1_WPTR_POLL_PAGE_EXCEPTION 21 21
# 	QUEUE2_WPTR_POLL_PAGE_EXCEPTION 22 22
# 	QUEUE3_WPTR_POLL_PAGE_EXCEPTION 23 23
# 	QUEUE4_WPTR_POLL_PAGE_EXCEPTION 24 24
# 	QUEUE5_WPTR_POLL_PAGE_EXCEPTION 25 25
# 	QUEUE6_WPTR_POLL_PAGE_EXCEPTION 26 26
# 	QUEUE7_WPTR_POLL_PAGE_EXCEPTION 27 27
regSDMA1_STATUS6_REG = 0x67c
# 	ID 0 1
# 	TH1F32_INSTR_PTR 2 15
# 	TH1_EXCEPTION 16 31
regSDMA1_STATUS_REG = 0x625
# 	IDLE 0 0
# 	REG_IDLE 1 1
# 	RB_EMPTY 2 2
# 	RB_FULL 3 3
# 	RB_CMD_IDLE 4 4
# 	RB_CMD_FULL 5 5
# 	IB_CMD_IDLE 6 6
# 	IB_CMD_FULL 7 7
# 	BLOCK_IDLE 8 8
# 	INSIDE_IB 9 9
# 	EX_IDLE 10 10
# 	CGCG_FENCE 11 11
# 	PACKET_READY 12 12
# 	MC_WR_IDLE 13 13
# 	SRBM_IDLE 14 14
# 	CONTEXT_EMPTY 15 15
# 	DELTA_RPTR_FULL 16 16
# 	RB_MC_RREQ_IDLE 17 17
# 	IB_MC_RREQ_IDLE 18 18
# 	MC_RD_IDLE 19 19
# 	DELTA_RPTR_EMPTY 20 20
# 	MC_RD_RET_STALL 21 21
# 	MC_RD_NO_POLL_IDLE 22 22
# 	PREV_CMD_IDLE 25 25
# 	SEM_IDLE 26 26
# 	SEM_REQ_STALL 27 27
# 	SEM_RESP_STATE 28 29
# 	INT_IDLE 30 30
# 	INT_REQ_STALL 31 31
regSDMA1_TILING_CONFIG = 0x663
# 	PIPE_INTERLEAVE_SIZE 4 6
regSDMA1_TIMESTAMP_CNTL = 0x679
# 	CAPTURE 0 0
regSDMA1_TLBI_GCR_CNTL = 0x662
# 	TLBI_CMD_DW 0 3
# 	GCR_CMD_DW 4 7
# 	GCR_CLKEN_CYCLE 8 11
# 	TLBI_CREDIT 16 23
# 	GCR_CREDIT 24 31
regSDMA1_UCODE1_CHECKSUM = 0x67d
# 	DATA 0 31
regSDMA1_UCODE_ADDR = 0x58a0
# 	VALUE 0 12
# 	THID 15 15
regSDMA1_UCODE_CHECKSUM = 0x629
# 	DATA 0 31
regSDMA1_UCODE_DATA = 0x58a1
# 	VALUE 0 31
regSDMA1_UCODE_SELFLOAD_CONTROL = 0x58a2
regSDMA1_UTCL1_CNTL = 0x63c
# 	REDO_DELAY 0 4
# 	PAGE_WAIT_DELAY 5 8
# 	RESP_MODE 9 10
# 	FORCE_INVALIDATION 14 14
# 	FORCE_INVREQ_HEAVY 15 15
# 	WR_EXE_PERMS_CTRL 16 16
# 	RD_EXE_PERMS_CTRL 17 17
# 	INVACK_DELAY 18 21
# 	REQL2_CREDIT 24 29
regSDMA1_UTCL1_INV0 = 0x642
# 	INV_PROC_BUSY 0 0
# 	GPUVM_FRAG_SIZE 1 6
# 	GPUVM_VMID 7 10
# 	GPUVM_MODE 11 12
# 	GPUVM_HIGH 13 13
# 	GPUVM_TAG 14 17
# 	GPUVM_VMID_HIGH 18 21
# 	GPUVM_VMID_LOW 22 25
# 	INV_TYPE 26 27
regSDMA1_UTCL1_INV1 = 0x643
# 	INV_ADDR_LO 0 31
regSDMA1_UTCL1_INV2 = 0x644
# 	CPF_VMID 0 15
# 	CPF_FLUSH_TYPE 16 16
# 	CPF_FRAG_SIZE 17 22
regSDMA1_UTCL1_PAGE = 0x63f
# 	VM_HOLE 0 0
# 	REQ_TYPE 1 4
# 	USE_MTYPE 6 9
# 	USE_PT_SNOOP 10 10
# 	USE_IO 11 11
# 	RD_L2_POLICY 12 13
# 	WR_L2_POLICY 14 15
# 	DMA_PAGE_SIZE 16 21
# 	USE_BC 22 22
# 	ADDR_IS_PA 23 23
# 	LLC_NOALLOC 24 24
regSDMA1_UTCL1_RD_STATUS = 0x640
# 	RD_VA_FIFO_EMPTY 0 0
# 	RD_REG_ENTRY_EMPTY 1 1
# 	RD_PAGE_FIFO_EMPTY 2 2
# 	RD_REQ_FIFO_EMPTY 3 3
# 	RD_VA_REQ_FIFO_EMPTY 4 4
# 	RESERVED0 5 5
# 	RESERVED1 6 6
# 	META_Q_EMPTY 7 7
# 	RD_VA_FIFO_FULL 8 8
# 	RD_REG_ENTRY_FULL 9 9
# 	RD_PAGE_FIFO_FULL 10 10
# 	RD_REQ_FIFO_FULL 11 11
# 	RD_VA_REQ_FIFO_FULL 12 12
# 	RESERVED2 13 13
# 	RESERVED3 14 14
# 	META_Q_FULL 15 15
# 	RD_L2_INTF_IDLE 16 16
# 	RD_REQRET_IDLE 17 17
# 	RD_REQ_IDLE 18 18
# 	RD_MERGE_TYPE 19 20
# 	RD_MERGE_DATA_PA_READY 21 21
# 	RD_MERGE_META_PA_READY 22 22
# 	RD_MERGE_REG_READY 23 23
# 	RD_MERGE_PAGE_FIFO_READY 24 24
# 	RD_MERGE_REQ_FIFO_READY 25 25
# 	RESERVED4 26 26
# 	RD_MERGE_OUT_RTR 27 27
# 	RDREQ_IN_RTR 28 28
# 	RDREQ_OUT_RTR 29 29
# 	INV_BUSY 30 30
# 	DBIT_REQ_IDLE 31 31
regSDMA1_UTCL1_RD_XNACK0 = 0x645
# 	XNACK_FAULT_ADDR_LO 0 31
regSDMA1_UTCL1_RD_XNACK1 = 0x646
# 	XNACK_FAULT_ADDR_HI 0 3
# 	XNACK_FAULT_VMID 4 7
# 	XNACK_FAULT_VECTOR 8 9
# 	XNACK_NULL_VECTOR 10 11
# 	XNACK_TIMEOUT_VECTOR 12 13
# 	XNACK_FAULT_FLAG 14 14
# 	XNACK_NULL_FLAG 15 15
# 	XNACK_TIMEOUT_FLAG 16 16
regSDMA1_UTCL1_TIMEOUT = 0x63e
# 	XNACK_LIMIT 0 15
regSDMA1_UTCL1_WATERMK = 0x63d
# 	WR_REQ_FIFO_WATERMK 0 3
# 	WR_REQ_FIFO_DEPTH_STEP 4 5
# 	RD_REQ_FIFO_WATERMK 6 9
# 	RD_REQ_FIFO_DEPTH_STEP 10 11
# 	WR_PAGE_FIFO_WATERMK 12 15
# 	WR_PAGE_FIFO_DEPTH_STEP 16 17
# 	RD_PAGE_FIFO_WATERMK 18 21
# 	RD_PAGE_FIFO_DEPTH_STEP 22 23
regSDMA1_UTCL1_WR_STATUS = 0x641
# 	WR_VA_FIFO_EMPTY 0 0
# 	WR_REG_ENTRY_EMPTY 1 1
# 	WR_PAGE_FIFO_EMPTY 2 2
# 	WR_REQ_FIFO_EMPTY 3 3
# 	WR_VA_REQ_FIFO_EMPTY 4 4
# 	WR_DATA2_EMPTY 5 5
# 	WR_DATA1_EMPTY 6 6
# 	RESERVED0 7 7
# 	WR_VA_FIFO_FULL 8 8
# 	WR_REG_ENTRY_FULL 9 9
# 	WR_PAGE_FIFO_FULL 10 10
# 	WR_REQ_FIFO_FULL 11 11
# 	WR_VA_REQ_FIFO_FULL 12 12
# 	WR_DATA2_FULL 13 13
# 	WR_DATA1_FULL 14 14
# 	F32_WR_RTR 15 15
# 	WR_L2_INTF_IDLE 16 16
# 	WR_REQRET_IDLE 17 17
# 	WR_REQ_IDLE 18 18
# 	WR_MERGE_TYPE 19 20
# 	WR_MERGE_DATA_PA_READY 21 21
# 	WR_MERGE_META_PA_READY 22 22
# 	WR_MERGE_REG_READY 23 23
# 	WR_MERGE_PAGE_FIFO_READY 24 24
# 	WR_MERGE_REQ_FIFO_READY 25 25
# 	WR_MERGE_DATA_SEL 26 26
# 	WR_MERGE_OUT_RTR 27 27
# 	WRREQ_IN_RTR 28 28
# 	WRREQ_OUT_RTR 29 29
# 	WRREQ_IN_DATA1_RTR 30 30
# 	WRREQ_IN_DATA2_RTR 31 31
regSDMA1_UTCL1_WR_XNACK0 = 0x647
# 	XNACK_FAULT_ADDR_LO 0 31
regSDMA1_UTCL1_WR_XNACK1 = 0x648
# 	XNACK_FAULT_ADDR_HI 0 3
# 	XNACK_FAULT_VMID 4 7
# 	XNACK_FAULT_VECTOR 8 9
# 	XNACK_NULL_VECTOR 10 11
# 	XNACK_TIMEOUT_VECTOR 12 13
# 	XNACK_FAULT_FLAG 14 14
# 	XNACK_NULL_FLAG 15 15
# 	XNACK_TIMEOUT_FLAG 16 16
regSDMA1_VERSION = 0x635
# 	MINVER 0 6
# 	MAJVER 8 14
# 	REV 16 21
regSDMA1_WATCHDOG_CNTL = 0x62e
# 	QUEUE_HANG_COUNT 0 7
# 	CMD_TIMEOUT_COUNT 8 15
regSE0_CAC_AGGR_GFXCLK_CYCLE = 0x1ae5
# 	SE0_AGGR_GFXCLK_CYCLE 0 31
regSE0_CAC_AGGR_LOWER = 0x1ad4
# 	SE0_AGGR_31_0 0 31
regSE0_CAC_AGGR_UPPER = 0x1ad5
# 	SE0_AGGR_63_32 0 31
regSE1_CAC_AGGR_GFXCLK_CYCLE = 0x1ae6
# 	SE1_AGGR_GFXCLK_CYCLE 0 31
regSE1_CAC_AGGR_LOWER = 0x1ad6
# 	SE1_AGGR_31_0 0 31
regSE1_CAC_AGGR_UPPER = 0x1ad7
# 	SE1_AGGR_63_32 0 31
regSE2_CAC_AGGR_GFXCLK_CYCLE = 0x1ae7
# 	SE2_AGGR_GFXCLK_CYCLE 0 31
regSE2_CAC_AGGR_LOWER = 0x1ad8
# 	SE2_AGGR_31_0 0 31
regSE2_CAC_AGGR_UPPER = 0x1ad9
# 	SE2_AGGR_63_32 0 31
regSE3_CAC_AGGR_GFXCLK_CYCLE = 0x1ae8
# 	SE3_AGGR_GFXCLK_CYCLE 0 31
regSE3_CAC_AGGR_LOWER = 0x1ada
# 	SE3_AGGR_31_0 0 31
regSE3_CAC_AGGR_UPPER = 0x1adb
# 	SE3_AGGR_63_32 0 31
regSE4_CAC_AGGR_GFXCLK_CYCLE = 0x1ae9
# 	SE4_AGGR_GFXCLK_CYCLE 0 31
regSE4_CAC_AGGR_LOWER = 0x1adc
# 	SE4_AGGR_31_0 0 31
regSE4_CAC_AGGR_UPPER = 0x1add
# 	SE4_AGGR_63_32 0 31
regSE5_CAC_AGGR_GFXCLK_CYCLE = 0x1aea
# 	SE5_AGGR_GFXCLK_CYCLE 0 31
regSE5_CAC_AGGR_LOWER = 0x1ade
# 	SE5_AGGR_31_0 0 31
regSE5_CAC_AGGR_UPPER = 0x1adf
# 	SE5_AGGR_63_32 0 31
regSEDC_GL1_GL2_OVERRIDES = 0x1ac0
# 	SEDC_GL1C_GL2R_REQ_CREDITS 0 5
# 	SEDC_GL1C_GL2R_DATA_CREDITS 8 13
# 	SEDC_GL1C_GL2R_OUT_CLK_OVERRIDE 16 16
regSE_CAC_CTRL_1 = 0x1b70
# 	CAC_WINDOW 0 7
# 	TDP_WINDOW 8 31
regSE_CAC_CTRL_2 = 0x1b71
# 	CAC_ENABLE 0 0
# 	SE_LCAC_ENABLE 1 1
# 	WGP_CAC_CLK_OVERRIDE 2 2
# 	SE_CAC_INDEX_AUTO_INCR_EN 3 3
regSE_CAC_IND_DATA = 0x1bcf
# 	SE_CAC_IND_DATA 0 31
regSE_CAC_IND_INDEX = 0x1bce
# 	SE_CAC_IND_ADDR 0 31
regSE_CAC_WEIGHT_BCI_0 = 0x1b8a
# 	WEIGHT_BCI_SIG0 0 15
# 	WEIGHT_BCI_SIG1 16 31
regSE_CAC_WEIGHT_CB_0 = 0x1b8b
# 	WEIGHT_CB_SIG0 0 15
# 	WEIGHT_CB_SIG1 16 31
regSE_CAC_WEIGHT_CB_1 = 0x1b8c
# 	WEIGHT_CB_SIG2 0 15
# 	WEIGHT_CB_SIG3 16 31
regSE_CAC_WEIGHT_CB_10 = 0x1b95
# 	WEIGHT_CB_SIG20 0 15
# 	WEIGHT_CB_SIG21 16 31
regSE_CAC_WEIGHT_CB_11 = 0x1b96
# 	WEIGHT_CB_SIG22 0 15
# 	WEIGHT_CB_SIG23 16 31
regSE_CAC_WEIGHT_CB_2 = 0x1b8d
# 	WEIGHT_CB_SIG4 0 15
# 	WEIGHT_CB_SIG5 16 31
regSE_CAC_WEIGHT_CB_3 = 0x1b8e
# 	WEIGHT_CB_SIG6 0 15
# 	WEIGHT_CB_SIG7 16 31
regSE_CAC_WEIGHT_CB_4 = 0x1b8f
# 	WEIGHT_CB_SIG8 0 15
# 	WEIGHT_CB_SIG9 16 31
regSE_CAC_WEIGHT_CB_5 = 0x1b90
# 	WEIGHT_CB_SIG10 0 15
# 	WEIGHT_CB_SIG11 16 31
regSE_CAC_WEIGHT_CB_6 = 0x1b91
# 	WEIGHT_CB_SIG12 0 15
# 	WEIGHT_CB_SIG13 16 31
regSE_CAC_WEIGHT_CB_7 = 0x1b92
# 	WEIGHT_CB_SIG14 0 15
# 	WEIGHT_CB_SIG15 16 31
regSE_CAC_WEIGHT_CB_8 = 0x1b93
# 	WEIGHT_CB_SIG16 0 15
# 	WEIGHT_CB_SIG17 16 31
regSE_CAC_WEIGHT_CB_9 = 0x1b94
# 	WEIGHT_CB_SIG18 0 15
# 	WEIGHT_CB_SIG19 16 31
regSE_CAC_WEIGHT_CU_0 = 0x1b89
# 	WEIGHT_CU_SIG0 0 15
regSE_CAC_WEIGHT_DB_0 = 0x1b97
# 	WEIGHT_DB_SIG0 0 15
# 	WEIGHT_DB_SIG1 16 31
regSE_CAC_WEIGHT_DB_1 = 0x1b98
# 	WEIGHT_DB_SIG2 0 15
# 	WEIGHT_DB_SIG3 16 31
regSE_CAC_WEIGHT_DB_2 = 0x1b99
# 	WEIGHT_DB_SIG4 0 15
# 	WEIGHT_DB_SIG5 16 31
regSE_CAC_WEIGHT_DB_3 = 0x1b9a
# 	WEIGHT_DB_SIG6 0 15
# 	WEIGHT_DB_SIG7 16 31
regSE_CAC_WEIGHT_DB_4 = 0x1b9b
# 	WEIGHT_DB_SIG8 0 15
# 	WEIGHT_DB_SIG9 16 31
regSE_CAC_WEIGHT_GL1C_0 = 0x1ba1
# 	WEIGHT_GL1C_SIG0 0 15
# 	WEIGHT_GL1C_SIG1 16 31
regSE_CAC_WEIGHT_GL1C_1 = 0x1ba2
# 	WEIGHT_GL1C_SIG2 0 15
# 	WEIGHT_GL1C_SIG3 16 31
regSE_CAC_WEIGHT_GL1C_2 = 0x1ba3
# 	WEIGHT_GL1C_SIG4 0 15
regSE_CAC_WEIGHT_LDS_0 = 0x1b82
# 	WEIGHT_LDS_SIG0 0 15
# 	WEIGHT_LDS_SIG1 16 31
regSE_CAC_WEIGHT_LDS_1 = 0x1b83
# 	WEIGHT_LDS_SIG2 0 15
# 	WEIGHT_LDS_SIG3 16 31
regSE_CAC_WEIGHT_LDS_2 = 0x1b84
# 	WEIGHT_LDS_SIG4 0 15
# 	WEIGHT_LDS_SIG5 16 31
regSE_CAC_WEIGHT_LDS_3 = 0x1b85
# 	WEIGHT_LDS_SIG6 0 15
# 	WEIGHT_LDS_SIG7 16 31
regSE_CAC_WEIGHT_PA_0 = 0x1ba8
# 	WEIGHT_PA_SIG0 0 15
# 	WEIGHT_PA_SIG1 16 31
regSE_CAC_WEIGHT_PA_1 = 0x1ba9
# 	WEIGHT_PA_SIG2 0 15
# 	WEIGHT_PA_SIG3 16 31
regSE_CAC_WEIGHT_PA_2 = 0x1baa
# 	WEIGHT_PA_SIG4 0 15
# 	WEIGHT_PA_SIG5 16 31
regSE_CAC_WEIGHT_PA_3 = 0x1bab
# 	WEIGHT_PA_SIG6 0 15
# 	WEIGHT_PA_SIG7 16 31
regSE_CAC_WEIGHT_PC_0 = 0x1ba7
# 	WEIGHT_PC_SIG0 0 15
regSE_CAC_WEIGHT_RMI_0 = 0x1b9c
# 	WEIGHT_RMI_SIG0 0 15
# 	WEIGHT_RMI_SIG1 16 31
regSE_CAC_WEIGHT_RMI_1 = 0x1b9d
# 	WEIGHT_RMI_SIG2 0 15
# 	WEIGHT_RMI_SIG3 16 31
regSE_CAC_WEIGHT_SC_0 = 0x1bac
# 	WEIGHT_SC_SIG0 0 15
# 	WEIGHT_SC_SIG1 16 31
regSE_CAC_WEIGHT_SC_1 = 0x1bad
# 	WEIGHT_SC_SIG2 0 15
# 	WEIGHT_SC_SIG3 16 31
regSE_CAC_WEIGHT_SC_2 = 0x1bae
# 	WEIGHT_SC_SIG4 0 15
# 	WEIGHT_SC_SIG5 16 31
regSE_CAC_WEIGHT_SC_3 = 0x1baf
# 	WEIGHT_SC_SIG6 0 15
# 	WEIGHT_SC_SIG7 16 31
regSE_CAC_WEIGHT_SPI_0 = 0x1ba4
# 	WEIGHT_SPI_SIG0 0 15
# 	WEIGHT_SPI_SIG1 16 31
regSE_CAC_WEIGHT_SPI_1 = 0x1ba5
# 	WEIGHT_SPI_SIG2 0 15
# 	WEIGHT_SPI_SIG3 16 31
regSE_CAC_WEIGHT_SPI_2 = 0x1ba6
# 	WEIGHT_SPI_SIG4 0 15
regSE_CAC_WEIGHT_SP_0 = 0x1b80
# 	WEIGHT_SP_SIG0 0 15
# 	WEIGHT_SP_SIG1 16 31
regSE_CAC_WEIGHT_SP_1 = 0x1b81
# 	WEIGHT_SP_SIG2 0 15
regSE_CAC_WEIGHT_SQC_0 = 0x1b87
# 	WEIGHT_SQC_SIG0 0 15
# 	WEIGHT_SQC_SIG1 16 31
regSE_CAC_WEIGHT_SQC_1 = 0x1b88
# 	WEIGHT_SQC_SIG2 0 15
regSE_CAC_WEIGHT_SQ_0 = 0x1b7d
# 	WEIGHT_SQ_SIG0 0 15
# 	WEIGHT_SQ_SIG1 16 31
regSE_CAC_WEIGHT_SQ_1 = 0x1b7e
# 	WEIGHT_SQ_SIG2 0 15
# 	WEIGHT_SQ_SIG3 16 31
regSE_CAC_WEIGHT_SQ_2 = 0x1b7f
# 	WEIGHT_SQ_SIG4 0 15
regSE_CAC_WEIGHT_SXRB_0 = 0x1b9f
# 	WEIGHT_SXRB_SIG0 0 15
regSE_CAC_WEIGHT_SX_0 = 0x1b9e
# 	WEIGHT_SX_SIG0 0 15
regSE_CAC_WEIGHT_TA_0 = 0x1b72
# 	WEIGHT_TA_SIG0 0 15
regSE_CAC_WEIGHT_TCP_0 = 0x1b79
# 	WEIGHT_TCP_SIG0 0 15
# 	WEIGHT_TCP_SIG1 16 31
regSE_CAC_WEIGHT_TCP_1 = 0x1b7a
# 	WEIGHT_TCP_SIG2 0 15
# 	WEIGHT_TCP_SIG3 16 31
regSE_CAC_WEIGHT_TCP_2 = 0x1b7b
# 	WEIGHT_TCP_SIG4 0 15
# 	WEIGHT_TCP_SIG5 16 31
regSE_CAC_WEIGHT_TCP_3 = 0x1b7c
# 	WEIGHT_TCP_SIG6 0 15
# 	WEIGHT_TCP_SIG7 16 31
regSE_CAC_WEIGHT_TD_0 = 0x1b73
# 	WEIGHT_TD_SIG0 0 15
# 	WEIGHT_TD_SIG1 16 31
regSE_CAC_WEIGHT_TD_1 = 0x1b74
# 	WEIGHT_TD_SIG2 0 15
# 	WEIGHT_TD_SIG3 16 31
regSE_CAC_WEIGHT_TD_2 = 0x1b75
# 	WEIGHT_TD_SIG4 0 15
# 	WEIGHT_TD_SIG5 16 31
regSE_CAC_WEIGHT_TD_3 = 0x1b76
# 	WEIGHT_TD_SIG6 0 15
# 	WEIGHT_TD_SIG7 16 31
regSE_CAC_WEIGHT_TD_4 = 0x1b77
# 	WEIGHT_TD_SIG8 0 15
# 	WEIGHT_TD_SIG9 16 31
regSE_CAC_WEIGHT_TD_5 = 0x1b78
# 	WEIGHT_TD_SIG10 0 15
regSE_CAC_WEIGHT_UTCL1_0 = 0x1ba0
# 	WEIGHT_UTCL1_SIG0 0 15
regSE_CAC_WINDOW_AGGR_VALUE = 0x1bb0
# 	SE_CAC_WINDOW_AGGR_VALUE 0 31
regSE_CAC_WINDOW_GFXCLK_CYCLE = 0x1bb1
# 	SE_CAC_WINDOW_GFXCLK_CYCLE 0 9
regSH_MEM_BASES = 0x9e3
# 	PRIVATE_BASE 0 15
# 	SHARED_BASE 16 31
regSH_MEM_CONFIG = 0x9e4
# 	ADDRESS_MODE 0 0
# 	ALIGNMENT_MODE 2 3
# 	INITIAL_INST_PREFETCH 14 15
# 	ICACHE_USE_GL1 18 18
regSH_RESERVED_REG0 = 0x1c20
# 	DATA 0 31
regSH_RESERVED_REG1 = 0x1c21
# 	DATA 0 31
regSMU_RLC_RESPONSE = 0x4e01
# 	RESP 0 31
regSPI_ARB_CNTL_0 = 0x1949
# 	EXP_ARB_COL_WT 0 3
# 	EXP_ARB_POS_WT 4 7
# 	EXP_ARB_GDS_WT 8 11
regSPI_ARB_CYCLES_0 = 0x1f61
# 	TS0_DURATION 0 15
# 	TS1_DURATION 16 31
regSPI_ARB_CYCLES_1 = 0x1f62
# 	TS2_DURATION 0 15
# 	TS3_DURATION 16 31
regSPI_ARB_PRIORITY = 0x1f60
# 	PIPE_ORDER_TS0 0 2
# 	PIPE_ORDER_TS1 3 5
# 	PIPE_ORDER_TS2 6 8
# 	PIPE_ORDER_TS3 9 11
# 	TS0_DUR_MULT 12 13
# 	TS1_DUR_MULT 14 15
# 	TS2_DUR_MULT 16 17
# 	TS3_DUR_MULT 18 19
regSPI_ATTRIBUTE_RING_BASE = 0x2446
# 	BASE 0 31
regSPI_ATTRIBUTE_RING_SIZE = 0x2447
# 	MEM_SIZE 0 7
# 	BIG_PAGE 16 16
# 	L1_POLICY 17 18
# 	L2_POLICY 19 20
# 	LLC_NOALLOC 21 21
# 	GL1_PERF_COUNTER_DISABLE 22 22
regSPI_BARYC_CNTL = 0x1b8
# 	PERSP_CENTER_CNTL 0 0
# 	PERSP_CENTROID_CNTL 4 4
# 	LINEAR_CENTER_CNTL 8 8
# 	LINEAR_CENTROID_CNTL 12 12
# 	POS_FLOAT_LOCATION 16 17
# 	POS_FLOAT_ULC 20 20
# 	FRONT_FACE_ALL_BITS 24 24
regSPI_COMPUTE_QUEUE_RESET = 0x1f73
# 	RESET 0 0
regSPI_COMPUTE_WF_CTX_SAVE = 0x1f74
# 	INITIATE 0 0
# 	GDS_INTERRUPT_EN 1 1
# 	DONE_INTERRUPT_EN 2 2
# 	GDS_REQ_BUSY 30 30
# 	SAVE_BUSY 31 31
regSPI_COMPUTE_WF_CTX_SAVE_STATUS = 0x194e
# 	PIPE0_QUEUE0_SAVE_BUSY 0 0
# 	PIPE0_QUEUE1_SAVE_BUSY 1 1
# 	PIPE0_QUEUE2_SAVE_BUSY 2 2
# 	PIPE0_QUEUE3_SAVE_BUSY 3 3
# 	PIPE0_QUEUE4_SAVE_BUSY 4 4
# 	PIPE0_QUEUE5_SAVE_BUSY 5 5
# 	PIPE0_QUEUE6_SAVE_BUSY 6 6
# 	PIPE0_QUEUE7_SAVE_BUSY 7 7
# 	PIPE1_QUEUE0_SAVE_BUSY 8 8
# 	PIPE1_QUEUE1_SAVE_BUSY 9 9
# 	PIPE1_QUEUE2_SAVE_BUSY 10 10
# 	PIPE1_QUEUE3_SAVE_BUSY 11 11
# 	PIPE1_QUEUE4_SAVE_BUSY 12 12
# 	PIPE1_QUEUE5_SAVE_BUSY 13 13
# 	PIPE1_QUEUE6_SAVE_BUSY 14 14
# 	PIPE1_QUEUE7_SAVE_BUSY 15 15
# 	PIPE2_QUEUE0_SAVE_BUSY 16 16
# 	PIPE2_QUEUE1_SAVE_BUSY 17 17
# 	PIPE2_QUEUE2_SAVE_BUSY 18 18
# 	PIPE2_QUEUE3_SAVE_BUSY 19 19
# 	PIPE2_QUEUE4_SAVE_BUSY 20 20
# 	PIPE2_QUEUE5_SAVE_BUSY 21 21
# 	PIPE2_QUEUE6_SAVE_BUSY 22 22
# 	PIPE2_QUEUE7_SAVE_BUSY 23 23
# 	PIPE3_QUEUE0_SAVE_BUSY 24 24
# 	PIPE3_QUEUE1_SAVE_BUSY 25 25
# 	PIPE3_QUEUE2_SAVE_BUSY 26 26
# 	PIPE3_QUEUE3_SAVE_BUSY 27 27
# 	PIPE3_QUEUE4_SAVE_BUSY 28 28
# 	PIPE3_QUEUE5_SAVE_BUSY 29 29
# 	PIPE3_QUEUE6_SAVE_BUSY 30 30
# 	PIPE3_QUEUE7_SAVE_BUSY 31 31
regSPI_CONFIG_CNTL = 0x2440
# 	GPR_WRITE_PRIORITY 0 20
# 	EXP_PRIORITY_ORDER 21 23
# 	ENABLE_SQG_TOP_EVENTS 24 24
# 	ENABLE_SQG_BOP_EVENTS 25 25
# 	ALLOC_ARB_LRU_ENA 28 28
# 	EXP_ARB_LRU_ENA 29 29
# 	PS_PKR_PRIORITY_CNTL 30 31
regSPI_CONFIG_CNTL_1 = 0x2441
# 	VTX_DONE_DELAY 0 3
# 	INTERP_ONE_PRIM_PER_ROW 4 4
# 	PC_LIMIT_ENABLE 5 6
# 	PC_LIMIT_STRICT 7 7
# 	PS_GROUP_TIMEOUT_MODE 8 8
# 	OREO_EXPALLOC_STALL 9 9
# 	LBPW_CU_CHK_CNT 10 13
# 	CSC_PWR_SAVE_DISABLE 14 14
# 	CSG_PWR_SAVE_DISABLE 15 15
# 	MAX_VTX_SYNC_CNT 16 20
# 	EN_USER_ACCUM 21 21
# 	SA_SCREEN_MAP 22 22
# 	PS_GROUP_TIMEOUT 23 31
regSPI_CONFIG_CNTL_2 = 0x2442
# 	CONTEXT_SAVE_WAIT_GDS_REQUEST_CYCLE_OVHD 0 3
# 	CONTEXT_SAVE_WAIT_GDS_GRANT_CYCLE_OVHD 4 7
# 	PWS_CSG_WAIT_DISABLE 8 8
# 	PWS_HS_WAIT_DISABLE 9 9
# 	PWS_GS_WAIT_DISABLE 10 10
# 	PWS_PS_WAIT_DISABLE 11 11
# 	CSC_HALT_ACK_DELAY 12 16
regSPI_CONFIG_PS_CU_EN = 0x11f2
# 	PKR_OFFSET 0 3
# 	PKR2_OFFSET 4 7
# 	PKR3_OFFSET 8 11
regSPI_CSQ_WF_ACTIVE_COUNT_0 = 0x127c
# 	COUNT 0 10
# 	EVENTS 16 26
regSPI_CSQ_WF_ACTIVE_COUNT_1 = 0x127d
# 	COUNT 0 10
# 	EVENTS 16 26
regSPI_CSQ_WF_ACTIVE_COUNT_2 = 0x127e
# 	COUNT 0 10
# 	EVENTS 16 26
regSPI_CSQ_WF_ACTIVE_COUNT_3 = 0x127f
# 	COUNT 0 10
# 	EVENTS 16 26
regSPI_CSQ_WF_ACTIVE_STATUS = 0x127b
# 	ACTIVE 0 31
regSPI_DSM_CNTL = 0x11e3
# 	SPI_SR_MEM_DSM_IRRITATOR_DATA 0 1
# 	SPI_SR_MEM_ENABLE_SINGLE_WRITE 2 2
regSPI_DSM_CNTL2 = 0x11e4
# 	SPI_SR_MEM_ENABLE_ERROR_INJECT 0 1
# 	SPI_SR_MEM_SELECT_INJECT_DELAY 2 2
# 	SPI_SR_MEM_INJECT_DELAY 3 8
regSPI_EDC_CNT = 0x11e5
# 	SPI_SR_MEM_SED_COUNT 0 1
regSPI_EXP_THROTTLE_CTRL = 0x14c3
# 	ENABLE 0 0
# 	PERIOD 1 4
# 	UPSTEP 5 8
# 	DOWNSTEP 9 12
# 	LOW_STALL_MON_HIST_COUNT 13 15
# 	HIGH_STALL_MON_HIST_COUNT 16 18
# 	EXP_STALL_THRESHOLD 19 25
# 	SKEW_COUNT 26 28
# 	THROTTLE_RESET 29 29
regSPI_FEATURE_CTRL = 0x194a
# 	TUNNELING_WAVE_LIMIT 0 3
# 	RA_PROBE_IGNORE 4 4
# 	PS_THROTTLE_MAX_WAVE_LIMIT 5 10
# 	RA_PROBE_SKEW_WIF_CTRL 11 12
# 	RA_PROBE_SKEW_OOO_CTRL 13 13
# 	RA_PROBE_SKEW_DISABLE 14 14
regSPI_GDBG_PER_VMID_CNTL = 0x1f72
# 	STALL_VMID 0 0
# 	LAUNCH_MODE 1 2
# 	TRAP_EN 3 3
# 	EXCP_EN 4 12
# 	EXCP_REPLACE 13 13
# 	TRAP_ON_START 14 14
# 	TRAP_ON_END 15 15
regSPI_GDBG_TRAP_CONFIG = 0x1944
# 	PIPE0_EN 0 7
# 	PIPE1_EN 8 15
# 	PIPE2_EN 16 23
# 	PIPE3_EN 24 31
regSPI_GDBG_WAVE_CNTL = 0x1943
# 	STALL_RA 0 0
# 	STALL_LAUNCH 1 1
regSPI_GDBG_WAVE_CNTL3 = 0x1945
# 	STALL_PS 0 0
# 	STALL_GS 2 2
# 	STALL_HS 3 3
# 	STALL_CSG 4 4
# 	STALL_CS0 5 5
# 	STALL_CS1 6 6
# 	STALL_CS2 7 7
# 	STALL_CS3 8 8
# 	STALL_CS4 9 9
# 	STALL_CS5 10 10
# 	STALL_CS6 11 11
# 	STALL_CS7 12 12
# 	STALL_DURATION 13 27
# 	STALL_MULT 28 28
regSPI_GDS_CREDITS = 0x1278
# 	DS_DATA_CREDITS 0 7
# 	DS_CMD_CREDITS 8 15
regSPI_GFX_CNTL = 0x11dc
# 	RESET_COUNTS 0 0
regSPI_GFX_SCRATCH_BASE_HI = 0x1bc
# 	DATA 0 7
regSPI_GFX_SCRATCH_BASE_LO = 0x1bb
# 	DATA 0 31
regSPI_GS_THROTTLE_CNTL1 = 0x2444
# 	PH_POLL_INTERVAL 0 3
# 	PH_THROTTLE_BASE 4 7
# 	PH_THROTTLE_STEP_SIZE 8 11
# 	SPI_VGPR_THRESHOLD 12 15
# 	SPI_LDS_THRESHOLD 16 19
# 	SPI_POLL_INTERVAL 20 23
# 	SPI_THROTTLE_BASE 24 27
# 	SPI_THROTTLE_STEP_SIZE 28 31
regSPI_GS_THROTTLE_CNTL2 = 0x2445
# 	SPI_THROTTLE_MODE 0 1
# 	GRP_LIFETIME_THRESHOLD 2 5
# 	GRP_LIFETIME_THRESHOLD_FACTOR 6 7
# 	GRP_LIFETIME_PENALTY1 8 10
# 	GRP_LIFETIME_PENALTY2 11 13
# 	PS_STALL_THRESHOLD 14 15
# 	PH_MODE 16 16
# 	RESERVED 17 31
regSPI_INTERP_CONTROL_0 = 0x1b5
# 	FLAT_SHADE_ENA 0 0
# 	PNT_SPRITE_ENA 1 1
# 	PNT_SPRITE_OVRD_X 2 4
# 	PNT_SPRITE_OVRD_Y 5 7
# 	PNT_SPRITE_OVRD_Z 8 10
# 	PNT_SPRITE_OVRD_W 11 13
# 	PNT_SPRITE_TOP_1 14 14
regSPI_LB_CTR_CTRL = 0x1274
# 	LOAD 0 0
# 	WAVES_SELECT 1 2
# 	CLEAR_ON_READ 3 3
# 	RESET_COUNTS 4 4
regSPI_LB_DATA_REG = 0x1276
# 	CNT_DATA 0 31
regSPI_LB_DATA_WAVES = 0x1284
# 	COUNT0 0 15
# 	COUNT1 16 31
regSPI_LB_WGP_MASK = 0x1275
# 	WGP_MASK 0 15
regSPI_P0_TRAP_SCREEN_GPR_MIN = 0x1290
# 	VGPR_MIN 0 5
# 	SGPR_MIN 6 9
regSPI_P0_TRAP_SCREEN_PSBA_HI = 0x128d
# 	MEM_BASE 0 7
regSPI_P0_TRAP_SCREEN_PSBA_LO = 0x128c
# 	MEM_BASE 0 31
regSPI_P0_TRAP_SCREEN_PSMA_HI = 0x128f
# 	MEM_BASE 0 7
regSPI_P0_TRAP_SCREEN_PSMA_LO = 0x128e
# 	MEM_BASE 0 31
regSPI_P1_TRAP_SCREEN_GPR_MIN = 0x1295
# 	VGPR_MIN 0 5
# 	SGPR_MIN 6 9
regSPI_P1_TRAP_SCREEN_PSBA_HI = 0x1292
# 	MEM_BASE 0 7
regSPI_P1_TRAP_SCREEN_PSBA_LO = 0x1291
# 	MEM_BASE 0 31
regSPI_P1_TRAP_SCREEN_PSMA_HI = 0x1294
# 	MEM_BASE 0 7
regSPI_P1_TRAP_SCREEN_PSMA_LO = 0x1293
# 	MEM_BASE 0 31
regSPI_PERFCOUNTER0_HI = 0x3180
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER0_LO = 0x3181
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER0_SELECT = 0x3980
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSPI_PERFCOUNTER0_SELECT1 = 0x3984
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSPI_PERFCOUNTER1_HI = 0x3182
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER1_LO = 0x3183
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER1_SELECT = 0x3981
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSPI_PERFCOUNTER1_SELECT1 = 0x3985
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSPI_PERFCOUNTER2_HI = 0x3184
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER2_LO = 0x3185
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER2_SELECT = 0x3982
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSPI_PERFCOUNTER2_SELECT1 = 0x3986
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSPI_PERFCOUNTER3_HI = 0x3186
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER3_LO = 0x3187
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER3_SELECT = 0x3983
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSPI_PERFCOUNTER3_SELECT1 = 0x3987
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSPI_PERFCOUNTER4_HI = 0x3188
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER4_LO = 0x3189
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER4_SELECT = 0x3988
# 	PERF_SEL 0 9
regSPI_PERFCOUNTER5_HI = 0x318a
# 	PERFCOUNTER_HI 0 31
regSPI_PERFCOUNTER5_LO = 0x318b
# 	PERFCOUNTER_LO 0 31
regSPI_PERFCOUNTER5_SELECT = 0x3989
# 	PERF_SEL 0 9
regSPI_PERFCOUNTER_BINS = 0x398a
# 	BIN0_MIN 0 3
# 	BIN0_MAX 4 7
# 	BIN1_MIN 8 11
# 	BIN1_MAX 12 15
# 	BIN2_MIN 16 19
# 	BIN2_MAX 20 23
# 	BIN3_MIN 24 27
# 	BIN3_MAX 28 31
regSPI_PG_ENABLE_STATIC_WGP_MASK = 0x1277
# 	WGP_MASK 0 15
regSPI_PQEV_CTRL = 0x14c0
# 	SCAN_PERIOD 0 9
# 	QUEUE_DURATION 10 15
# 	COMPUTE_PIPE_EN 16 23
regSPI_PS_INPUT_ADDR = 0x1b4
# 	PERSP_SAMPLE_ENA 0 0
# 	PERSP_CENTER_ENA 1 1
# 	PERSP_CENTROID_ENA 2 2
# 	PERSP_PULL_MODEL_ENA 3 3
# 	LINEAR_SAMPLE_ENA 4 4
# 	LINEAR_CENTER_ENA 5 5
# 	LINEAR_CENTROID_ENA 6 6
# 	LINE_STIPPLE_TEX_ENA 7 7
# 	POS_X_FLOAT_ENA 8 8
# 	POS_Y_FLOAT_ENA 9 9
# 	POS_Z_FLOAT_ENA 10 10
# 	POS_W_FLOAT_ENA 11 11
# 	FRONT_FACE_ENA 12 12
# 	ANCILLARY_ENA 13 13
# 	SAMPLE_COVERAGE_ENA 14 14
# 	POS_FIXED_PT_ENA 15 15
regSPI_PS_INPUT_CNTL_0 = 0x191
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_1 = 0x192
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_10 = 0x19b
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_11 = 0x19c
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_12 = 0x19d
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_13 = 0x19e
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_14 = 0x19f
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_15 = 0x1a0
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_16 = 0x1a1
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_17 = 0x1a2
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_18 = 0x1a3
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_19 = 0x1a4
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_2 = 0x193
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_20 = 0x1a5
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_21 = 0x1a6
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_22 = 0x1a7
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_23 = 0x1a8
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_24 = 0x1a9
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_25 = 0x1aa
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_26 = 0x1ab
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_27 = 0x1ac
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_28 = 0x1ad
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_29 = 0x1ae
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_3 = 0x194
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_30 = 0x1af
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_31 = 0x1b0
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_4 = 0x195
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_5 = 0x196
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_6 = 0x197
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_7 = 0x198
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_8 = 0x199
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_CNTL_9 = 0x19a
# 	OFFSET 0 5
# 	DEFAULT_VAL 8 9
# 	FLAT_SHADE 10 10
# 	ROTATE_PC_PTR 11 11
# 	PRIM_ATTR 12 12
# 	PT_SPRITE_TEX 17 17
# 	DUP 18 18
# 	FP16_INTERP_MODE 19 19
# 	USE_DEFAULT_ATTR1 20 20
# 	DEFAULT_VAL_ATTR1 21 22
# 	PT_SPRITE_TEX_ATTR1 23 23
# 	ATTR0_VALID 24 24
# 	ATTR1_VALID 25 25
regSPI_PS_INPUT_ENA = 0x1b3
# 	PERSP_SAMPLE_ENA 0 0
# 	PERSP_CENTER_ENA 1 1
# 	PERSP_CENTROID_ENA 2 2
# 	PERSP_PULL_MODEL_ENA 3 3
# 	LINEAR_SAMPLE_ENA 4 4
# 	LINEAR_CENTER_ENA 5 5
# 	LINEAR_CENTROID_ENA 6 6
# 	LINE_STIPPLE_TEX_ENA 7 7
# 	POS_X_FLOAT_ENA 8 8
# 	POS_Y_FLOAT_ENA 9 9
# 	POS_Z_FLOAT_ENA 10 10
# 	POS_W_FLOAT_ENA 11 11
# 	FRONT_FACE_ENA 12 12
# 	ANCILLARY_ENA 13 13
# 	SAMPLE_COVERAGE_ENA 14 14
# 	POS_FIXED_PT_ENA 15 15
regSPI_PS_IN_CONTROL = 0x1b6
# 	NUM_INTERP 0 5
# 	PARAM_GEN 6 6
# 	OFFCHIP_PARAM_EN 7 7
# 	LATE_PC_DEALLOC 8 8
# 	NUM_PRIM_INTERP 9 13
# 	BC_OPTIMIZE_DISABLE 14 14
# 	PS_W32_EN 15 15
regSPI_PS_MAX_WAVE_ID = 0x11da
# 	MAX_WAVE_ID 0 11
# 	MAX_COLLISION_WAVE_ID 16 25
regSPI_RESOURCE_RESERVE_CU_0 = 0x1c00
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_1 = 0x1c01
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_10 = 0x1c0a
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_11 = 0x1c0b
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_12 = 0x1c0c
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_13 = 0x1c0d
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_14 = 0x1c0e
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_15 = 0x1c0f
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_2 = 0x1c02
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_3 = 0x1c03
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_4 = 0x1c04
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_5 = 0x1c05
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_6 = 0x1c06
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_7 = 0x1c07
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_8 = 0x1c08
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_CU_9 = 0x1c09
# 	VGPR 0 3
# 	SGPR 4 7
# 	LDS 8 11
# 	WAVES 12 14
# 	BARRIERS 15 18
regSPI_RESOURCE_RESERVE_EN_CU_0 = 0x1c10
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_1 = 0x1c11
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_10 = 0x1c1a
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_11 = 0x1c1b
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_12 = 0x1c1c
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_13 = 0x1c1d
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_14 = 0x1c1e
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_15 = 0x1c1f
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_2 = 0x1c12
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_3 = 0x1c13
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_4 = 0x1c14
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_5 = 0x1c15
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_6 = 0x1c16
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_7 = 0x1c17
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_8 = 0x1c18
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_RESOURCE_RESERVE_EN_CU_9 = 0x1c19
# 	EN 0 0
# 	TYPE_MASK 1 15
# 	QUEUE_MASK 16 23
regSPI_SHADER_COL_FORMAT = 0x1c5
# 	COL0_EXPORT_FORMAT 0 3
# 	COL1_EXPORT_FORMAT 4 7
# 	COL2_EXPORT_FORMAT 8 11
# 	COL3_EXPORT_FORMAT 12 15
# 	COL4_EXPORT_FORMAT 16 19
# 	COL5_EXPORT_FORMAT 20 23
# 	COL6_EXPORT_FORMAT 24 27
# 	COL7_EXPORT_FORMAT 28 31
regSPI_SHADER_GS_MESHLET_DIM = 0x1a4c
# 	MESHLET_NUM_THREAD_X 0 7
# 	MESHLET_NUM_THREAD_Y 8 15
# 	MESHLET_NUM_THREAD_Z 16 23
# 	MESHLET_THREADGROUP_SIZE 24 31
regSPI_SHADER_GS_MESHLET_EXP_ALLOC = 0x1a4d
# 	MAX_EXP_VERTS 0 8
# 	MAX_EXP_PRIMS 9 17
regSPI_SHADER_IDX_FORMAT = 0x1c2
# 	IDX0_EXPORT_FORMAT 0 3
regSPI_SHADER_PGM_CHKSUM_GS = 0x1a20
# 	CHECKSUM 0 31
regSPI_SHADER_PGM_CHKSUM_HS = 0x1aa0
# 	CHECKSUM 0 31
regSPI_SHADER_PGM_CHKSUM_PS = 0x19a6
# 	CHECKSUM 0 31
regSPI_SHADER_PGM_HI_ES = 0x1a69
# 	MEM_BASE 0 7
regSPI_SHADER_PGM_HI_ES_GS = 0x1a25
# 	MEM_BASE 0 7
regSPI_SHADER_PGM_HI_GS = 0x1a29
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_HI_HS = 0x1aa9
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_HI_LS = 0x1ae9
# 	MEM_BASE 0 7
regSPI_SHADER_PGM_HI_LS_HS = 0x1aa5
# 	MEM_BASE 0 7
regSPI_SHADER_PGM_HI_PS = 0x19a9
# 	MEM_BASE 0 7
regSPI_SHADER_PGM_LO_ES = 0x1a68
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_ES_GS = 0x1a24
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_GS = 0x1a28
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_HS = 0x1aa8
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_LS = 0x1ae8
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_LS_HS = 0x1aa4
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_LO_PS = 0x19a8
# 	MEM_BASE 0 31
regSPI_SHADER_PGM_RSRC1_GS = 0x1a2a
# 	VGPRS 0 5
# 	SGPRS 6 9
# 	PRIORITY 10 11
# 	FLOAT_MODE 12 19
# 	PRIV 20 20
# 	DX10_CLAMP 21 21
# 	IEEE_MODE 23 23
# 	CU_GROUP_ENABLE 24 24
# 	MEM_ORDERED 25 25
# 	FWD_PROGRESS 26 26
# 	WGP_MODE 27 27
# 	GS_VGPR_COMP_CNT 29 30
# 	FP16_OVFL 31 31
regSPI_SHADER_PGM_RSRC1_HS = 0x1aaa
# 	VGPRS 0 5
# 	SGPRS 6 9
# 	PRIORITY 10 11
# 	FLOAT_MODE 12 19
# 	PRIV 20 20
# 	DX10_CLAMP 21 21
# 	IEEE_MODE 23 23
# 	MEM_ORDERED 24 24
# 	FWD_PROGRESS 25 25
# 	WGP_MODE 26 26
# 	LS_VGPR_COMP_CNT 28 29
# 	FP16_OVFL 30 30
regSPI_SHADER_PGM_RSRC1_PS = 0x19aa
# 	VGPRS 0 5
# 	SGPRS 6 9
# 	PRIORITY 10 11
# 	FLOAT_MODE 12 19
# 	PRIV 20 20
# 	DX10_CLAMP 21 21
# 	IEEE_MODE 23 23
# 	CU_GROUP_DISABLE 24 24
# 	MEM_ORDERED 25 25
# 	FWD_PROGRESS 26 26
# 	LOAD_PROVOKING_VTX 27 27
# 	FP16_OVFL 29 29
regSPI_SHADER_PGM_RSRC2_GS = 0x1a2b
# 	SCRATCH_EN 0 0
# 	USER_SGPR 1 5
# 	TRAP_PRESENT 6 6
# 	EXCP_EN 7 15
# 	ES_VGPR_COMP_CNT 16 17
# 	OC_LDS_EN 18 18
# 	LDS_SIZE 19 26
# 	USER_SGPR_MSB 27 27
# 	SHARED_VGPR_CNT 28 31
regSPI_SHADER_PGM_RSRC2_HS = 0x1aab
# 	SCRATCH_EN 0 0
# 	USER_SGPR 1 5
# 	TRAP_PRESENT 6 6
# 	OC_LDS_EN 7 7
# 	TG_SIZE_EN 8 8
# 	EXCP_EN 9 17
# 	LDS_SIZE 18 26
# 	USER_SGPR_MSB 27 27
# 	SHARED_VGPR_CNT 28 31
regSPI_SHADER_PGM_RSRC2_PS = 0x19ab
# 	SCRATCH_EN 0 0
# 	USER_SGPR 1 5
# 	TRAP_PRESENT 6 6
# 	WAVE_CNT_EN 7 7
# 	EXTRA_LDS_SIZE 8 15
# 	EXCP_EN 16 24
# 	LOAD_COLLISION_WAVEID 25 25
# 	LOAD_INTRAWAVE_COLLISION 26 26
# 	USER_SGPR_MSB 27 27
# 	SHARED_VGPR_CNT 28 31
regSPI_SHADER_PGM_RSRC3_GS = 0x1a27
# 	CU_EN 0 15
# 	WAVE_LIMIT 16 21
# 	LOCK_LOW_THRESHOLD 22 25
# 	GROUP_FIFO_DEPTH 26 31
regSPI_SHADER_PGM_RSRC3_HS = 0x1aa7
# 	WAVE_LIMIT 0 5
# 	LOCK_LOW_THRESHOLD 6 9
# 	GROUP_FIFO_DEPTH 10 15
# 	CU_EN 16 31
regSPI_SHADER_PGM_RSRC3_PS = 0x19a7
# 	CU_EN 0 15
# 	WAVE_LIMIT 16 21
# 	LDS_GROUP_SIZE 22 23
regSPI_SHADER_PGM_RSRC4_GS = 0x1a21
# 	CU_EN 0 0
# 	RESERVED 1 13
# 	PH_THROTTLE_EN 14 14
# 	SPI_THROTTLE_EN 15 15
# 	SPI_SHADER_LATE_ALLOC_GS 16 22
# 	INST_PREF_SIZE 23 28
# 	TRAP_ON_START 29 29
# 	TRAP_ON_END 30 30
# 	IMAGE_OP 31 31
regSPI_SHADER_PGM_RSRC4_HS = 0x1aa1
# 	CU_EN 0 15
# 	INST_PREF_SIZE 16 21
# 	TRAP_ON_START 29 29
# 	TRAP_ON_END 30 30
# 	IMAGE_OP 31 31
regSPI_SHADER_PGM_RSRC4_PS = 0x19a1
# 	CU_EN 0 15
# 	INST_PREF_SIZE 16 21
# 	TRAP_ON_START 29 29
# 	TRAP_ON_END 30 30
# 	IMAGE_OP 31 31
regSPI_SHADER_POS_FORMAT = 0x1c3
# 	POS0_EXPORT_FORMAT 0 3
# 	POS1_EXPORT_FORMAT 4 7
# 	POS2_EXPORT_FORMAT 8 11
# 	POS3_EXPORT_FORMAT 12 15
# 	POS4_EXPORT_FORMAT 16 19
regSPI_SHADER_REQ_CTRL_ESGS = 0x1a50
# 	SOFT_GROUPING_EN 0 0
# 	NUMBER_OF_REQUESTS_PER_CU 1 4
# 	SOFT_GROUPING_ALLOCATION_TIMEOUT 5 8
# 	HARD_LOCK_HYSTERESIS 9 9
# 	HARD_LOCK_LOW_THRESHOLD 10 14
# 	PRODUCER_REQUEST_LOCKOUT 15 15
# 	GLOBAL_SCANNING_EN 16 16
# 	ALLOCATION_RATE_THROTTLING_THRESHOLD 17 19
regSPI_SHADER_REQ_CTRL_LSHS = 0x1ad0
# 	SOFT_GROUPING_EN 0 0
# 	NUMBER_OF_REQUESTS_PER_CU 1 4
# 	SOFT_GROUPING_ALLOCATION_TIMEOUT 5 8
# 	HARD_LOCK_HYSTERESIS 9 9
# 	HARD_LOCK_LOW_THRESHOLD 10 14
# 	PRODUCER_REQUEST_LOCKOUT 15 15
# 	GLOBAL_SCANNING_EN 16 16
# 	ALLOCATION_RATE_THROTTLING_THRESHOLD 17 19
regSPI_SHADER_REQ_CTRL_PS = 0x19d0
# 	SOFT_GROUPING_EN 0 0
# 	NUMBER_OF_REQUESTS_PER_CU 1 4
# 	SOFT_GROUPING_ALLOCATION_TIMEOUT 5 8
# 	HARD_LOCK_HYSTERESIS 9 9
# 	HARD_LOCK_LOW_THRESHOLD 10 14
# 	PRODUCER_REQUEST_LOCKOUT 15 15
# 	GLOBAL_SCANNING_EN 16 16
# 	ALLOCATION_RATE_THROTTLING_THRESHOLD 17 19
regSPI_SHADER_RSRC_LIMIT_CTRL = 0x194b
# 	WAVES_PER_SIMD32 0 4
# 	VGPR_PER_SIMD32 5 11
# 	VGPR_WRAP_DISABLE 12 12
# 	BARRIER_LIMIT 13 18
# 	BARRIER_LIMIT_HIERARCHY_LEVEL 19 19
# 	LDS_LIMIT 20 27
# 	LDS_LIMIT_HIERARCHY_LEVEL 28 28
# 	PERFORMANCE_LIMIT_ENABLE 31 31
regSPI_SHADER_USER_ACCUM_ESGS_0 = 0x1a52
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_ESGS_1 = 0x1a53
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_ESGS_2 = 0x1a54
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_ESGS_3 = 0x1a55
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_LSHS_0 = 0x1ad2
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_LSHS_1 = 0x1ad3
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_LSHS_2 = 0x1ad4
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_LSHS_3 = 0x1ad5
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_PS_0 = 0x19d2
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_PS_1 = 0x19d3
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_PS_2 = 0x19d4
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_ACCUM_PS_3 = 0x19d5
# 	CONTRIBUTION 0 6
regSPI_SHADER_USER_DATA_ADDR_HI_GS = 0x1a23
# 	MEM_BASE 0 31
regSPI_SHADER_USER_DATA_ADDR_HI_HS = 0x1aa3
# 	MEM_BASE 0 31
regSPI_SHADER_USER_DATA_ADDR_LO_GS = 0x1a22
# 	MEM_BASE 0 31
regSPI_SHADER_USER_DATA_ADDR_LO_HS = 0x1aa2
# 	MEM_BASE 0 31
regSPI_SHADER_USER_DATA_GS_0 = 0x1a2c
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_1 = 0x1a2d
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_10 = 0x1a36
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_11 = 0x1a37
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_12 = 0x1a38
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_13 = 0x1a39
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_14 = 0x1a3a
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_15 = 0x1a3b
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_16 = 0x1a3c
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_17 = 0x1a3d
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_18 = 0x1a3e
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_19 = 0x1a3f
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_2 = 0x1a2e
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_20 = 0x1a40
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_21 = 0x1a41
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_22 = 0x1a42
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_23 = 0x1a43
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_24 = 0x1a44
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_25 = 0x1a45
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_26 = 0x1a46
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_27 = 0x1a47
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_28 = 0x1a48
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_29 = 0x1a49
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_3 = 0x1a2f
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_30 = 0x1a4a
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_31 = 0x1a4b
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_4 = 0x1a30
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_5 = 0x1a31
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_6 = 0x1a32
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_7 = 0x1a33
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_8 = 0x1a34
# 	DATA 0 31
regSPI_SHADER_USER_DATA_GS_9 = 0x1a35
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_0 = 0x1aac
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_1 = 0x1aad
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_10 = 0x1ab6
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_11 = 0x1ab7
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_12 = 0x1ab8
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_13 = 0x1ab9
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_14 = 0x1aba
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_15 = 0x1abb
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_16 = 0x1abc
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_17 = 0x1abd
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_18 = 0x1abe
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_19 = 0x1abf
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_2 = 0x1aae
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_20 = 0x1ac0
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_21 = 0x1ac1
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_22 = 0x1ac2
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_23 = 0x1ac3
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_24 = 0x1ac4
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_25 = 0x1ac5
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_26 = 0x1ac6
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_27 = 0x1ac7
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_28 = 0x1ac8
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_29 = 0x1ac9
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_3 = 0x1aaf
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_30 = 0x1aca
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_31 = 0x1acb
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_4 = 0x1ab0
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_5 = 0x1ab1
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_6 = 0x1ab2
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_7 = 0x1ab3
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_8 = 0x1ab4
# 	DATA 0 31
regSPI_SHADER_USER_DATA_HS_9 = 0x1ab5
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_0 = 0x19ac
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_1 = 0x19ad
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_10 = 0x19b6
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_11 = 0x19b7
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_12 = 0x19b8
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_13 = 0x19b9
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_14 = 0x19ba
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_15 = 0x19bb
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_16 = 0x19bc
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_17 = 0x19bd
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_18 = 0x19be
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_19 = 0x19bf
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_2 = 0x19ae
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_20 = 0x19c0
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_21 = 0x19c1
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_22 = 0x19c2
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_23 = 0x19c3
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_24 = 0x19c4
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_25 = 0x19c5
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_26 = 0x19c6
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_27 = 0x19c7
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_28 = 0x19c8
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_29 = 0x19c9
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_3 = 0x19af
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_30 = 0x19ca
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_31 = 0x19cb
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_4 = 0x19b0
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_5 = 0x19b1
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_6 = 0x19b2
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_7 = 0x19b3
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_8 = 0x19b4
# 	DATA 0 31
regSPI_SHADER_USER_DATA_PS_9 = 0x19b5
# 	DATA 0 31
regSPI_SHADER_Z_FORMAT = 0x1c4
# 	Z_EXPORT_FORMAT 0 3
regSPI_SX_EXPORT_BUFFER_SIZES = 0x1279
# 	COLOR_BUFFER_SIZE 0 15
# 	POSITION_BUFFER_SIZE 16 31
regSPI_SX_SCOREBOARD_BUFFER_SIZES = 0x127a
# 	COLOR_SCOREBOARD_SIZE 0 15
# 	POSITION_SCOREBOARD_SIZE 16 31
regSPI_TMPRING_SIZE = 0x1ba
# 	WAVES 0 11
# 	WAVESIZE 12 26
regSPI_USER_ACCUM_VMID_CNTL = 0x1f71
# 	EN_USER_ACCUM 0 3
regSPI_VS_OUT_CONFIG = 0x1b1
# 	VS_EXPORT_COUNT 1 5
# 	NO_PC_EXPORT 7 7
# 	PRIM_EXPORT_COUNT 8 12
regSPI_WAVE_LIMIT_CNTL = 0x2443
# 	PS_WAVE_GRAN 0 1
# 	GS_WAVE_GRAN 4 5
# 	HS_WAVE_GRAN 6 7
regSPI_WCL_PIPE_PERCENT_CS0 = 0x1f69
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS1 = 0x1f6a
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS2 = 0x1f6b
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS3 = 0x1f6c
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS4 = 0x1f6d
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS5 = 0x1f6e
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS6 = 0x1f6f
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_CS7 = 0x1f70
# 	VALUE 0 6
regSPI_WCL_PIPE_PERCENT_GFX = 0x1f67
# 	VALUE 0 6
# 	HS_GRP_VALUE 12 16
# 	GS_GRP_VALUE 22 26
regSPI_WCL_PIPE_PERCENT_HP3D = 0x1f68
# 	VALUE 0 6
# 	HS_GRP_VALUE 12 16
# 	GS_GRP_VALUE 22 26
regSPI_WF_LIFETIME_CNTL = 0x124a
# 	SAMPLE_PERIOD 0 3
# 	EN 4 4
regSPI_WF_LIFETIME_LIMIT_0 = 0x124b
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_LIMIT_1 = 0x124c
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_LIMIT_2 = 0x124d
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_LIMIT_3 = 0x124e
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_LIMIT_4 = 0x124f
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_LIMIT_5 = 0x1250
# 	MAX_CNT 0 30
# 	EN_WARN 31 31
regSPI_WF_LIFETIME_STATUS_0 = 0x1255
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_11 = 0x1260
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_13 = 0x1262
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_14 = 0x1263
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_15 = 0x1264
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_16 = 0x1265
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_17 = 0x1266
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_18 = 0x1267
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_19 = 0x1268
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_2 = 0x1257
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_20 = 0x1269
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_21 = 0x126b
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_4 = 0x1259
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_6 = 0x125b
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_7 = 0x125c
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSPI_WF_LIFETIME_STATUS_9 = 0x125e
# 	MAX_CNT 0 30
# 	INT_SENT 31 31
regSP_CONFIG = 0x10ab
# 	DEST_CACHE_EVICT_COUNTER 0 1
# 	ALU_BUSY_MGCG_OVERRIDE 2 2
# 	DISABLE_TRANS_COEXEC 3 3
# 	CAC_COUNTER_OVERRIDE 4 4
# 	SP_SX_EXPVDATA_FGCG_OVERRIDE 5 5
regSQC_CACHES = 0x2348
# 	TARGET_INST 0 0
# 	TARGET_DATA 1 1
# 	INVALIDATE 2 2
# 	COMPLETE 16 16
regSQC_CONFIG = 0x10a1
# 	INST_CACHE_SIZE 0 1
# 	DATA_CACHE_SIZE 2 3
# 	MISS_FIFO_DEPTH 4 5
# 	HIT_FIFO_DEPTH 6 6
# 	FORCE_ALWAYS_MISS 7 7
# 	FORCE_IN_ORDER 8 8
# 	PER_VMID_INV_DISABLE 9 9
# 	EVICT_LRU 10 11
# 	FORCE_2_BANK 12 12
# 	FORCE_1_BANK 13 13
# 	LS_DISABLE_CLOCKS 14 21
# 	CACHE_CTRL_GCR_FIX_DISABLE 22 22
# 	CACHE_CTRL_ALMOST_MAX_INFLIGHT_CONFIG 23 25
# 	SPARE 26 31
regSQG_CONFIG = 0x10ba
# 	GL1H_PREFETCH_PAGE 0 3
# 	SQG_ICPFT_EN 13 13
# 	SQG_ICPFT_CLR 14 14
# 	XNACK_INTR_MASK 16 31
regSQG_GL1H_STATUS = 0x10b9
# 	R0_ACK_ERR_DETECTED 0 0
# 	R0_XNACK_ERR_DETECTED 1 1
# 	R1_ACK_ERR_DETECTED 2 2
# 	R1_XNACK_ERR_DETECTED 3 3
regSQG_PERFCOUNTER0_HI = 0x31e5
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER0_LO = 0x31e4
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER0_SELECT = 0x39d0
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER1_HI = 0x31e7
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER1_LO = 0x31e6
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER1_SELECT = 0x39d1
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER2_HI = 0x31e9
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER2_LO = 0x31e8
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER2_SELECT = 0x39d2
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER3_HI = 0x31eb
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER3_LO = 0x31ea
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER3_SELECT = 0x39d3
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER4_HI = 0x31ed
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER4_LO = 0x31ec
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER4_SELECT = 0x39d4
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER5_HI = 0x31ef
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER5_LO = 0x31ee
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER5_SELECT = 0x39d5
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER6_HI = 0x31f1
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER6_LO = 0x31f0
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER6_SELECT = 0x39d6
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER7_HI = 0x31f3
# 	PERFCOUNTER_HI 0 31
regSQG_PERFCOUNTER7_LO = 0x31f2
# 	PERFCOUNTER_LO 0 31
regSQG_PERFCOUNTER7_SELECT = 0x39d7
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQG_PERFCOUNTER_CTRL = 0x39d8
# 	PS_EN 0 0
# 	GS_EN 2 2
# 	HS_EN 4 4
# 	CS_EN 6 6
# 	DISABLE_ME0PIPE0_PERF 14 14
# 	DISABLE_ME0PIPE1_PERF 15 15
# 	DISABLE_ME1PIPE0_PERF 16 16
# 	DISABLE_ME1PIPE1_PERF 17 17
# 	DISABLE_ME1PIPE2_PERF 18 18
# 	DISABLE_ME1PIPE3_PERF 19 19
regSQG_PERFCOUNTER_CTRL2 = 0x39da
# 	FORCE_EN 0 0
# 	VMID_EN 1 16
regSQG_PERF_SAMPLE_FINISH = 0x39db
# 	STATUS 0 6
regSQG_STATUS = 0x10a4
# 	REG_BUSY 0 0
regSQ_ALU_CLK_CTRL = 0x508e
# 	FORCE_WGP_ON_SA0 0 15
# 	FORCE_WGP_ON_SA1 16 31
regSQ_ARB_CONFIG = 0x10ac
# 	WG_RR_INTERVAL 0 1
# 	FWD_PROG_INTERVAL 4 5
regSQ_CMD = 0x111b
# 	CMD 0 3
# 	MODE 4 6
# 	CHECK_VMID 7 7
# 	DATA 8 11
# 	WAVE_ID 16 20
# 	QUEUE_ID 24 26
# 	VM_ID 28 31
regSQ_CONFIG = 0x10a0
# 	ECO_SPARE 0 7
# 	NEW_TRANS_ARB_SCHEME 8 8
# 	DISABLE_VMEM_EXEC_ZERO_SKIP 9 9
# 	DISABLE_SGPR_RD_KILL 0 0
# 	ENABLE_HIPRIO_ON_EXP_RDY_GS 18 18
# 	PRIO_VAL_ON_EXP_RDY_GS 19 20
# 	WCLK_HYSTERESIS_CNT 0 0
# 	DISABLE_END_CLAUSE_TX 27 27
regSQ_DEBUG = 0x9e5
# 	SINGLE_MEMOP 0 0
# 	SINGLE_ALU_OP 1 1
# 	WAIT_DEP_CTR_ZERO 2 2
regSQ_DEBUG_HOST_TRAP_STATUS = 0x10b6
# 	PENDING_COUNT 0 6
regSQ_DEBUG_STS_GLOBAL = 0x9e1
# 	BUSY 0 0
# 	INTERRUPT_BUSY 1 1
# 	WAVE_LEVEL_SA0 4 15
# 	WAVE_LEVEL_SA1 16 27
regSQ_DEBUG_STS_GLOBAL2 = 0x9e2
# 	REG_FIFO_LEVEL_GFX0 0 7
# 	REG_FIFO_LEVEL_GFX1 8 15
# 	REG_FIFO_LEVEL_COMPUTE 16 23
regSQ_DSM_CNTL = 0x10a6
# 	WAVEFRONT_STALL_0 0 0
# 	WAVEFRONT_STALL_1 1 1
# 	SPI_BACKPRESSURE_0 2 2
# 	SPI_BACKPRESSURE_1 3 3
# 	SEL_DSM_SGPR_IRRITATOR_DATA0 8 8
# 	SEL_DSM_SGPR_IRRITATOR_DATA1 9 9
# 	SGPR_ENABLE_SINGLE_WRITE 10 10
# 	SEL_DSM_LDS_IRRITATOR_DATA0 16 16
# 	SEL_DSM_LDS_IRRITATOR_DATA1 17 17
# 	LDS_ENABLE_SINGLE_WRITE01 18 18
# 	SEL_DSM_LDS_IRRITATOR_DATA2 19 19
# 	SEL_DSM_LDS_IRRITATOR_DATA3 20 20
# 	LDS_ENABLE_SINGLE_WRITE23 21 21
# 	SEL_DSM_SP_IRRITATOR_DATA0 24 24
# 	SEL_DSM_SP_IRRITATOR_DATA1 25 25
# 	SP_ENABLE_SINGLE_WRITE 26 26
regSQ_DSM_CNTL2 = 0x10a7
# 	SGPR_ENABLE_ERROR_INJECT 0 1
# 	SGPR_SELECT_INJECT_DELAY 2 2
# 	LDS_D_ENABLE_ERROR_INJECT 3 4
# 	LDS_D_SELECT_INJECT_DELAY 5 5
# 	LDS_I_ENABLE_ERROR_INJECT 6 7
# 	LDS_I_SELECT_INJECT_DELAY 8 8
# 	SP_ENABLE_ERROR_INJECT 9 10
# 	SP_SELECT_INJECT_DELAY 11 11
# 	LDS_INJECT_DELAY 14 19
# 	SP_INJECT_DELAY 20 25
# 	SQ_INJECT_DELAY 26 31
regSQ_FIFO_SIZES = 0x10a5
# 	INTERRUPT_FIFO_SIZE 0 3
# 	TTRACE_FIFO_SIZE 8 9
# 	EXPORT_BUF_GS_RESERVED 12 13
# 	EXPORT_BUF_PS_RESERVED 14 15
# 	EXPORT_BUF_REDUCE 16 17
# 	VMEM_DATA_FIFO_SIZE 18 19
# 	EXPORT_BUF_PRIMPOS_LIMIT 20 21
regSQ_IND_DATA = 0x1119
# 	DATA 0 31
regSQ_IND_INDEX = 0x1118
# 	WAVE_ID 0 4
# 	WORKITEM_ID 5 10
# 	AUTO_INCR 11 11
# 	INDEX 16 31
regSQ_INTERRUPT_AUTO_MASK = 0x10be
# 	MASK 0 23
regSQ_INTERRUPT_MSG_CTRL = 0x10bf
# 	STALL 0 0
regSQ_LDS_CLK_CTRL = 0x5090
# 	FORCE_WGP_ON_SA0 0 15
# 	FORCE_WGP_ON_SA1 16 31
regSQ_PERFCOUNTER0_LO = 0x31c0
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER0_SELECT = 0x39c0
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER10_SELECT = 0x39ca
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER11_SELECT = 0x39cb
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER12_SELECT = 0x39cc
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER13_SELECT = 0x39cd
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER14_SELECT = 0x39ce
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER15_SELECT = 0x39cf
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER1_LO = 0x31c2
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER1_SELECT = 0x39c1
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER2_LO = 0x31c4
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER2_SELECT = 0x39c2
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER3_LO = 0x31c6
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER3_SELECT = 0x39c3
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER4_LO = 0x31c8
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER4_SELECT = 0x39c4
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER5_LO = 0x31ca
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER5_SELECT = 0x39c5
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER6_LO = 0x31cc
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER6_SELECT = 0x39c6
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER7_LO = 0x31ce
# 	PERFCOUNTER_LO 0 31
regSQ_PERFCOUNTER7_SELECT = 0x39c7
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER8_SELECT = 0x39c8
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER9_SELECT = 0x39c9
# 	PERF_SEL 0 8
# 	SPM_MODE 20 23
# 	PERF_MODE 28 31
regSQ_PERFCOUNTER_CTRL = 0x39e0
# 	PS_EN 0 0
# 	GS_EN 2 2
# 	HS_EN 4 4
# 	CS_EN 6 6
# 	DISABLE_ME0PIPE0_PERF 14 14
# 	DISABLE_ME0PIPE1_PERF 15 15
# 	DISABLE_ME1PIPE0_PERF 16 16
# 	DISABLE_ME1PIPE1_PERF 17 17
# 	DISABLE_ME1PIPE2_PERF 18 18
# 	DISABLE_ME1PIPE3_PERF 19 19
regSQ_PERFCOUNTER_CTRL2 = 0x39e2
# 	FORCE_EN 0 0
# 	VMID_EN 1 16
regSQ_PERF_SNAPSHOT_CTRL = 0x10bb
# 	TIMER_ON_OFF 0 0
# 	VMID_MASK 1 16
# 	COUNT_SEL 17 17
# 	COUNT_INTERVAL 18 21
regSQ_RANDOM_WAVE_PRI = 0x10a3
# 	RET 0 6
# 	RUI 7 9
# 	RNG 10 23
# 	FORCE_IB_ARB_PRIO_MSK_VALID 31 31
regSQ_RUNTIME_CONFIG = 0x9e0
# 	UNUSED_REGISTER 0 0
regSQ_SHADER_TBA_HI = 0x9e7
# 	ADDR_HI 0 7
# 	TRAP_EN 31 31
regSQ_SHADER_TBA_LO = 0x9e6
# 	ADDR_LO 0 31
regSQ_SHADER_TMA_HI = 0x9e9
# 	ADDR_HI 0 7
regSQ_SHADER_TMA_LO = 0x9e8
# 	ADDR_LO 0 31
regSQ_TEX_CLK_CTRL = 0x508f
# 	FORCE_WGP_ON_SA0 0 15
# 	FORCE_WGP_ON_SA1 16 31
regSQ_THREAD_TRACE_BUF0_BASE = 0x39e8
# 	BASE_LO 0 31
regSQ_THREAD_TRACE_BUF0_SIZE = 0x39e9
# 	BASE_HI 0 3
# 	SIZE 8 29
regSQ_THREAD_TRACE_BUF1_BASE = 0x39ea
# 	BASE_LO 0 31
regSQ_THREAD_TRACE_BUF1_SIZE = 0x39eb
# 	BASE_HI 0 3
# 	SIZE 8 29
regSQ_THREAD_TRACE_CTRL = 0x39ec
# 	MODE 0 1
# 	ALL_VMID 2 2
# 	GL1_PERF_EN 3 3
# 	INTERRUPT_EN 4 4
# 	DOUBLE_BUFFER 5 5
# 	HIWATER 6 8
# 	REG_AT_HWM 9 10
# 	SPI_STALL_EN 11 11
# 	SQ_STALL_EN 12 12
# 	UTIL_TIMER 13 13
# 	WAVESTART_MODE 14 15
# 	RT_FREQ 16 17
# 	SYNC_COUNT_MARKERS 18 18
# 	SYNC_COUNT_DRAWS 19 19
# 	LOWATER_OFFSET 20 22
# 	AUTO_FLUSH_PADDING_DIS 28 28
# 	AUTO_FLUSH_MODE 29 29
# 	DRAW_EVENT_EN 31 31
regSQ_THREAD_TRACE_DROPPED_CNTR = 0x39fa
# 	CNTR 0 31
regSQ_THREAD_TRACE_GFX_DRAW_CNTR = 0x39f6
# 	CNTR 0 31
regSQ_THREAD_TRACE_GFX_MARKER_CNTR = 0x39f7
# 	CNTR 0 31
regSQ_THREAD_TRACE_HP3D_DRAW_CNTR = 0x39f8
# 	CNTR 0 31
regSQ_THREAD_TRACE_HP3D_MARKER_CNTR = 0x39f9
# 	CNTR 0 31
regSQ_THREAD_TRACE_MASK = 0x39ed
# 	SIMD_SEL 0 1
# 	WGP_SEL 4 7
# 	SA_SEL 9 9
# 	WTYPE_INCLUDE 10 16
# 	EXCLUDE_NONDETAIL_SHADERDATA 17 17
regSQ_THREAD_TRACE_STATUS = 0x39f4
# 	FINISH_PENDING 0 11
# 	FINISH_DONE 12 23
# 	WRITE_ERROR 24 24
# 	BUSY 25 25
# 	OWNER_VMID 28 31
regSQ_THREAD_TRACE_STATUS2 = 0x39f5
# 	BUF0_FULL 0 0
# 	BUF1_FULL 1 1
# 	PACKET_LOST_BUF_NO_LOCKDOWN 4 4
# 	BUF_ISSUE_STATUS 8 12
# 	BUF_ISSUE 13 13
# 	WRITE_BUF_FULL 14 14
regSQ_THREAD_TRACE_TOKEN_MASK = 0x39ee
# 	TOKEN_EXCLUDE 0 10
# 	TTRACE_EXEC 11 11
# 	BOP_EVENTS_TOKEN_INCLUDE 12 12
# 	REG_INCLUDE 16 23
# 	INST_EXCLUDE 24 25
# 	REG_EXCLUDE 26 28
# 	REG_DETAIL_ALL 31 31
regSQ_THREAD_TRACE_USERDATA_0 = 0x2340
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_1 = 0x2341
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_2 = 0x2342
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_3 = 0x2343
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_4 = 0x2344
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_5 = 0x2345
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_6 = 0x2346
# 	DATA 0 31
regSQ_THREAD_TRACE_USERDATA_7 = 0x2347
# 	DATA 0 31
regSQ_THREAD_TRACE_WPTR = 0x39ef
# 	OFFSET 0 28
# 	BUFFER_ID 31 31
regSQ_WATCH0_ADDR_H = 0x10d0
# 	ADDR 0 15
regSQ_WATCH0_ADDR_L = 0x10d1
# 	ADDR 6 31
regSQ_WATCH0_CNTL = 0x10d2
# 	MASK 0 23
# 	VMID 24 27
# 	VALID 31 31
regSQ_WATCH1_ADDR_H = 0x10d3
# 	ADDR 0 15
regSQ_WATCH1_ADDR_L = 0x10d4
# 	ADDR 6 31
regSQ_WATCH1_CNTL = 0x10d5
# 	MASK 0 23
# 	VMID 24 27
# 	VALID 31 31
regSQ_WATCH2_ADDR_H = 0x10d6
# 	ADDR 0 15
regSQ_WATCH2_ADDR_L = 0x10d7
# 	ADDR 6 31
regSQ_WATCH2_CNTL = 0x10d8
# 	MASK 0 23
# 	VMID 24 27
# 	VALID 31 31
regSQ_WATCH3_ADDR_H = 0x10d9
# 	ADDR 0 15
regSQ_WATCH3_ADDR_L = 0x10da
# 	ADDR 6 31
regSQ_WATCH3_CNTL = 0x10db
# 	MASK 0 23
# 	VMID 24 27
# 	VALID 31 31
regSX_BLEND_OPT_CONTROL = 0x1d7
# 	MRT0_COLOR_OPT_DISABLE 0 0
# 	MRT0_ALPHA_OPT_DISABLE 1 1
# 	MRT1_COLOR_OPT_DISABLE 4 4
# 	MRT1_ALPHA_OPT_DISABLE 5 5
# 	MRT2_COLOR_OPT_DISABLE 8 8
# 	MRT2_ALPHA_OPT_DISABLE 9 9
# 	MRT3_COLOR_OPT_DISABLE 12 12
# 	MRT3_ALPHA_OPT_DISABLE 13 13
# 	MRT4_COLOR_OPT_DISABLE 16 16
# 	MRT4_ALPHA_OPT_DISABLE 17 17
# 	MRT5_COLOR_OPT_DISABLE 20 20
# 	MRT5_ALPHA_OPT_DISABLE 21 21
# 	MRT6_COLOR_OPT_DISABLE 24 24
# 	MRT6_ALPHA_OPT_DISABLE 25 25
# 	MRT7_COLOR_OPT_DISABLE 28 28
# 	MRT7_ALPHA_OPT_DISABLE 29 29
# 	PIXEN_ZERO_OPT_DISABLE 31 31
regSX_BLEND_OPT_EPSILON = 0x1d6
# 	MRT0_EPSILON 0 3
# 	MRT1_EPSILON 4 7
# 	MRT2_EPSILON 8 11
# 	MRT3_EPSILON 12 15
# 	MRT4_EPSILON 16 19
# 	MRT5_EPSILON 20 23
# 	MRT6_EPSILON 24 27
# 	MRT7_EPSILON 28 31
regSX_DEBUG_1 = 0x11b8
# 	SX_DB_QUAD_CREDIT 0 6
# 	ENABLE_FIFO_DEBUG_WRITE 7 7
# 	DISABLE_BLEND_OPT_DONT_RD_DST 8 8
# 	DISABLE_BLEND_OPT_BYPASS 9 9
# 	DISABLE_BLEND_OPT_DISCARD_PIXEL 10 10
# 	DISABLE_QUAD_PAIR_OPT 11 11
# 	DISABLE_PIX_EN_ZERO_OPT 12 12
# 	DISABLE_REP_FGCG 13 13
# 	ENABLE_SAME_PC_GDS_CGTS 14 14
# 	DISABLE_RAM_FGCG 15 15
# 	PC_DISABLE_SAME_ADDR_OPT 16 16
# 	DISABLE_COL_VAL_READ_OPT 17 17
# 	DISABLE_BC_RB_PLUS 18 18
# 	DISABLE_NATIVE_DOWNCVT_FMT_MAPPING 19 19
# 	DISABLE_SCBD_READ_PWR_OPT 20 20
# 	DISABLE_GDS_CGTS_OPT 21 21
# 	DISABLE_DOWNCVT_PWR_OPT 22 22
# 	DISABLE_POS_BUFF_REUSE_OPT 23 23
# 	DEBUG_DATA 24 31
regSX_MRT0_BLEND_OPT = 0x1d8
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT1_BLEND_OPT = 0x1d9
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT2_BLEND_OPT = 0x1da
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT3_BLEND_OPT = 0x1db
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT4_BLEND_OPT = 0x1dc
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT5_BLEND_OPT = 0x1dd
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT6_BLEND_OPT = 0x1de
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_MRT7_BLEND_OPT = 0x1df
# 	COLOR_SRC_OPT 0 2
# 	COLOR_DST_OPT 4 6
# 	COLOR_COMB_FCN 8 10
# 	ALPHA_SRC_OPT 16 18
# 	ALPHA_DST_OPT 20 22
# 	ALPHA_COMB_FCN 24 26
regSX_PERFCOUNTER0_HI = 0x3241
# 	PERFCOUNTER_HI 0 31
regSX_PERFCOUNTER0_LO = 0x3240
# 	PERFCOUNTER_LO 0 31
regSX_PERFCOUNTER0_SELECT = 0x3a40
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSX_PERFCOUNTER0_SELECT1 = 0x3a44
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSX_PERFCOUNTER1_HI = 0x3243
# 	PERFCOUNTER_HI 0 31
regSX_PERFCOUNTER1_LO = 0x3242
# 	PERFCOUNTER_LO 0 31
regSX_PERFCOUNTER1_SELECT = 0x3a41
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regSX_PERFCOUNTER1_SELECT1 = 0x3a45
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regSX_PERFCOUNTER2_HI = 0x3245
# 	PERFCOUNTER_HI 0 31
regSX_PERFCOUNTER2_LO = 0x3244
# 	PERFCOUNTER_LO 0 31
regSX_PERFCOUNTER2_SELECT = 0x3a42
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regSX_PERFCOUNTER3_HI = 0x3247
# 	PERFCOUNTER_HI 0 31
regSX_PERFCOUNTER3_LO = 0x3246
# 	PERFCOUNTER_LO 0 31
regSX_PERFCOUNTER3_SELECT = 0x3a43
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regSX_PS_DOWNCONVERT = 0x1d5
# 	MRT0 0 3
# 	MRT1 4 7
# 	MRT2 8 11
# 	MRT3 12 15
# 	MRT4 16 19
# 	MRT5 20 23
# 	MRT6 24 27
# 	MRT7 28 31
regSX_PS_DOWNCONVERT_CONTROL = 0x1d4
# 	MRT0_FMT_MAPPING_DISABLE 0 0
# 	MRT1_FMT_MAPPING_DISABLE 1 1
# 	MRT2_FMT_MAPPING_DISABLE 2 2
# 	MRT3_FMT_MAPPING_DISABLE 3 3
# 	MRT4_FMT_MAPPING_DISABLE 4 4
# 	MRT5_FMT_MAPPING_DISABLE 5 5
# 	MRT6_FMT_MAPPING_DISABLE 6 6
# 	MRT7_FMT_MAPPING_DISABLE 7 7
regTA_BC_BASE_ADDR = 0x20
# 	ADDRESS 0 31
regTA_BC_BASE_ADDR_HI = 0x21
# 	ADDRESS 0 7
regTA_CGTT_CTRL = 0x509d
# 	ON_DELAY 0 3
# 	OFF_HYSTERESIS 4 11
# 	SOFT_STALL_OVERRIDE7 16 16
# 	SOFT_STALL_OVERRIDE6 17 17
# 	SOFT_STALL_OVERRIDE5 18 18
# 	SOFT_STALL_OVERRIDE4 19 19
# 	SOFT_STALL_OVERRIDE3 20 20
# 	SOFT_STALL_OVERRIDE2 21 21
# 	SOFT_STALL_OVERRIDE1 22 22
# 	SOFT_STALL_OVERRIDE0 23 23
# 	SOFT_OVERRIDE7 24 24
# 	SOFT_OVERRIDE6 25 25
# 	SOFT_OVERRIDE5 26 26
# 	SOFT_OVERRIDE4 27 27
# 	SOFT_OVERRIDE3 28 28
# 	SOFT_OVERRIDE2 29 29
# 	SOFT_OVERRIDE1 30 30
# 	SOFT_OVERRIDE0 31 31
regTA_CNTL = 0x12e1
# 	TA_SQ_XNACK_FGCG_DISABLE 0 0
# 	ALIGNER_CREDIT 16 20
# 	TD_FIFO_CREDIT 22 31
regTA_CNTL2 = 0x12e5
# 	POINT_SAMPLE_ACCEL_DIS 16 16
# 	TRUNCATE_COORD_MODE 18 18
# 	ELIMINATE_UNLIT_QUAD_DIS 19 19
regTA_CNTL_AUX = 0x12e2
# 	SCOAL_DSWIZZLE_N 0 0
# 	DEPTH_AS_PITCH_DIS 1 1
# 	CORNER_SAMPLES_MIN_DIM 2 2
# 	OVERRIDE_QUAD_MODE_DIS 3 3
# 	DERIV_ADJUST_DIS 4 4
# 	TFAULT_EN_OVERRIDE 5 5
# 	GATHERH_DST_SEL 6 6
# 	DISABLE_GATHER4_BC_SWIZZLE 7 7
# 	ANISO_MAG_STEP_CLAMP 8 8
# 	AUTO_ALIGN_FORMAT 9 9
# 	ANISO_HALF_THRESH 10 11
# 	ANISO_ERROR_FP_VBIAS 12 12
# 	ANISO_STEP_ORDER 13 13
# 	ANISO_STEP 14 14
# 	MINMAG_UNNORM 15 15
# 	ANISO_WEIGHT_MODE 16 16
# 	ANISO_RATIO_LUT 17 17
# 	ANISO_TAP 18 18
# 	DETERMINISM_RESERVED_DISABLE 20 20
# 	DETERMINISM_OPCODE_STRICT_DISABLE 21 21
# 	DETERMINISM_MISC_DISABLE 22 22
# 	DETERMINISM_SAMPLE_C_DFMT_DISABLE 23 23
# 	DETERMINISM_SAMPLER_MSAA_DISABLE 24 24
# 	DETERMINISM_WRITEOP_READFMT_DISABLE 25 25
# 	DETERMINISM_DFMT_NFMT_DISABLE 26 26
# 	CUBEMAP_SLICE_CLAMP 28 28
# 	TRUNC_SMALL_NEG 29 29
# 	ARRAY_ROUND_MODE 30 31
regTA_CS_BC_BASE_ADDR = 0x2380
# 	ADDRESS 0 31
regTA_CS_BC_BASE_ADDR_HI = 0x2381
# 	ADDRESS 0 7
regTA_PERFCOUNTER0_HI = 0x32c1
# 	PERFCOUNTER_HI 0 31
regTA_PERFCOUNTER0_LO = 0x32c0
# 	PERFCOUNTER_LO 0 31
regTA_PERFCOUNTER0_SELECT = 0x3ac0
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regTA_PERFCOUNTER0_SELECT1 = 0x3ac1
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regTA_PERFCOUNTER1_HI = 0x32c3
# 	PERFCOUNTER_HI 0 31
regTA_PERFCOUNTER1_LO = 0x32c2
# 	PERFCOUNTER_LO 0 31
regTA_PERFCOUNTER1_SELECT = 0x3ac2
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regTA_SCRATCH = 0x1304
# 	SCRATCH 0 31
regTA_STATUS = 0x12e8
# 	FG_PFIFO_EMPTYB 12 12
# 	FG_LFIFO_EMPTYB 13 13
# 	FG_SFIFO_EMPTYB 14 14
# 	FL_PFIFO_EMPTYB 16 16
# 	FL_LFIFO_EMPTYB 17 17
# 	FL_SFIFO_EMPTYB 18 18
# 	FA_PFIFO_EMPTYB 20 20
# 	FA_LFIFO_EMPTYB 21 21
# 	FA_SFIFO_EMPTYB 22 22
# 	IN_BUSY 24 24
# 	FG_BUSY 25 25
# 	LA_BUSY 26 26
# 	FL_BUSY 27 27
# 	TA_BUSY 28 28
# 	FA_BUSY 29 29
# 	AL_BUSY 30 30
# 	BUSY 31 31
regTCP_CNTL = 0x19a2
regTCP_CNTL2 = 0x19a3
# 	LS_DISABLE_CLOCKS 0 7
# 	TCP_FMT_MGCG_DISABLE 8 8
# 	TCPF_LATENCY_BYPASS_DISABLE 9 9
# 	TCP_WRITE_DATA_MGCG_DISABLE 10 10
# 	TCP_INNER_BLOCK_MGCG_DISABLE 11 11
# 	TCP_ADRS_IMG_CALC_MGCG_DISABLE 12 12
# 	V64_COMBINE_ENABLE 13 13
# 	TAGRAM_ADDR_SWIZZLE_DISABLE 14 14
# 	RETURN_ORDER_OVERRIDE 15 15
# 	POWER_OPT_DISABLE 16 16
# 	GCR_RSP_FGCG_DISABLE 17 17
# 	PERF_EN_OVERRIDE 18 19
# 	TC_TD_RAM_CLKEN_DISABLE 20 20
# 	TC_TD_DATA_CLKEN_DISABLE 21 21
# 	TCP_GL1_REQ_CLKEN_DISABLE 22 22
# 	TCP_GL1R_SRC_CLKEN_DISABLE 23 23
# 	SPARE_BIT 26 26
# 	TAGRAM_XY_BIAS_OVERRIDE 27 28
# 	TCP_REQ_MGCG_DISABLE 29 29
# 	TCP_MISS_MGCG_DISABLE 30 30
# 	DISABLE_MIPMAP_PARAM_CALC_SELF_GATING 31 31
regTCP_DEBUG_DATA = 0x19a6
# 	DATA 0 17
regTCP_DEBUG_INDEX = 0x19a5
# 	INDEX 0 4
regTCP_INVALIDATE = 0x19a0
# 	START 0 0
regTCP_PERFCOUNTER0_HI = 0x3341
# 	PERFCOUNTER_HI 0 31
regTCP_PERFCOUNTER0_LO = 0x3340
# 	PERFCOUNTER_LO 0 31
regTCP_PERFCOUNTER0_SELECT = 0x3b40
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regTCP_PERFCOUNTER0_SELECT1 = 0x3b41
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regTCP_PERFCOUNTER1_HI = 0x3343
# 	PERFCOUNTER_HI 0 31
regTCP_PERFCOUNTER1_LO = 0x3342
# 	PERFCOUNTER_LO 0 31
regTCP_PERFCOUNTER1_SELECT = 0x3b42
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regTCP_PERFCOUNTER1_SELECT1 = 0x3b43
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regTCP_PERFCOUNTER2_HI = 0x3345
# 	PERFCOUNTER_HI 0 31
regTCP_PERFCOUNTER2_LO = 0x3344
# 	PERFCOUNTER_LO 0 31
regTCP_PERFCOUNTER2_SELECT = 0x3b44
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regTCP_PERFCOUNTER3_HI = 0x3347
# 	PERFCOUNTER_HI 0 31
regTCP_PERFCOUNTER3_LO = 0x3346
# 	PERFCOUNTER_LO 0 31
regTCP_PERFCOUNTER3_SELECT = 0x3b45
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regTCP_PERFCOUNTER_FILTER = 0x3348
# 	BUFFER 0 0
# 	FLAT 1 1
# 	DIM 2 4
# 	DATA_FORMAT 5 11
# 	NUM_FORMAT 13 16
# 	SW_MODE 17 21
# 	NUM_SAMPLES 22 23
# 	OPCODE_TYPE 24 26
# 	SLC 27 27
# 	DLC 28 28
# 	GLC 29 29
# 	COMPRESSION_ENABLE 30 30
regTCP_PERFCOUNTER_FILTER2 = 0x3349
# 	REQ_MODE 0 2
regTCP_PERFCOUNTER_FILTER_EN = 0x334a
# 	BUFFER 0 0
# 	FLAT 1 1
# 	DIM 2 2
# 	DATA_FORMAT 3 3
# 	NUM_FORMAT 4 4
# 	SW_MODE 5 5
# 	NUM_SAMPLES 6 6
# 	OPCODE_TYPE 7 7
# 	SLC 8 8
# 	DLC 9 9
# 	GLC 10 10
# 	COMPRESSION_ENABLE 11 11
# 	REQ_MODE 12 12
regTCP_STATUS = 0x19a1
# 	TCP_BUSY 0 0
# 	INPUT_BUSY 1 1
# 	ADRS_BUSY 2 2
# 	TAGRAMS_BUSY 3 3
# 	CNTRL_BUSY 4 4
# 	LFIFO_BUSY 5 5
# 	READ_BUSY 6 6
# 	FORMAT_BUSY 7 7
# 	VM_BUSY 8 8
# 	MEMIF_BUSY 9 9
# 	GCR_BUSY 10 10
# 	OFIFO_BUSY 11 11
# 	OFIFO_QUEUE_BUSY 12 13
# 	XNACK_PRT 15 15
regTCP_WATCH0_ADDR_H = 0x2048
# 	ADDR 0 15
regTCP_WATCH0_ADDR_L = 0x2049
# 	ADDR 7 31
regTCP_WATCH0_CNTL = 0x204a
# 	MASK 0 22
# 	VMID 24 27
# 	MODE 29 30
# 	VALID 31 31
regTCP_WATCH1_ADDR_H = 0x204b
# 	ADDR 0 15
regTCP_WATCH1_ADDR_L = 0x204c
# 	ADDR 7 31
regTCP_WATCH1_CNTL = 0x204d
# 	MASK 0 22
# 	VMID 24 27
# 	MODE 29 30
# 	VALID 31 31
regTCP_WATCH2_ADDR_H = 0x204e
# 	ADDR 0 15
regTCP_WATCH2_ADDR_L = 0x204f
# 	ADDR 7 31
regTCP_WATCH2_CNTL = 0x2050
# 	MASK 0 22
# 	VMID 24 27
# 	MODE 29 30
# 	VALID 31 31
regTCP_WATCH3_ADDR_H = 0x2051
# 	ADDR 0 15
regTCP_WATCH3_ADDR_L = 0x2052
# 	ADDR 7 31
regTCP_WATCH3_CNTL = 0x2053
# 	MASK 0 22
# 	VMID 24 27
# 	MODE 29 30
# 	VALID 31 31
regTD_DSM_CNTL = 0x12cf
regTD_DSM_CNTL2 = 0x12d0
regTD_PERFCOUNTER0_HI = 0x3301
# 	PERFCOUNTER_HI 0 31
regTD_PERFCOUNTER0_LO = 0x3300
# 	PERFCOUNTER_LO 0 31
regTD_PERFCOUNTER0_SELECT = 0x3b00
# 	PERF_SEL 0 9
# 	PERF_SEL1 10 19
# 	CNTR_MODE 20 23
# 	PERF_MODE1 24 27
# 	PERF_MODE 28 31
regTD_PERFCOUNTER0_SELECT1 = 0x3b01
# 	PERF_SEL2 0 9
# 	PERF_SEL3 10 19
# 	PERF_MODE3 24 27
# 	PERF_MODE2 28 31
regTD_PERFCOUNTER1_HI = 0x3303
# 	PERFCOUNTER_HI 0 31
regTD_PERFCOUNTER1_LO = 0x3302
# 	PERFCOUNTER_LO 0 31
regTD_PERFCOUNTER1_SELECT = 0x3b02
# 	PERF_SEL 0 9
# 	CNTR_MODE 20 23
# 	PERF_MODE 28 31
regTD_SCRATCH = 0x12d3
# 	SCRATCH 0 31
regTD_STATUS = 0x12c6
# 	BUSY 31 31
regUCONFIG_RESERVED_REG0 = 0x20a2
# 	DATA 0 31
regUCONFIG_RESERVED_REG1 = 0x20a3
# 	DATA 0 31
regUTCL1_ALOG = 0x158f
# 	UTCL1_ALOG_MODE1_FILTER1_THRESHOLD 0 2
# 	UTCL1_ALOG_MODE1_FILTER2_BYPASS 3 3
# 	UTCL1_ALOG_ACTIVE 4 4
# 	UTCL1_ALOG_MODE 5 5
# 	UTCL1_ALOG_MODE2_LOCK_WINDOW 6 8
# 	UTCL1_ALOG_ONLY_MISS 9 9
# 	UTCL1_ALOG_MODE2_INTR_THRESHOLD 10 11
# 	UTCL1_ALOG_SPACE_EN 12 14
# 	UTCL1_ALOG_CLEAN 15 15
# 	UTCL1_ALOG_IDLE 16 16
# 	UTCL1_ALOG_TRACK_SEGMENT_SIZE 17 22
# 	UTCL1_ALOG_MODE1_FILTER1_BYPASS 23 23
# 	UTCL1_ALOG_MODE1_INTR_ON_ALLOC 24 24
regUTCL1_CTRL_0 = 0x1980
# 	UTCL1_L0_REQ_VFIFO_DISABLE 0 0
# 	UTCL1_UTCL2_INVACK_CDC_FIFO_DISABLE 1 1
# 	RESERVED_0 2 2
# 	UTCL1_UTCL2_REQ_CREDITS 3 8
# 	UTCL1_UTCL0_INVREQ_CREDITS 9 12
# 	UTCL1_LIMIT_INV_TO_ONE 13 13
# 	UTCL1_LIMIT_XLAT_TO_ONE 14 14
# 	UTCL1_UTCL2_FGCG_REPEATERS_OVERRIDE 15 15
# 	UTCL1_INV_FILTER_VMID 16 16
# 	UTCL1_RANGE_INV_FORCE_CHK_ALL 17 17
# 	UTCL1_UTCL0_RET_FGCG_REPEATERS_OVERRIDE 18 18
# 	UTCL1_UTCL0_INVREQ_FGCG_REPEATERS_OVERRIDE 19 19
# 	GCRD_FGCG_DISABLE 20 20
# 	UTCL1_MH_RANGE_INV_TO_VMID_OVERRIDE 21 21
# 	UTCL1_MH_DISABLE_DUPLICATES 22 22
# 	UTCL1_MH_DISABLE_REQUEST_SQUASHING 23 23
# 	UTCL1_MH_DISABLE_RECENT_BUFFER 24 24
# 	UTCL1_XLAT_FAULT_LOCK_CTRL 25 26
# 	UTCL1_REDUCE_CC_SIZE 27 28
# 	RESERVED_1 29 29
# 	MH_SPARE0 30 30
# 	RESERVED_2 31 31
regUTCL1_CTRL_1 = 0x158c
# 	UTCL1_CACHE_CORE_BYPASS 0 0
# 	UTCL1_TCP_BYPASS 1 1
# 	UTCL1_SQCI_BYPASS 2 2
# 	UTCL1_SQCD_BYPASS 3 3
# 	UTCL1_RMI_BYPASS 4 4
# 	UTCL1_SQG_BYPASS 5 5
# 	UTCL1_FORCE_RANGE_INV_TO_VMID 6 6
# 	UTCL1_FORCE_INV_ALL 7 7
# 	UTCL1_FORCE_INV_ALL_DONE 8 8
# 	UTCL1_PAGE_SIZE_1 9 10
# 	UTCL1_PAGE_SIZE_2 11 12
# 	UTCL1_PAGE_SIZE_3 13 14
# 	UTCL1_PAGE_SIZE_4 15 16
# 	RESERVED 17 31
regUTCL1_CTRL_2 = 0x1985
# 	UTCL1_RNG_TO_VMID_INV_OVRD 0 3
# 	UTCL1_PMM_INTERRUPT_CREDITS_OVERRIDE 4 9
# 	UTCL1_CACHE_WRITE_PERM 10 10
# 	UTCL1_PAGE_OVRD_DISABLE 11 11
# 	UTCL1_SPARE0 12 12
# 	UTCL1_SPARE1 13 13
# 	RESERVED 14 31
regUTCL1_FIFO_SIZING = 0x1986
# 	UTCL1_UTCL2_INVACK_CDC_FIFO_THRESH 0 2
# 	UTCL1_GENERAL_SIZING_CTRL_LOW 3 15
# 	UTCL1_GENERAL_SIZING_CTRL_HIGH 16 31
regUTCL1_PERFCOUNTER0_HI = 0x35a1
# 	PERFCOUNTER_HI 0 31
regUTCL1_PERFCOUNTER0_LO = 0x35a0
# 	PERFCOUNTER_LO 0 31
regUTCL1_PERFCOUNTER0_SELECT = 0x3da0
# 	PERF_SEL 0 9
# 	COUNTER_MODE 28 31
regUTCL1_PERFCOUNTER1_HI = 0x35a3
# 	PERFCOUNTER_HI 0 31
regUTCL1_PERFCOUNTER1_LO = 0x35a2
# 	PERFCOUNTER_LO 0 31
regUTCL1_PERFCOUNTER1_SELECT = 0x3da1
# 	PERF_SEL 0 9
# 	COUNTER_MODE 28 31
regUTCL1_PERFCOUNTER2_HI = 0x35a5
# 	PERFCOUNTER_HI 0 31
regUTCL1_PERFCOUNTER2_LO = 0x35a4
# 	PERFCOUNTER_LO 0 31
regUTCL1_PERFCOUNTER2_SELECT = 0x3da2
# 	PERF_SEL 0 9
# 	COUNTER_MODE 28 31
regUTCL1_PERFCOUNTER3_HI = 0x35a7
# 	PERFCOUNTER_HI 0 31
regUTCL1_PERFCOUNTER3_LO = 0x35a6
# 	PERFCOUNTER_LO 0 31
regUTCL1_PERFCOUNTER3_SELECT = 0x3da3
# 	PERF_SEL 0 9
# 	COUNTER_MODE 28 31
regUTCL1_STATUS = 0x1594
# 	UTCL1_HIT_PATH_BUSY 0 0
# 	UTCL1_MH_BUSY 1 1
# 	UTCL1_INV_BUSY 2 2
# 	UTCL1_PENDING_UTCL2_REQ 3 3
# 	UTCL1_PENDING_UTCL2_RET 4 4
# 	UTCL1_LAST_UTCL2_RET_XNACK 5 6
# 	UTCL1_RANGE_INV_IN_PROGRESS 7 7
# 	RESERVED 8 8
regUTCL1_UTCL0_INVREQ_DISABLE = 0x1984
# 	UTCL1_UTCL0_INVREQ_DISABLE 0 31
regVGT_DMA_BASE = 0x1fa
# 	BASE_ADDR 0 31
regVGT_DMA_BASE_HI = 0x1f9
# 	BASE_ADDR 0 15
regVGT_DMA_DATA_FIFO_DEPTH = 0xfcd
# 	DMA_DATA_FIFO_DEPTH 0 9
regVGT_DMA_INDEX_TYPE = 0x29f
# 	INDEX_TYPE 0 1
# 	SWAP_MODE 2 3
# 	BUF_TYPE 4 5
# 	RDREQ_POLICY 6 7
# 	ATC 8 8
# 	NOT_EOP 9 9
# 	REQ_PATH 10 10
# 	MTYPE 11 13
# 	DISABLE_INSTANCE_PACKING 14 14
regVGT_DMA_MAX_SIZE = 0x29e
# 	MAX_SIZE 0 31
regVGT_DMA_NUM_INSTANCES = 0x2a2
# 	NUM_INSTANCES 0 31
regVGT_DMA_REQ_FIFO_DEPTH = 0xfce
# 	DMA_REQ_FIFO_DEPTH 0 5
regVGT_DMA_SIZE = 0x29d
# 	NUM_INDICES 0 31
regVGT_DRAW_INITIATOR = 0x1fc
# 	SOURCE_SELECT 0 1
# 	MAJOR_MODE 2 3
# 	SPRITE_EN_R6XX 4 4
# 	NOT_EOP 5 5
# 	USE_OPAQUE 6 6
# 	REG_RT_INDEX 29 31
regVGT_DRAW_INIT_FIFO_DEPTH = 0xfcf
# 	DRAW_INIT_FIFO_DEPTH 0 5
regVGT_DRAW_PAYLOAD_CNTL = 0x2a6
# 	EN_REG_RT_INDEX 1 1
# 	EN_PRIM_PAYLOAD 3 3
# 	EN_DRAW_VP 4 4
regVGT_ENHANCE = 0x294
# 	MISC 0 31
regVGT_ESGS_RING_ITEMSIZE = 0x2ab
# 	ITEMSIZE 0 14
regVGT_EVENT_ADDRESS_REG = 0x1fe
# 	ADDRESS_LOW 0 27
regVGT_EVENT_INITIATOR = 0x2a4
# 	EVENT_TYPE 0 5
# 	ADDRESS_HI 10 26
# 	EXTENDED_EVENT 27 27
regVGT_GS_INSTANCE_CNT = 0x2e4
# 	ENABLE 0 0
# 	CNT 2 8
# 	EN_MAX_VERT_OUT_PER_GS_INSTANCE 31 31
regVGT_GS_MAX_VERT_OUT = 0x2ce
# 	MAX_VERT_OUT 0 10
regVGT_GS_MAX_WAVE_ID = 0x1009
# 	MAX_WAVE_ID 0 11
regVGT_GS_OUT_PRIM_TYPE = 0x2266
# 	OUTPRIM_TYPE 0 5
regVGT_HOS_MAX_TESS_LEVEL = 0x286
# 	MAX_TESS 0 31
regVGT_HOS_MIN_TESS_LEVEL = 0x287
# 	MIN_TESS 0 31
regVGT_HS_OFFCHIP_PARAM = 0x224f
# 	OFFCHIP_BUFFERING 0 9
# 	OFFCHIP_GRANULARITY 10 11
regVGT_INDEX_TYPE = 0x2243
# 	INDEX_TYPE 0 1
# 	DISABLE_INSTANCE_PACKING 14 14
regVGT_INSTANCE_BASE_ID = 0x225a
# 	INSTANCE_BASE_ID 0 31
regVGT_LS_HS_CONFIG = 0x2d6
# 	NUM_PATCHES 0 7
# 	HS_NUM_INPUT_CP 8 13
# 	HS_NUM_OUTPUT_CP 14 19
regVGT_MC_LAT_CNTL = 0xfd6
# 	MC_TIME_STAMP_RES 0 3
regVGT_MULTI_PRIM_IB_RESET_INDX = 0x103
# 	RESET_INDX 0 31
regVGT_NUM_INDICES = 0x224c
# 	NUM_INDICES 0 31
regVGT_NUM_INSTANCES = 0x224d
# 	NUM_INSTANCES 0 31
regVGT_PRIMITIVEID_EN = 0x2a1
# 	PRIMITIVEID_EN 0 0
# 	DISABLE_RESET_ON_EOI 1 1
# 	NGG_DISABLE_PROVOK_REUSE 2 2
regVGT_PRIMITIVEID_RESET = 0x2a3
# 	VALUE 0 31
regVGT_PRIMITIVE_TYPE = 0x2242
# 	PRIM_TYPE 0 5
regVGT_REUSE_OFF = 0x2ad
# 	REUSE_OFF 0 0
regVGT_SHADER_STAGES_EN = 0x2d5
# 	LS_EN 0 1
# 	HS_EN 2 2
# 	ES_EN 3 4
# 	GS_EN 5 5
# 	VS_EN 6 7
# 	DYNAMIC_HS 8 8
# 	VS_WAVE_ID_EN 12 12
# 	PRIMGEN_EN 13 13
# 	ORDERED_ID_MODE 14 14
# 	MAX_PRIMGRP_IN_WAVE 15 18
# 	GS_FAST_LAUNCH 19 20
# 	HS_W32_EN 21 21
# 	GS_W32_EN 22 22
# 	VS_W32_EN 23 23
# 	NGG_WAVE_ID_EN 24 24
# 	PRIMGEN_PASSTHRU_EN 25 25
# 	PRIMGEN_PASSTHRU_NO_MSG 26 26
regVGT_STRMOUT_DRAW_OPAQUE_BUFFER_FILLED_SIZE = 0x2cb
# 	SIZE 0 31
regVGT_STRMOUT_DRAW_OPAQUE_OFFSET = 0x2ca
# 	OFFSET 0 31
regVGT_STRMOUT_DRAW_OPAQUE_VERTEX_STRIDE = 0x2cc
# 	VERTEX_STRIDE 0 8
regVGT_SYS_CONFIG = 0x1003
# 	DUAL_CORE_EN 0 0
# 	MAX_LS_HS_THDGRP 1 6
# 	ADC_EVENT_FILTER_DISABLE 7 7
# 	NUM_SUBGROUPS_IN_FLIGHT 8 18
regVGT_TESS_DISTRIBUTION = 0x2d4
# 	ACCUM_ISOLINE 0 7
# 	ACCUM_TRI 8 15
# 	ACCUM_QUAD 16 23
# 	DONUT_SPLIT 24 28
# 	TRAP_SPLIT 29 31
regVGT_TF_MEMORY_BASE = 0x2250
# 	BASE 0 31
regVGT_TF_MEMORY_BASE_HI = 0x2261
# 	BASE_HI 0 7
regVGT_TF_PARAM = 0x2db
# 	TYPE 0 1
# 	PARTITIONING 2 4
# 	TOPOLOGY 5 7
# 	NOT_USED 9 9
# 	NUM_DS_WAVES_PER_SIMD 10 13
# 	DISABLE_DONUTS 14 14
# 	RDREQ_POLICY 15 16
# 	DISTRIBUTION_MODE 17 18
# 	DETECT_ONE 19 19
# 	DETECT_ZERO 20 20
# 	MTYPE 23 25
regVGT_TF_RING_SIZE = 0x224e
# 	SIZE 0 16
regVIOLATION_DATA_ASYNC_VF_PROG = 0xdf1
# 	SSRCID 0 3
# 	VFID 4 9
# 	VIOLATION_ERROR 31 31
regWD_CNTL_STATUS = 0xfdf
# 	DIST_BUSY 0 0
# 	DIST_BE_BUSY 1 1
# 	GE_UTCL1_BUSY 2 2
# 	WD_TE11_BUSY 3 3
# 	PC_MANAGER_BUSY 4 4
# 	WLC_BUSY 5 5
regWD_ENHANCE = 0x2a0
# 	MISC 0 31
regWD_QOS = 0xfe2
# 	DRAW_STALL 0 0
regWD_UTCL1_CNTL = 0xfe3
# 	XNACK_REDO_TIMER_CNT 0 19
# 	VMID_RESET_MODE 23 23
# 	DROP_MODE 24 24
# 	BYPASS 25 25
# 	INVALIDATE 26 26
# 	FRAG_LIMIT_MODE 27 27
# 	FORCE_SNOOP 28 28
# 	MTYPE_OVERRIDE 29 29
# 	LLC_NOALLOC_OVERRIDE 30 30
regWD_UTCL1_STATUS = 0xfe4
# 	FAULT_DETECTED 0 0
# 	RETRY_DETECTED 1 1
# 	PRT_DETECTED 2 2
# 	FAULT_UTCL1ID 8 13
# 	RETRY_UTCL1ID 16 21
# 	PRT_UTCL1ID 24 29