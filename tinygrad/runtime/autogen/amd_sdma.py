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
