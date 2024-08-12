# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/usr/local/cuda/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


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





CUdeviceptr_v1 = ctypes.c_uint32
class struct_CUDA_MEMCPY2D_v1_st(Structure):
    pass

struct_CUDA_MEMCPY2D_v1_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY2D_v1_st._fields_ = [
    ('dummy', ctypes.c_int32),
]

CUDA_MEMCPY2D_v1 = struct_CUDA_MEMCPY2D_v1_st
class struct_CUDA_MEMCPY3D_v1_st(Structure):
    pass

struct_CUDA_MEMCPY3D_v1_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY3D_v1_st._fields_ = [
    ('dummy', ctypes.c_int32),
]

CUDA_MEMCPY3D_v1 = struct_CUDA_MEMCPY3D_v1_st
class struct_CUDA_ARRAY_DESCRIPTOR_v1_st(Structure):
    pass

struct_CUDA_ARRAY_DESCRIPTOR_v1_st._pack_ = 1 # source:False
struct_CUDA_ARRAY_DESCRIPTOR_v1_st._fields_ = [
    ('dummy', ctypes.c_int32),
]

CUDA_ARRAY_DESCRIPTOR_v1 = struct_CUDA_ARRAY_DESCRIPTOR_v1_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st(Structure):
    pass

struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st._pack_ = 1 # source:False
struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st._fields_ = [
    ('dummy', ctypes.c_int32),
]

CUDA_ARRAY3D_DESCRIPTOR_v1 = struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st
__all__ = \
    ['CUDA_ARRAY3D_DESCRIPTOR_v1', 'CUDA_ARRAY_DESCRIPTOR_v1',
    'CUDA_MEMCPY2D_v1', 'CUDA_MEMCPY3D_v1', 'CUdeviceptr_v1',
    'struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st',
    'struct_CUDA_ARRAY_DESCRIPTOR_v1_st',
    'struct_CUDA_MEMCPY2D_v1_st', 'struct_CUDA_MEMCPY3D_v1_st']
