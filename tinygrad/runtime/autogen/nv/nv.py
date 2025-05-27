# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-include', '/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/nvtypes.h', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/nvidia-uvm', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/unix/include', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/ctrl']
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





KERN_FSP_COT_PAYLOAD_H = True # macro
class struct_c__SA_MCTP_HEADER(Structure):
    pass

struct_c__SA_MCTP_HEADER._pack_ = 1 # source:False
struct_c__SA_MCTP_HEADER._fields_ = [
    ('constBlob', ctypes.c_uint32),
    ('msgType', ctypes.c_ubyte),
    ('vendorId', ctypes.c_uint16),
]

MCTP_HEADER = struct_c__SA_MCTP_HEADER
class struct_c__SA_NVDM_PAYLOAD_COT(Structure):
    pass

struct_c__SA_NVDM_PAYLOAD_COT._pack_ = 1 # source:False
struct_c__SA_NVDM_PAYLOAD_COT._fields_ = [
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('gspFmcSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemSize', ctypes.c_uint32),
    ('frtsVidmemOffset', ctypes.c_uint64),
    ('frtsVidmemSize', ctypes.c_uint32),
    ('hash384', ctypes.c_uint32 * 12),
    ('publicKey', ctypes.c_uint32 * 96),
    ('signature', ctypes.c_uint32 * 96),
    ('gspBootArgsSysmemOffset', ctypes.c_uint64),
]

NVDM_PAYLOAD_COT = struct_c__SA_NVDM_PAYLOAD_COT
__all__ = \
    ['KERN_FSP_COT_PAYLOAD_H', 'MCTP_HEADER', 'NVDM_PAYLOAD_COT',
    'struct_c__SA_MCTP_HEADER', 'struct_c__SA_NVDM_PAYLOAD_COT']
