# -*- coding: utf-8 -*-
#
# TARGET arch is: []
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





_UAPI_LINUX_ION_H = True # macro
# def ION_HEAP_SYSTEM_MASK((1<<ION_HEAP_TYPE_SYSTEM):  # macro
#    return )  
# def ION_HEAP_SYSTEM_CONTIG_MASK((1<<ION_HEAP_TYPE_SYSTEM_CONTIG):  # macro
#    return )  
# def ION_HEAP_CARVEOUT_MASK((1<<ION_HEAP_TYPE_CARVEOUT):  # macro
#    return )  
# def ION_HEAP_TYPE_DMA_MASK((1<<ION_HEAP_TYPE_DMA):  # macro
#    return )  
# def ION_NUM_HEAP_IDS(sizeof(unsignedint):  # macro
#    return *8)  
ION_FLAG_CACHED = 1 # macro
ION_FLAG_CACHED_NEEDS_SYNC = 2 # macro
ION_IOC_MAGIC = 'I' # macro
# ION_IOC_ALLOC = _IOWR ( 'I' , 0 , struct ion_allocation_data ) # macro
# ION_IOC_FREE = _IOWR ( 'I' , 1 , struct ion_handle_data ) # macro
# ION_IOC_MAP = _IOWR ( 'I' , 2 , struct ion_fd_data ) # macro
# ION_IOC_SHARE = _IOWR ( 'I' , 4 , struct ion_fd_data ) # macro
# ION_IOC_IMPORT = _IOWR ( 'I' , 5 , struct ion_fd_data ) # macro
# ION_IOC_SYNC = _IOWR ( 'I' , 7 , struct ion_fd_data ) # macro
# ION_IOC_CUSTOM = _IOWR ( 'I' , 6 , struct ion_custom_data ) # macro
ion_user_handle_t = ctypes.c_int32

# values for enumeration 'ion_heap_type'
ion_heap_type__enumvalues = {
    0: 'ION_HEAP_TYPE_SYSTEM',
    1: 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    2: 'ION_HEAP_TYPE_CARVEOUT',
    3: 'ION_HEAP_TYPE_CHUNK',
    4: 'ION_HEAP_TYPE_DMA',
    5: 'ION_HEAP_TYPE_CUSTOM',
    16: 'ION_NUM_HEAPS',
}
ION_HEAP_TYPE_SYSTEM = 0
ION_HEAP_TYPE_SYSTEM_CONTIG = 1
ION_HEAP_TYPE_CARVEOUT = 2
ION_HEAP_TYPE_CHUNK = 3
ION_HEAP_TYPE_DMA = 4
ION_HEAP_TYPE_CUSTOM = 5
ION_NUM_HEAPS = 16
ion_heap_type = ctypes.c_uint32 # enum
class struct_ion_allocation_data(Structure):
    pass

struct_ion_allocation_data._pack_ = 1 # source:False
struct_ion_allocation_data._fields_ = [
    ('len', ctypes.c_uint64),
    ('align', ctypes.c_uint64),
    ('heap_id_mask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('handle', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ion_fd_data(Structure):
    pass

struct_ion_fd_data._pack_ = 1 # source:False
struct_ion_fd_data._fields_ = [
    ('handle', ctypes.c_int32),
    ('fd', ctypes.c_int32),
]

class struct_ion_handle_data(Structure):
    pass

struct_ion_handle_data._pack_ = 1 # source:False
struct_ion_handle_data._fields_ = [
    ('handle', ctypes.c_int32),
]

class struct_ion_custom_data(Structure):
    pass

struct_ion_custom_data._pack_ = 1 # source:False
struct_ion_custom_data._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('arg', ctypes.c_uint64),
]

__all__ = \
    ['ION_FLAG_CACHED', 'ION_FLAG_CACHED_NEEDS_SYNC',
    'ION_HEAP_TYPE_CARVEOUT', 'ION_HEAP_TYPE_CHUNK',
    'ION_HEAP_TYPE_CUSTOM', 'ION_HEAP_TYPE_DMA',
    'ION_HEAP_TYPE_SYSTEM', 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    'ION_IOC_MAGIC', 'ION_NUM_HEAPS', '_UAPI_LINUX_ION_H',
    'ion_heap_type', 'ion_user_handle_t',
    'struct_ion_allocation_data', 'struct_ion_custom_data',
    'struct_ion_fd_data', 'struct_ion_handle_data']
