# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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



_libraries = {}
_libraries['libocloc.so'] = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libocloc.so')


_OCLOC_API_H = True # macro
def OCLOC_MAKE_VERSION(_major, _minor):  # macro
   return ((_major<<16)|(_minor&0x0000ffff))  
OCLOC_NAME_VERSION_MAX_NAME_SIZE = 64 # macro
SIGNATURE = True # macro
pOclocInvoke = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))

# values for enumeration '_ocloc_version_t'
_ocloc_version_t__enumvalues = {
    65536: 'OCLOC_VERSION_1_0',
    65536: 'OCLOC_VERSION_CURRENT',
    2147483647: 'OCLOC_VERSION_FORCE_UINT32',
}
OCLOC_VERSION_1_0 = 65536
OCLOC_VERSION_CURRENT = 65536
OCLOC_VERSION_FORCE_UINT32 = 2147483647
_ocloc_version_t = ctypes.c_uint32 # enum
ocloc_version_t = _ocloc_version_t
ocloc_version_t__enumvalues = _ocloc_version_t__enumvalues

# values for enumeration '_ocloc_error_t'
_ocloc_error_t__enumvalues = {
    0: 'OCLOC_SUCCESS',
    -6: 'OCLOC_OUT_OF_HOST_MEMORY',
    -11: 'OCLOC_BUILD_PROGRAM_FAILURE',
    -33: 'OCLOC_INVALID_DEVICE',
    -44: 'OCLOC_INVALID_PROGRAM',
    -5150: 'OCLOC_INVALID_COMMAND_LINE',
    -5151: 'OCLOC_INVALID_FILE',
    -5152: 'OCLOC_COMPILATION_CRASH',
}
OCLOC_SUCCESS = 0
OCLOC_OUT_OF_HOST_MEMORY = -6
OCLOC_BUILD_PROGRAM_FAILURE = -11
OCLOC_INVALID_DEVICE = -33
OCLOC_INVALID_PROGRAM = -44
OCLOC_INVALID_COMMAND_LINE = -5150
OCLOC_INVALID_FILE = -5151
OCLOC_COMPILATION_CRASH = -5152
_ocloc_error_t = ctypes.c_int32 # enum
ocloc_error_t = _ocloc_error_t
ocloc_error_t__enumvalues = _ocloc_error_t__enumvalues
class struct__ocloc_name_version(Structure):
    pass

struct__ocloc_name_version._pack_ = 1 # source:False
struct__ocloc_name_version._fields_ = [
    ('version', ctypes.c_uint32),
    ('name', ctypes.c_char * 64),
]

ocloc_name_version = struct__ocloc_name_version
uint32_t = ctypes.c_uint32
try:
    oclocInvoke = _libraries['libocloc.so'].oclocInvoke
    oclocInvoke.restype = ctypes.c_int32
    oclocInvoke.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), uint32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), uint32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))]
except AttributeError:
    pass
try:
    oclocFreeOutput = _libraries['libocloc.so'].oclocFreeOutput
    oclocFreeOutput.restype = ctypes.c_int32
    oclocFreeOutput.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))]
except AttributeError:
    pass
try:
    oclocVersion = _libraries['libocloc.so'].oclocVersion
    oclocVersion.restype = ctypes.c_int32
    oclocVersion.argtypes = []
except AttributeError:
    pass
__all__ = \
    ['OCLOC_BUILD_PROGRAM_FAILURE', 'OCLOC_COMPILATION_CRASH',
    'OCLOC_INVALID_COMMAND_LINE', 'OCLOC_INVALID_DEVICE',
    'OCLOC_INVALID_FILE', 'OCLOC_INVALID_PROGRAM',
    'OCLOC_NAME_VERSION_MAX_NAME_SIZE', 'OCLOC_OUT_OF_HOST_MEMORY',
    'OCLOC_SUCCESS', 'OCLOC_VERSION_1_0', 'OCLOC_VERSION_CURRENT',
    'OCLOC_VERSION_FORCE_UINT32', 'SIGNATURE', '_OCLOC_API_H',
    '_ocloc_error_t', '_ocloc_version_t', 'oclocFreeOutput',
    'oclocInvoke', 'oclocVersion', 'ocloc_error_t',
    'ocloc_error_t__enumvalues', 'ocloc_name_version',
    'ocloc_version_t', 'ocloc_version_t__enumvalues', 'pOclocInvoke',
    'struct__ocloc_name_version', 'uint32_t']
