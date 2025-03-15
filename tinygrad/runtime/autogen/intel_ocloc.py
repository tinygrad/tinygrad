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



_libraries = {}
_libraries['libocloc.so'] = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libocloc.so')


_OCLOC_API_H = True # macro
def OCLOC_MAKE_VERSION(_major, _minor):  # macro
   return ((_major<<16)|(_minor&0x0000ffff))  
SIGNATURE = True # macro

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
    'OCLOC_OUT_OF_HOST_MEMORY', 'OCLOC_SUCCESS', 'OCLOC_VERSION_1_0',
    'OCLOC_VERSION_CURRENT', 'OCLOC_VERSION_FORCE_UINT32',
    'SIGNATURE', '_OCLOC_API_H', '_ocloc_error_t', '_ocloc_version_t',
    'oclocFreeOutput', 'oclocInvoke', 'oclocVersion', 'ocloc_error_t',
    'ocloc_error_t__enumvalues', 'ocloc_version_t',
    'ocloc_version_t__enumvalues', 'uint32_t']
