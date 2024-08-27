# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes, ctypes.util, os


c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 8:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*8

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libc'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libc'] = ctypes.CDLL(ctypes.util.find_library('c')) #  ctypes.CDLL('libc')


size_t = ctypes.c_uint64
try:
    sys_icache_invalidate = _libraries['libc'].sys_icache_invalidate
    sys_icache_invalidate.restype = None
    sys_icache_invalidate.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
__all__ = \
    ['size_t', 'sys_icache_invalidate']
