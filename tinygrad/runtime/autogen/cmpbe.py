# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes, os
from tinygrad.runtime.support import c

dll = c.DLL('cmpbe', os.getenv('CMPBE_PATH', 'mali'), emsg='set CMPBE_PATH to the compiler-only libmali.so')

@dll.bind(None, ctypes.c_void_p, ctypes.c_uint32)
def cmpbe_v2_init_options(options:ctypes.c_void_p, hw_rev:int) -> None: ...
@dll.bind(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
def cmpbe_v2_set_option_value(options:ctypes.c_void_p, option:int, value:int) -> int: ...
@dll.bind(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, c.POINTER[ctypes.c_void_p])
def cmpbe_v2_compile_single_shader(ctx:ctypes.c_void_p, source:ctypes.c_void_p, source_size:int,
  shader:ctypes.c_void_p, stage:int, flags:int, variant:int, variant_info:ctypes.c_void_p,
  user:ctypes.c_void_p, result:c.POINTER[ctypes.c_void_p]) -> int: ...
@dll.bind(None, ctypes.c_void_p)
def cmpbe_v2_free_compilation_result(result:ctypes.c_void_p) -> None: ...
@dll.bind(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, c.POINTER[ctypes.c_void_p])
def cmpbe_v2_deserialize_MBS2_to_C(ctx:ctypes.c_void_p, mbs2:ctypes.c_void_p, output:c.POINTER[ctypes.c_void_p]) -> int: ...

def put_ptr(buf, offset:int, value:int|None):
  ctypes.c_uint64.from_address(ctypes.addressof(buf) + offset).value = value or 0
