import ctypes, functools
from typing import Annotated, cast
from tinygrad.runtime.support import c

@c.record
class py_buffer(c.Struct):
  SIZE = 80
  buf: Annotated[ctypes.c_void_p, 0] # there are more fields, but we only care about this one
c.init_records()

@functools.partial(c.DLL.bind, cast(c.DLL, ctypes.pythonapi))
def PyObject_GetBuffer(obj:ctypes.py_object, view:c.POINTER[py_buffer], flags:ctypes.c_int) -> ctypes.c_int: pass # type: ignore[empty-body]
@functools.partial(c.DLL.bind, cast(c.DLL, ctypes.pythonapi))
def PyBuffer_Release(view:c.POINTER[py_buffer]): pass # type: ignore[empty-body]
@functools.partial(c.DLL.bind, cast(c.DLL, ctypes.pythonapi))
def PyMemoryView_FromMemory(mem:ctypes.c_void_p, size:ctypes.c_ssize_t, flags:ctypes.c_int) -> ctypes.py_object: pass # type: ignore[empty-body]

BUF_SIMPLE = 0
BUF_READ = 0x100
BUF_WRITE = 0x200

def mv_address(mv:memoryview):
  PyObject_GetBuffer(ctypes.py_object(mv), (view:=py_buffer()), BUF_SIMPLE) # ctypes automatically does error handling on PyDLL
  ret = view.buf
  PyBuffer_Release(view)
  return ret

def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array: return (to_type * len(mv)).from_address(mv_address(mv))
def to_mv(ptr:int, sz:int) -> memoryview: return PyMemoryView_FromMemory(ptr, sz, BUF_WRITE) # FIXME: reference counting?
