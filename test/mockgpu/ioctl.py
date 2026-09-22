import ctypes, errno, copy
from contextlib import contextmanager
from typing import Any
from test.mockgpu.cfunc_ref import BoundCFunctionRef

def _detached_error(error: Exception) -> Exception:
  detached = copy.copy(error)
  detached.__traceback__ = detached.__cause__ = detached.__context__ = None
  return detached

class IoctlBridge:
  def __init__(self, native, tracked_fds):
    self.native, self.tracked_fds = native, tracked_fds
    self.last_error: tuple[int, int, Exception] | None = None
    self.pending_error: tuple[int, int, Exception] | None = None
    self.checking = False
    self.callback: Any = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, use_errno=True)(self.call)
    self.callback_ref = BoundCFunctionRef.bind(self.callback, 'ioctl')
    setattr(self.callback, '__name__', 'ioctl')
    self.callback.__module__ = 'tinygrad.runtime.autogen.libc'

  def call(self, fd, request, arg):
    self.last_error = None
    if (file := self.tracked_fds.get(fd)) is None: return self.native(fd, ctypes.c_ulong(request), ctypes.c_void_p(arg))
    if self.checking and self.pending_error is not None:
      ctypes.set_errno(errno.EIO)
      return -1
    try:
      result = file.ioctl(fd, request, arg)
      if type(result) is not int or not -(1 << 31) <= result < 1 << 31: raise TypeError(f'invalid ioctl result {result!r}')
      return result
    except Exception as exc:
      # Exceptions cannot cross a C callback; retain the diagnostic and return an explicit errno.
      exc = _detached_error(exc)
      self.last_error = (fd, request, exc)
      if self.checking and self.pending_error is None: self.pending_error = self.last_error
      ctypes.set_errno((exc.errno or errno.EIO) if isinstance(exc, OSError) else errno.EIO)
      return -1

  def run(self, fn, recover):
    with self.scope(recover): return fn()

  @contextmanager
  def scope(self, recover):
    if self.checking:
      yield
      return
    self.checking, self.pending_error = True, None
    try: yield
    finally:
      try: self.check(recover)
      finally: self.checking, self.pending_error = False, None

  def check(self, recover):
    if self.pending_error is not None:
      fd, _, stored_error = self.pending_error
      try: recover(fd)
      except Exception as recovery_error: raise _detached_error(stored_error) from recovery_error
      raise _detached_error(stored_error)
