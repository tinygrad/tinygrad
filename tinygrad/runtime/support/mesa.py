import os, ctypes, ctypes.util
from tinygrad.helpers import getenv

MESA_PATH = getenv('MESA_PATH', '/usr/lib')
loaded = False
if (not os.path.exists(path:=f"{MESA_PATH}/libtinymesa_cpu.so") and not (path:=ctypes.util.find_library('tinymesa_cpu') or '')
    and not os.path.exists(path:=f"{MESA_PATH}/libtinymesa.so") and not (path:=ctypes.util.find_library('tinymesa') or '')):
  error = FileNotFoundError(f"libtinymesa not found ({MESA_PATH=}). See https://github.com/sirhcm/tinymesa (release: mesa-25.2.3-3cb1b80)")
else: loaded, dll = True, ctypes.CDLL(path)

def __getattr__(name):
  if not loaded: raise error
  return getattr(dll, name)

