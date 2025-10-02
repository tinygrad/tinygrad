import os, ctypes, ctypes.util
from tinygrad.helpers import getenv

MESA_PATH = getenv('MESA_PATH', '/usr/lib')
found = (os.path.exists(path:=f"{MESA_PATH}/libtinymesa_cpu.so") or (path:=ctypes.util.find_library('tinymesa_cpu')) or
         os.path.exists(path:=f"{MESA_PATH}/libtinymesa.so") or (path:=ctypes.util.find_library('tinymesa')))
