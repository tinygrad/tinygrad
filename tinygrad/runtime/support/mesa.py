import os, ctypes, ctypes.util
from tinygrad.helpers import getenv

found = (os.path.exists(path:=(PATH:=getenv('MESA_PATH', '/usr/lib'))+"/libtinymesa_cpu.so") or (path:=ctypes.util.find_library('tinymesa_cpu') or '')
         or os.path.exists(path:=f"{PATH}/libtinymesa.so") or (path:=ctypes.util.find_library('tinymesa') or ''))
