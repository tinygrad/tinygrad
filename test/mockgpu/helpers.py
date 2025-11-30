import ctypes
from tinygrad.helpers import findlib

def _try_dlopen_gpuocelot():
  try:
    gpuocelot_lib = ctypes.CDLL(findlib("gpuocelot"))
    gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return gpuocelot_lib
  except FileNotFoundError: return None

def _try_dlopen_remu():
  try:
    remu = ctypes.CDLL(findlib("remu", ["extra/remu/target/release"]))
    remu.run_asm.restype = ctypes.c_int32
    remu.run_asm.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    return remu
  except FileNotFoundError: return None
