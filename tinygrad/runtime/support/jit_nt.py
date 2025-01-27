import ctypes, ctypes.util
from mmap import PAGESIZE
from tinygrad.helpers import round_up, cpu_time_execution

helper_handle = ctypes.CDLL(ctypes.util.find_library('kernel32'))

MEM_COMMIT = 0x00001000
PAGE_EXECUTE_READWRITE = 0x40

VirtualAlloc = helper_handle.VirtualAlloc
VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
VirtualAlloc.restype = ctypes.c_void_p

class CPUProgram:
  def __init__(self, name:str, lib:bytes):
    mem = VirtualAlloc(None, round_up(len(lib), PAGESIZE), MEM_COMMIT, PAGE_EXECUTE_READWRITE)
    ctypes.memmove(mem, lib, len(lib))
    self.fxn = ctypes.CFUNCTYPE(None)(mem)

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)
