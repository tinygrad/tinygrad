import ctypes, ctypes.util
from mmap import PAGESIZE
from tinygrad.helpers import round_up, cpu_time_execution

helper_handle = ctypes.CDLL(ctypes.util.find_library('kernel32'))

MEM_COMMIT = 0x00001000
PAGE_EXECUTE_READWRITE = 0x40

VirtualAlloc = helper_handle.VirtualAlloc
VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
VirtualAlloc.restype = ctypes.c_void_p

GetCurrentProcess = helper_handle.GetCurrentProcess
GetCurrentProcess.argtypes = []
GetCurrentProcess.restype = ctypes.c_void_p

FlushInstructionCache = helper_handle.FlushInstructionCache
FlushInstructionCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
FlushInstructionCache.restype = ctypes.c_bool

# CPUProgram is a jit/shellcode program that can be just mmapped and jumped to
class CPUProgram:
  def __init__(self, name:str, lib:bytes):
    mem = VirtualAlloc(None, round_up(len(lib), PAGESIZE), MEM_COMMIT, PAGE_EXECUTE_READWRITE)
    ctypes.memmove(mem, lib, len(lib))
    assert FlushInstructionCache(GetCurrentProcess(), mem, round_up(len(lib), PAGESIZE)), 'failed to flush instruction cache'
    self.fxn = ctypes.CFUNCTYPE(None)(mem)

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)
