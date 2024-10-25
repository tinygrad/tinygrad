from tinygrad.device import Compiled, MallocAllocator
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer import Program
from tinygrad.renderer.asm import ASMRenderer
from tinygrad.runtime.autogen import libc
import ctypes, mmap

class ASMProgram:
  def __init__(self, device: Compiled, name:str, code:bytes):
    self.device, self.name, self.len = device, name, len(code)
    self.buf = libc.mmap(0, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                         mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    ctypes.memmove(self.buf, (ctypes.c_char * len(code)).from_buffer_copy(code), len(code))
    pass
  def __del__(self):
    libc.munmap(self.buf, self.len)
  def __call__(self, *bufs, vals=(), wait=False): 
    return cpu_time_execution(lambda: ctypes.CFUNCTYPE(self.buf)(*bufs, *vals), enable=wait)

class ASMDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, ASMRenderer(), None, ASMProgram, None)
