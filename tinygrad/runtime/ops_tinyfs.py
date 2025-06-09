from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):]
    super().__init__(device, TinyFSAllocator(self), None, None, None)

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<TinyFSBuffer size={self.size} offset={self.offset}>"

class TinyFSAllocator(Allocator[TinyFSDevice]):
  def _alloc(self, size, options):
    return TinyFSBuffer(self.dev, size)
  def _free(self, opaque, options): pass

  def _copyin(self, dest:TinyFSBuffer, src:memoryview):
    pass

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    headers = {
      "User-Agent": "tinygrad 0.10.3",
      "Range": f"bytes={src.offset}-{src.offset + src.size - 1}",
    }
    if DEBUG >= 2: print(f"tinyfs request: {self.dev.op} with headers {headers}")
