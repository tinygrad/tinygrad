import socket
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.helpers import DEBUG, getenv
from tinygrad.runtime.ops_null import NullRenderer, NullProgram

TINYFS_ENDPOINT = getenv("TINYFS_ENDPOINT", "localhost:6767")

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):]
    super().__init__(device, TinyFSAllocator(self), None, None, None)

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, sock=None):
    self.device, self.size, self.offset = device, size, offset
    if sock is None:
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.sock.connect((TINYFS_ENDPOINT.split(":")[0], int(TINYFS_ENDPOINT.split(":")[1])))
      print("tinyfs connect", self.sock.getsockname(), "->", self.sock.getpeername())
    else:
      self.sock = sock
  def __repr__(self): return f"<TinyFSBuffer size={self.size} offset={self.offset}>"

class TinyFSAllocator(Allocator[TinyFSDevice]):
  def _alloc(self, size, options):
    return TinyFSBuffer(self.dev, size)

  def _free(self, opaque:TinyFSBuffer, options):
    opaque.sock.close()
    del opaque.sock

  def _copyin(self, dest:TinyFSBuffer, src:memoryview):
    if DEBUG >= 2: print(f"tinyfs copyin: {self.dev.op} dest {dest}")

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"tinyfs copyout: {self.dev.op} src {src}")

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    print(f"tinyfs offset: {self.dev.op} buf {buf} size {size} offset {offset}")
    return TinyFSBuffer(buf.device, size, offset, buf.sock)
