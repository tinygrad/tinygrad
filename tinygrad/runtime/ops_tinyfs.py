import socket, uuid
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG, getenv

TINYFS_ENDPOINT = getenv("TINYFS_ENDPOINT", "localhost:6767")

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):].upper()
    super().__init__(device, TinyFSAllocator(self), None, None, None)

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, sock=None, request_id=None):
    self.device, self.size, self.offset = device, size, offset
    self.request_id: uuid.UUID|None = request_id
    if sock is None:
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.sock.connect((TINYFS_ENDPOINT.split(":")[0], int(TINYFS_ENDPOINT.split(":")[1])))
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
    if DEBUG >= 2: print(f"Copying in {dest.size} bytes to {dest.device.op}")
    dest.sock.send(f"{dest.device.op}_IN {dest.size}\r\n".encode())

    # read the response uuid
    dest.request_id = uuid.UUID(bytes=dest.sock.recv(16))
    if DEBUG >= 2: print(f"Request ID: {dest.request_id}")

    dest.sock.sendall(src)

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"Copying out {src.size} bytes from {src.device.op}")
    src.sock.send(f"{src.device.op}_OUT {src.size} {src.request_id}\r\n".encode())

    src.request_id = uuid.UUID(bytes=src.sock.recv(16))
    if DEBUG >= 2: print(f"Request ID: {src.request_id}")

    recv = 0
    while recv < src.size:
      recv += src.sock.recv_into(dest[recv:], src.size - recv)

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    assert offset == 0, f"only offset 0 supported, found offset {offset}"
    return TinyFSBuffer(buf.device, size, offset, buf.sock, buf.request_id)
