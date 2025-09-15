import socket, uuid, json, asyncio, os
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import Tensor

TINYFS_ENDPOINT = getenv("TINYFS_ENDPOINT", "localhost:6767")

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):].upper()
    super().__init__(device, TinyFSAllocator(self), None, None, None)

    # fetch node info
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TINYFS_ENDPOINT.rsplit(":", 1)[0], int(TINYFS_ENDPOINT.rsplit(":", 1)[1])))
    s.send(f"INFO\r\n".encode())
    s.recv(16)

    info = b""
    while not info.endswith(b"\r\n"):
      info += s.recv(1024)
    self.node_info = json.loads(info[:-2])
    print(self.node_info)

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, sock=None, request_id=None, locs=None, src=None):
    self.device, self.size, self.offset = device, size, offset
    self.request_id: uuid.UUID|None = request_id
    self.locs = locs or []
    self.src: bytearray = src or bytearray()
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

    if dest.device.op == "LOAD":
      locs = b""
      while not locs.endswith(b"\r\n"):
        locs += dest.sock.recv(1024)
      dest.locs = json.loads(locs[:-2])
      dest.src = bytearray(len(src))
      dest.src[:] = src

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"Copying out {src.size} bytes from {src.device.op}")
    if src.device.op == "LOAD" and getenv("USE_ASYNC_COPY", 1):
      asyncio.run(self._copyout_async(dest, src))
    else:
      src.sock.send(f"{src.device.op}_OUT {src.size} {src.request_id}\r\n".encode())
      src.request_id = uuid.UUID(bytes=src.sock.recv(16))
      if DEBUG >= 2: print(f"Request ID: {src.request_id}")
      recv = 0
      while recv < src.size:
        recv += src.sock.recv_into(dest[recv:], src.size - recv)

  async def _copyout_async(self, dest:memoryview, src:TinyFSBuffer):
    queue = asyncio.Queue()
    for item in enumerate(src.locs): queue.put_nowait(item)
    for _ in range(nw := (os.cpu_count() or getenv("ASYNC_COPY_WORKERS", 1))): queue.put_nowait(None)

    async def _worker():
      conns = {}
      while True:
        if (item := await queue.get()) is None:
          queue.task_done()
          break
        i, loc = item
        if loc not in conns:
          addr = src.device.node_info[loc][-1]
          conns[loc] = await asyncio.open_connection(*addr.rsplit(":", 1))

        ptr = i * Tensor.CHUNK_SIZE
        size = min(len(dest[ptr:ptr+Tensor.CHUNK_SIZE]), Tensor.CHUNK_SIZE)

        conns[loc][1].write(f"CHUNK_OUT {size}\r\n".encode())
        conns[loc][1].write(src.src[i*16:(i+1)*16])
        await conns[loc][1].drain()

        chunk = await conns[loc][0].readexactly(size)
        dest[ptr:ptr+len(chunk)] = chunk

        queue.task_done()
      for _, writer in conns.values():
        writer.close()
        await writer.wait_closed()

    workers = [asyncio.create_task(_worker()) for _ in range(nw)]
    await queue.join()

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    assert offset == 0, f"only offset 0 supported, found offset {offset}"
    return TinyFSBuffer(buf.device, size, offset, buf.sock, buf.request_id, buf.locs, buf.src)
