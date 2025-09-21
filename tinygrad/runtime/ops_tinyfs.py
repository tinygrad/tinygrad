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
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect((TINYFS_ENDPOINT.rsplit(":", 1)[0], int(TINYFS_ENDPOINT.rsplit(":", 1)[1])))
    self.sock.send("INFO\r\n".encode())
    self.sock.recv(16)

    info = b""
    while not info.endswith(b"\r\n"):
      info += self.sock.recv(1024)
    self.node_info = json.loads(info[:-2])

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, request_id=None, copyout_queue=None):
    self.device, self.size, self.offset = device, size, offset
    self.request_id: uuid.UUID|None = request_id
    self.copyout_queue = copyout_queue
  def __repr__(self): return f"<TinyFSBuffer size={self.size} offset={self.offset}>"

class TinyFSAllocator(Allocator[TinyFSDevice]):
  def _alloc(self, size, options):
    return TinyFSBuffer(self.dev, size)

  def _copyin(self, dest:TinyFSBuffer, src:memoryview):
    if DEBUG >= 2: print(f"Copying in {dest.size} bytes to {dest.device.op}")
    self.dev.sock.send(f"{dest.device.op}_IN {dest.size}\r\n".encode())

    # read the response uuid
    dest.request_id = uuid.UUID(bytes=self.dev.sock.recv(16))
    if DEBUG >= 2: print(f"Request ID: {dest.request_id}")

    self.dev.sock.sendall(src)

    if dest.device.op == "LOAD":
      locs = b""
      while not locs.endswith(b"\r\n"):
        locs += self.dev.sock.recv(1024)
      locs = json.loads(locs[:-2])

      dest.copyout_queue = asyncio.Queue()
      for i, loc in enumerate(locs):
        dest.copyout_queue.put_nowait((i, loc, src[i*16:(i+1)*16]))

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"Copying out {src.size} bytes from {src.device.op}")
    if src.device.op == "LOAD":
      asyncio.run(self._copyout_async(dest, src))
    else:
      self.dev.sock.send(f"{src.device.op}_OUT {src.size} {src.request_id}\r\n".encode())
      src.request_id = uuid.UUID(bytes=self.dev.sock.recv(16))
      if DEBUG >= 2: print(f"Request ID: {src.request_id}")
      recv = 0
      while recv < src.size:
        recv += self.dev.sock.recv_into(dest[recv:], src.size - recv)

  async def _copyout_async(self, dest:memoryview, src:TinyFSBuffer):
    for _ in range(nw := getenv("ASYNC_COPY_WORKERS", 1)): src.copyout_queue.put_nowait(None)

    async def _worker():
      conns = {}
      while True:
        if (item := await src.copyout_queue.get()) is None:
          src.copyout_queue.task_done()
          break
        i, loc, h = item
        if loc not in conns:
          addr = src.device.node_info[loc][-1]
          conns[loc] = await asyncio.open_connection(*addr.rsplit(":", 1))

        ptr = i * Tensor.CHUNK_SIZE
        size = min(len(dest[ptr:ptr+Tensor.CHUNK_SIZE]), Tensor.CHUNK_SIZE)

        conns[loc][1].write(f"CHUNK_OUT {size}\r\n".encode())
        conns[loc][1].write(h)
        await conns[loc][1].drain()

        chunk = await conns[loc][0].readexactly(size)
        dest[ptr:ptr+len(chunk)] = chunk

        src.copyout_queue.task_done()
      for _, writer in conns.values():
        writer.close()
        await writer.wait_closed()

    workers = [asyncio.create_task(_worker()) for _ in range(nw)]
    await src.copyout_queue.join()
    await asyncio.gather(*workers)
    del src.copyout_queue

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    assert offset == 0, f"only offset 0 supported, found offset {offset}"
    return TinyFSBuffer(buf.device, size, offset, buf.request_id, buf.copyout_queue)
