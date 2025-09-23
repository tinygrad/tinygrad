import socket, uuid, json, asyncio, threading
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import Tensor

TINYFS_ENDPOINT = getenv("TINYFS_ENDPOINT", "localhost:6767")

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):].upper()
    super().__init__(device, TinyFSAllocator(self), None, None, None)

    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect((TINYFS_ENDPOINT.rsplit(":", 1)[0], int(TINYFS_ENDPOINT.rsplit(":", 1)[1])))
    self.sock.send("INFO\r\n".encode())
    self.sock.recv(16)
    self.sfile = self.sock.makefile("rwb")

    # fetch node info
    info = self.sfile.readline()
    self.node_info = json.loads(info)
    if DEBUG >= 2: print(f"nodes: {self.node_info}")

    # spawn thread for async copyout
    self.start_event = threading.Event()
    self.t = threading.Thread(target=self._start_thread, daemon=True)
    self.t.start()
    self.start_event.wait()

  def finalize(self):
    self.sfile.close()
    if hasattr(self, "loop"):
      self.loop.call_soon_threadsafe(self.loop.stop)
    self.t.join()

  def _start_thread(self):
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.start_event.set()
    self.loop.run_forever()
    self.loop.close()

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, request_id=None, copyout_queue=None):
    self.device, self.size, self.offset = device, size, offset
    self.request_id: uuid.UUID|None = request_id
    self.copyout_queue = copyout_queue or []
  def __repr__(self): return f"<TinyFSBuffer size={self.size} offset={self.offset}>"

class TinyFSAllocator(Allocator[TinyFSDevice]):
  def _alloc(self, size, options):
    return TinyFSBuffer(self.dev, size)

  def _copyin(self, dest:TinyFSBuffer, src:memoryview):
    if DEBUG >= 2: print(f"Copying in {dest.size} bytes to {dest.device.op}")
    self.dev.sfile.write(f"{dest.device.op}_IN {dest.size}\r\n".encode())
    self.dev.sfile.flush()

    # read the response uuid
    dest.request_id = uuid.UUID(bytes=self.dev.sfile.read(16))
    if DEBUG >= 2: print(f"Request ID: {dest.request_id}")

    self.dev.sfile.write(src)
    self.dev.sfile.flush()

    if dest.device.op == "LOAD":
      locs = self.dev.sfile.readline()
      locs = json.loads(locs)

      dest.copyout_queue = []
      for i, loc in enumerate(locs):
        dest.copyout_queue.append((i, loc, src[i*16:(i+1)*16]))

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"Copying out {src.size} bytes from {src.device.op}")
    if src.device.op == "LOAD":
      asyncio.run_coroutine_threadsafe(self._copyout_async(dest, src), src.device.loop).result()
    else:
      self.dev.sfile.write(f"{src.device.op}_OUT {src.size} {src.request_id}\r\n".encode())
      self.dev.sfile.flush()
      src.request_id = uuid.UUID(bytes=self.dev.sfile.read(16))
      if DEBUG >= 2: print(f"Request ID: {src.request_id}")
      self.dev.sfile.readinto(dest)

  async def _copyout_async(self, dest:memoryview, src:TinyFSBuffer):
    queue = asyncio.Queue()
    for item in src.copyout_queue: queue.put_nowait(item)
    for _ in range(nw := getenv("ASYNC_COPY_WORKERS", 4)): queue.put_nowait(None)

    async def _worker():
      conns = {}
      loop = asyncio.get_running_loop()
      while True:
        if (item := await queue.get()) is None:
          queue.task_done()
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

        await loop.run_in_executor(None, lambda: dest[ptr:ptr+len(chunk)].__setitem__(slice(None), chunk))

        queue.task_done()
      for _, writer in conns.values():
        writer.close()
        await writer.wait_closed()

    workers = [asyncio.create_task(_worker()) for _ in range(nw)]
    await queue.join()
    await asyncio.gather(*workers)
    src.copyout_queue.clear()

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    assert offset == 0, f"only offset 0 supported, found offset {offset}"
    return TinyFSBuffer(buf.device, size, offset, buf.request_id, buf.copyout_queue)
