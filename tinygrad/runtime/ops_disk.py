import os, sys, mmap, io, ctypes, contextlib, pathlib, functools, collections, struct
from dataclasses import replace
from typing import cast
from tinygrad.helpers import OSX, mv_address, flatten, to_tuple, unwrap, ceildiv
from tinygrad.device import BufferStorage, MMIOInterface, Compiled, Allocator, Buffer, BufferSpec, Device, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, uopfunc
from tinygrad.runtime.support.hcq2 import ccall, ins, chunks
with contextlib.suppress(ImportError):
  import _posixshmem
  from tinygrad.runtime.autogen import io_uring, libc

class DiskDevice(Compiled):
  def synchronize(self, timeout:int|None=None): pass

  def __init__(self, device:str):
    self.size: int|None = None
    self.fd: int|None = None
    self.refcount, self.info = 0, Buffer(HCQ_RUNTIME_DEV.value, 16, options=BufferSpec(nolru=True)) # info: [fd, mmap address] for batch reads
    super().__init__(device, DiskAllocator(self), [], None)
  def _might_open(self, size:int):
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None and hasattr(self, "mem"):
      self.refcount += 1
      return
    filename = self.device[len("disk:"):]

    if sys.platform != "win32" and filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0))
      except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      if not pathlib.Path(filename).is_block_device() and os.fstat(self.fd).st_size < size: os.ftruncate(self.fd, size)
      self.mem = mmap.mmap(self.fd, size)
      self.info.host.view()[:16] = struct.pack("2Q", self.fd, mv_address(memoryview(self.mem)))
    self.size = size
    if hasattr(self.mem, 'madvise') and (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
      with contextlib.suppress(OSError): self.mem.madvise(hp) # some systems have transparent_hugepage disabled
    self.refcount += 1
  def _might_close(self):
    self.refcount -= 1
    if self.refcount == 0:
      if self.fd is not None: os.close(self.fd)
      if hasattr(self, "mem"):
        try: self.mem.close()
        except BufferError: pass
      self.size = None

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert hasattr(self.device, "mem"), f"DiskBuffer wasn't opened: {self.device.device}"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  lru = False
  def _alloc(self, size:int, options) -> BufferStorage:
    self.dev._might_open(size)
    return BufferStorage(opaque:=DiskBuffer(self.dev, size), None, MMIOInterface(mv_address(opaque._buf()), size))

  def _free(self, storage:BufferStorage, options): self.dev._might_close()
  def _as_buffer(self, src:DiskBuffer): return src._buf()
  def _copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def _copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and self.dev.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.dev.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        bytes_read = 0
        while (n := fo.readinto(dest[bytes_read:])) is not None and n > 0: bytes_read += n
    else:
      dest[:] = src._buf()

  def _offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)

# *****************
# UOps implementation

CHUNK_SZ, READ_SZ, SLOTS, RING_ENTRIES = 64 << 20, 1 << 20, 2, 256
SLOT_SZ, STAGE_SZ = CHUNK_SZ + 4096, SLOTS * (CHUNK_SZ + 4096) + 16 # +4096: reads are page aligned

@uopfunc
def disk_read(file:UOp, src_va:UOp, nbytes:UOp, stage:UOp, counts:UOp, n0:UOp, rings:UOp, sqes:UOp) -> UOp: # src_va in the file's mmap
  fd, p = unwrap(uring())[:2]
  file, counts, rings, sqes = (b.replace(arg=replace(b.arg, volatile=True, device=None)) for b in (file, counts, rings, sqes))

  # waits for the gpu to free the slot of chunk n
  n = n0 + (chunk:=UOp.range(ceildiv(nbytes, CHUNK_SZ), 0, dtype=dtypes.uint64))
  free = (copied:=counts.after(chunk, loop:=UOp.loop(2))[1]).backedge(loop, copied.cast(dtypes.int64) <= n.cast(dtypes.int64) - SLOTS)

  # queues the chunk as READ_SZ reads
  pos = src_va + chunk * CHUNK_SZ - file[1]
  span = (pos % 4096 + (nbytes - chunk * CHUNK_SZ).minimum(CHUNK_SZ) + 4095) // 4096 * 4096
  read = UOp.range(nreads:=ceildiv(span, READ_SZ), 1, dtype=dtypes.uint64)
  tail = rings.after(free)[sq_tail:=p.sq_off.tail // 4].cast(dtypes.uint64)
  sqe = [file[0] << 32 | io_uring.IOSQE_ASYNC << 8 | io_uring.IORING_OP_READ, pos - pos % 4096 + read * READ_SZ,
         stage.getaddr(HCQ_RUNTIME_DEV.value) + n % SLOTS * SLOT_SZ + read * READ_SZ, (span - read * READ_SZ).minimum(READ_SZ)]
  queued = UOp.group(*[sqes[(tail + read) % RING_ENTRIES * 8 + j].store(v) for j, v in enumerate(sqe)]).end(read)

  # submits, waits for all, marks the chunk read
  to_submit = nreads.cast(dtypes.int).after(rings.after(queued)[sq_tail].store((tail + nreads).cast(dtypes.uint32)))
  done = ccall(libc.syscall, io_uring.NR_io_uring_enter, fd, to_submit, to_submit, io_uring.IORING_ENTER_GETEVENTS, 0, 0)
  reaped = rings.after(done)[p.cq_off.head // 4].store(rings.after(done)[p.cq_off.tail // 4])
  return counts.after(reaped)[0].store(n + 1).end(chunk).sink()

# *****************
# 2. rewriter

def is_disk_read(c:UOp) -> bool:
  dev, src = [to_tuple(b.device)[0] for b in c.src[1:3]] if c.op is Ops.CALL and c.body.op is Ops.STORE else ("", "")
  return src.startswith("DISK:") and not src.startswith("DISK:shm:") and Device[dev].has_copy_queue and not dev.startswith("NULL") and bool(uring())

def disk_copy_rewriter(s:UOp) -> UOp|None:
  lins = [submit.without_after.src[1].without_after for submit in s.src]
  if not (copies:=[c for lin in lins for c in lin.src if is_disk_read(c)]): return None

  # the host reads a chunk from disk into a slot, the gpu copies it
  offs, gpu_ops, done = collections.Counter[str](), collections.defaultdict[UOp, list[UOp]](list), s.src[-1]
  rings = UOp.alloc((unwrap(uring())[2].nbytes // 4,), dtypes.uint32, 0, device=HCQ_RUNTIME_DEV.value).rtag("uring_rings") # heads, tails, cqes
  sqes = UOp.alloc((8 * RING_ENTRIES,), dtypes.uint64, 0, device=HCQ_RUNTIME_DEV.value).rtag("uring_sqes")
  for c in copies:
    src, dev, file = c.src[2], to_tuple(c.src[1].device)[0], UOp.alloc((2,), dtypes.uint64, 0, device=c.src[2].device).rtag("disk_info")
    stage = UOp.alloc((STAGE_SZ,), dtypes.uint8, 0, device=dev).rtag("disk_stage") # the slots, then [chunks read, chunks copied]
    n0, src_va = (counts:=stage[STAGE_SZ - 16:].bitcast(dtypes.uint64)).index(0).load() + offs[dev], src.getaddr(HCQ_RUNTIME_DEV.value)
    offs[dev] += ceildiv(src.nbytes(), CHUNK_SZ)
    done = disk_read(file, src_va, UOp.const(src.nbytes(), dtypes.uint64), stage, counts.after(done), n0, rings, sqes)

    # waits for the chunk, copies, frees the slot
    dst, slots = c.src[1].getaddr(dev), stage.getaddr(dev) + (src_va - file.index(1).load()) % 4096
    for chunk, nb in chunks(src.nbytes(), CHUNK_SZ, dtypes.uint64):
      n = n0 + chunk
      ops = [ins("wait", counts, n + 1), ins("copy", dst + chunk * CHUNK_SZ, slots + n % SLOTS * SLOT_SZ, nb), ins("store", counts[1:], n + 1)]
      gpu_ops[c] += [UOp(Ops.LINEAR, src=tuple(ops)).end(chunk)] if isinstance(chunk, UOp) else ops
  return s.replace(src=(*s.src, done)).substitute({lin: lin.replace(src=tuple(flatten(gpu_ops.get(c, [c]) for c in lin.src))) for lin in lins})
Compiled.pm_batch = Compiled.pm_batch + PatternMatcher([(UPat(Ops.SINK, name="s"), disk_copy_rewriter)])

# *****************
# 3. bufferize

def shared(fd:int, at:int, n:int) -> Buffer: # n bytes of memory the kernel shares at offset at
  return Buffer(HCQ_RUNTIME_DEV.value, n, options=BufferSpec(external_ptr=libc.mmap(0, n, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd, at)))

@functools.cache
def uring() -> tuple[int, io_uring.struct_io_uring_params, Buffer, Buffer]|None: # fd, offsets, the rings (heads, tails, cqes), the sqes
  if sys.platform != "linux" or hasattr(sys, "getandroidapilevel"): return None
  p = io_uring.struct_io_uring_params(flags=io_uring.IORING_SETUP_NO_SQARRAY)
  if (fd:=libc.syscall(io_uring.NR_io_uring_setup, RING_ENTRIES, ctypes.byref(p))) < 0: return None
  return fd, p, shared(fd, io_uring.IORING_OFF_SQ_RING, p.cq_off.cqes + 16 * p.cq_entries), shared(fd, io_uring.IORING_OFF_SQES, 64 * RING_ENTRIES)

@functools.cache
def _stage(dev) -> Buffer: return Buffer(dev, STAGE_SZ, options=BufferSpec(host=True, uncached=True, cpu_access=True), initial_value=bytes(STAGE_SZ))

Compiled.pm_bufferize += PatternMatcher([(UPat(Ops.ALLOC, tag="disk_info", name="b"), lambda b: cast(DiskDevice, Device[b.device]).info),
  (UPat(Ops.ALLOC, tag="disk_stage", name="b"), lambda b: _stage(b.device)),
  (UPat(Ops.ALLOC, tag="uring_rings"), lambda: unwrap(uring())[2]), (UPat(Ops.ALLOC, tag="uring_sqes"), lambda: unwrap(uring())[3])])
