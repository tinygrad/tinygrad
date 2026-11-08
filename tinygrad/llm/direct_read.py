# file reads straight into memory with io_uring and O_DIRECT: many aligned reads in flight (a raid of nvme drives reads at its full rate, the
# page cache's copy and its eviction not in the way). the pieces of a read that aren't page aligned (a file's tensors are 32 byte aligned)
# are read through the page cache: a buffer laid out with its file's offsets modulo a page reads at the drives' rate
from __future__ import annotations
import os, mmap, ctypes, struct, array, threading, collections, errno
from tinygrad.helpers import round_up, mv_address

# (IOSQE_ASYNC: the reads issued from the kernel's workers, not inline: 24.8 GB/s on a raid of 4 nvme drives, 21 without)
PAGE, READ_SZ, QD, FLAGS = 4096, 4 << 20, 64, 1 << 4

class Uring:
  """an io_uring of reads: read(reqs) reads every (fd, file offset, address, bytes) in full, QD at a time"""
  def __init__(self, entries:int=QD):
    from tinygrad.runtime.autogen import io_uring, libc
    self.io_uring, self.libc = io_uring, libc
    p = io_uring.struct_io_uring_params(flags=io_uring.IORING_SETUP_NO_SQARRAY)
    if (fd:=libc.syscall(io_uring.NR_io_uring_setup, entries, ctypes.byref(p))) < 0: raise OSError(ctypes.get_errno(), "io_uring_setup failed")
    self.fd, self.p, self.entries = fd, p, entries
    # (one mapping of the rings, IORING_FEAT_SINGLE_MMAP)
    self.rings = mmap.mmap(fd, max(p.sq_off.array + 4 * p.sq_entries, p.cq_off.cqes + 16 * p.cq_entries), mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=io_uring.IORING_OFF_SQ_RING)
    self.sqes_map = mmap.mmap(fd, 64 * p.sq_entries, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=io_uring.IORING_OFF_SQES)
    self.r32, self.sqes = memoryview(self.rings).cast('I'), memoryview(self.sqes_map).cast('Q')
  def close(self):
    self.r32.release(); self.sqes.release()  # noqa: E702
    self.rings.close(); self.sqes_map.close()  # noqa: E702
    os.close(self.fd)
  def read(self, reqs:list[tuple[int, int, int, int]]):
    p, r, sq, iu = self.p, self.r32, self.sqes, self.io_uring
    mask, cmask = r[p.sq_off.ring_mask // 4], r[p.cq_off.ring_mask // 4]
    todo, uid, inflight = collections.deque(reqs), 0, 0
    live:dict[int, tuple[int, int, int, int]] = {}
    while todo or inflight:
      tail, n = r[p.sq_off.tail // 4], 0
      while todo and inflight + n < self.entries:
        fd, off, addr, ln = todo.popleft()
        i = (tail + n) & mask
        sq[i*8:i*8+8] = array.array('Q', [fd << 32 | FLAGS << 8 | iu.IORING_OP_READ, off, addr, ln, uid, 0, 0, 0])
        live[uid], uid, n = (fd, off, addr, ln), uid + 1, n + 1
      r[p.sq_off.tail // 4] = (tail + n) & 0xffffffff
      inflight += n
      # (half the reads in flight waited for at once: a wakeup a read is the gil again for every read, slow with other threads running.
      # interrupted: the completions reaped, the rest waited for again)
      want = max(1, inflight // 2)
      if self.libc.syscall(iu.NR_io_uring_enter, self.fd, n, want, iu.IORING_ENTER_GETEVENTS, 0, 0) < 0 and ctypes.get_errno() != errno.EINTR:
        raise OSError(ctypes.get_errno(), "io_uring_enter failed")
      head, ctail = r[p.cq_off.head // 4], r[p.cq_off.tail // 4]
      while head != ctail:
        ud, res = struct.unpack_from("<Qi", self.rings, p.cq_off.cqes + (head & cmask) * 16)
        head, inflight = (head + 1) & 0xffffffff, inflight - 1
        fd, off, addr, ln = live.pop(ud)
        if res < 0: raise OSError(-res, f"read of {ln} bytes at {off} failed: {os.strerror(-res)}")
        if res == 0: raise EOFError(f"read of {ln} bytes at {off}: end of file")
        # (a short read: the rest again)
        if res < ln: todo.append((fd, off + res, addr + res, ln - res))
      r[p.cq_off.head // 4] = head

_local = threading.local()
_rings:list[Uring] = []
_rings_lock = threading.Lock()
def _ring() -> Uring|None:
  """the thread's ring (None: no io_uring here)"""
  if not hasattr(_local, "ring"):
    ring:Uring|None = None
    try: ring = Uring()
    except (OSError, ImportError, AttributeError): pass
    if ring is not None:
      with _rings_lock: _rings.append(ring)
    _local.ring = ring
  return _local.ring

def close_rings():
  """the rings of every thread closed (a load's done)"""
  with _rings_lock:
    for ring in _rings: ring.close()
    _rings.clear()
  _local.__dict__.clear()

class DirectFile:
  """a file's descriptors: O_DIRECT (None if the file system has none) and buffered"""
  def __init__(self, path:str):
    self.fd = os.open(path, os.O_RDONLY)
    try: self.dfd:int|None = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0)) if hasattr(os, "O_DIRECT") else None
    except OSError: self.dfd = None
    self.size = os.fstat(self.fd).st_size
  def close(self):
    os.close(self.fd)
    if self.dfd is not None: os.close(self.dfd)

slow_bytes = [0]  # (the bytes read through the page cache: the pieces not aligned)

def read_many(items:list[tuple[DirectFile, int, memoryview]]):
  """
  every (file, src, mv): the file's bytes src.. into mv. their pages O_DIRECT through the thread's ring (all of them in flight together) if
  mv's address is src modulo a page, the pieces of pages at their ends (or all of it, else) through the page cache
  """
  reqs, ring = [], _ring()
  for f, src, mv in items:
    ln, addr = len(mv), mv_address(mv)
    a0, a1 = round_up(src, PAGE), (src + ln) // PAGE * PAGE
    if f.dfd is None or (addr - src) % PAGE or a1 <= a0 or ring is None:
      _pread(f.fd, mv, src)
      slow_bytes[0] += ln
      continue
    if a0 > src: _pread(f.fd, mv[:a0 - src], src)
    if src + ln > a1: _pread(f.fd, mv[a1 - src:], a1)
    slow_bytes[0] += (a0 - src) + (src + ln - a1)
    reqs += [(f.dfd, o, addr + (o - src), min(READ_SZ, a1 - o)) for o in range(a0, a1, READ_SZ)]
  if reqs and ring is not None: ring.read(reqs)

def _pread(fd:int, mv:memoryview, offset:int):
  while len(mv):
    if (n:=os.preadv(fd, [mv], offset)) == 0: raise EOFError(f"short read at file offset {offset}")
    mv, offset = mv[n:], offset + n
