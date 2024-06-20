from __future__ import annotations
import os, mmap, _posixshmem, io, ctypes, ctypes.util
from typing import Optional
from tinygrad.helpers import OSX, round_up, to_mv
from tinygrad.device import Compiled, Allocator
import tinygrad.runtime.autogen.uring as io_uring
import tinygrad.runtime.autogen.aio as aio

# libc = ctypes.CDLL(ctypes.util.find_library("c"))
# libc.lseek.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
# libc.lseek.restype = ctypes.c_longlong
# libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
# libc.read.restype = ctypes.c_longlong

# print(libc.aio_read)

def check(status): assert status == 0

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert self.device.mem is not None, "DiskBuffer wasn't opened"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, device:DiskDevice): self.device = device
  def _alloc(self, size:int, options):
    self.device._might_open(size)
    return DiskBuffer(self.device, size)
  def _free(self, opaque, options): self.device._might_close()
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and hasattr(self.device, 'fd'):
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.device.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        fo.readinto(dest)
    else:
      dest[:] = src._buf()

  def _copyout_sharded(self, src, size, _get_free_buf, seg_len):
    cqe_t = ctypes.POINTER(io_uring.struct_io_uring_cqe)
    cqe = ctypes.POINTER(io_uring.struct_io_uring_cqe)()
    cqes = (cqe_t * 1)()
    fd_offset = src.offset - (minor_offset := src.offset % mmap.PAGESIZE)
    processed_reqs_cnt, copied_in, next_read_offset, reqs, total_copy_size = 0, 0, 0, [], round_up(size + minor_offset, mmap.PAGESIZE)
    while next_read_offset < total_copy_size or len(reqs) != processed_reqs_cnt:
      if next_read_offset < total_copy_size and (copy_batch := _get_free_buf()) is not None:
        sqe = io_uring.io_uring_get_sqe(ctypes.byref(self.device.io_uring))
        bytes_to_read = min(seg_len, total_copy_size - next_read_offset)
        # ctypes.memset(copy_batch[0], 0xff, bytes_to_read)

        sqe.contents.opcode, sqe.contents.fd, sqe.contents.off = io_uring.IORING_OP_READ, self.device.fd, fd_offset + next_read_offset
        sqe.contents.addr, sqe.contents.len, sqe.contents.user_data = copy_batch[0], bytes_to_read, len(reqs)
        submitted = io_uring.io_uring_submit(ctypes.byref(self.device.io_uring))
        assert submitted == 1

        reqs.append((copy_batch, copied_in, minor_offset, real_copy_size := min(bytes_to_read - minor_offset, size - copied_in)))
        next_read_offset += bytes_to_read
        copied_in += real_copy_size
        minor_offset = 0

      if io_uring.io_uring_peek_batch_cqe(ctypes.byref(self.device.io_uring), cqes, 1) == 1:
        cqe = cqes[0]

        # print("done")

        assert cqe.contents.res >= reqs[cqe.contents.user_data][3] + reqs[cqe.contents.user_data][2], f"read from disk failed"
        # print(cqe.contents.user_data, to_mv(reqs[cqe.contents.user_data][0][0], 0x1000)[0], hex(reqs[cqe.contents.user_data][0][0]), hex(reqs[cqe.contents.user_data][1]), hex(reqs[cqe.contents.user_data][2]), hex(reqs[cqe.contents.user_data][3]))
        yield reqs[cqe.contents.user_data]
        reqs[cqe.contents.user_data] = None
        self.device.io_uring.cq.khead[0] = self.device.io_uring.cq.khead[0] + 1 # advance
        processed_reqs_cnt += 1

  def offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)

class DiskDevice(Compiled):
  io_uring = None

  def __init__(self, device:str):
    if DiskDevice.io_uring is None:
      check(io_uring.io_uring_queue_init(4096, ctypes.byref(ring:=io_uring.struct_io_uring()), 0))
      DiskDevice.io_uring = ring

    self.size: Optional[int] = None
    self.count = 0
    super().__init__(device, DiskAllocator(self), None, None, None)
  def _might_open(self, size):
    self.count += 1
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None: return
    filename = self.dname[len("disk:"):]
    self.size = size

    if filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, self.size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|(0 if OSX else os.O_DIRECT))
      except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      if os.fstat(self.fd).st_size < self.size: os.ftruncate(self.fd, self.size)
      self.mem = mmap.mmap(self.fd, self.size)
    if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None: self.mem.madvise(hp) # type: ignore
  def _might_close(self):
    self.count -= 1
    if self.count == 0:
      if hasattr(self, 'fd'): os.close(self.fd)
      self.size = None
