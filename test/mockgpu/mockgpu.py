import ctypes, ctypes.util, time, os, builtins
import fcntl
from tinygrad.runtime.support.hcq import HAL
from test.mockgpu.nv.nvdriver import NVDriver
from test.mockgpu.amd.amddriver import AMDDriver
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

drivers = [AMDDriver(), NVDriver()]
tracked_fds = {}

orignal_memoryview = builtins.memoryview
class TrackedMemoryView:
  def __init__(self, data, rcb, wcb):
    self.mv = orignal_memoryview(data)
    self.rcb, self.wcb = rcb, wcb

  def __getitem__(self, index):
    self.rcb(self.mv, index)
    return self.mv[index]

  def __setitem__(self, index, value):
    self.mv[index] = value
    self.wcb(self.mv, index)

  def cast(self, new_type, **kwargs):
    self.mv = self.mv.cast(new_type, **kwargs)
    return self

  @property
  def nbytes(self): return self.mv.nbytes
  def __len__(self): return len(self.mv)
  def __repr__(self): return repr(self.mv)

def _memoryview(cls, mem):
  if isinstance(mem, int) or isinstance(mem, ctypes.Array):
    addr = ctypes.addressof(mem) if isinstance(mem, ctypes.Array) else mem
    for d in drivers:
      for st,en,rcb,wcb in d.tracked_addresses:
        if st <= addr <= en: return TrackedMemoryView(mem, rcb, wcb)
  return orignal_memoryview(mem)
builtins.memoryview = type("memoryview", (), {'__new__': _memoryview}) # type: ignore

def _open(path, flags):
  for driver in drivers:
    for file in driver.tracked_files:
      if path == file.path:
        virtfd = driver.open(path, flags, 0o777, file)
        tracked_fds[virtfd.fd] = virtfd
        return virtfd.fd
  if os.path.exists(path):
    return os.open(path, flags, 0o777)
  else: return None

class MockHAL(HAL):
  def __init__(self, path:str, flags=os.O_RDONLY):
    self.fd = _open(path, flags)
    self.offset = 0

  def __del__(self):
    if self.fd in tracked_fds:
      tracked_fds[self.fd].close(self.fd)
      tracked_fds.pop(self.fd)
    else: os.close(self.fd)

  def ioctl(self, request, arg):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].ioctl(self.fd, request, ctypes.addressof(arg))
    return fcntl.ioctl(self.fd, request, arg)

  def mmap(self, start, sz, prot, flags, offset):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].mmap(start, sz, prot, flags, self.fd, offset)
    return libc.mmap(start, sz, prot, self.fd, offset)

  def read(self, size=None):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].read_text(size, self.offset)
    file = open(self.fd)
    file.seek(self.offset)
    return file.read(size)

  def readlink(self):
    if self.fd in tracked_fds: #NOTE is'nt used right now
      return tracked_fds[self.fd].readlink()
    return os.readlink(self.fd)

  def write(self, content):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].write_text(content)
    return open(self.fd).write(content)

  def listdir(self):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].listdir()
    return os.listdir(self.fd)

  def seek(self, offset): self.offset += offset
  def munmap(buf, sz): return libc.munmap(buf, sz)

  def exists(path):
    ret = _open(path, os.O_RDONLY)
    return True if ret else False

  def eventfd(initval, flags=None):
    fd = os.eventfd(initval, flags)
    ret = MockHAL.__new__(MockHAL)
    ret.fd = fd
    ret.offset = 0
    return ret
