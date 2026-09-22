import ctypes, time, os, fcntl, typing, threading
from contextlib import contextmanager
from tinygrad.helpers import DEV
from tinygrad.runtime.support.system import FileIOInterface
from tinygrad.runtime.autogen import libc
from test.mockgpu.nv.nvdriver import NVDriver
from test.mockgpu.amd.amddriver import AMDDriver
from test.mockgpu.am.amdriver import AMDriver, AMUSBDriver
from test.mockgpu.qcom.qcomdriver import QCOMDriver
from test.mockgpu.ioctl import IoctlBridge
start = time.perf_counter()
original_memoryview = memoryview

class MockRuntime:
  def __init__(self, devices=None):
    devices = DEV.value if devices is None else devices
    classes = {"MOCKPCI+AMD": AMDriver, "MOCKKFD+AMD": AMDDriver, "MOCK+AMD": AMDDriver, "MOCKUSB+AMD": AMUSBDriver,
               "MOCK+NV": NVDriver, "MOCK+QCOM": QCOMDriver}
    self.drivers = [cls() for t in devices if (cls:=classes.get(f"{t.interface}+{t.device}"))]
    self.tracked_fds: dict[int, typing.Any] = {}
    self.ioctl_bridge = IoctlBridge(libc.dll.ioctl, self.tracked_fds)
    self._closed = False
    self._submission_thread: int|None = None

  @contextmanager
  def launch_scope(self):
    if self._closed: raise RuntimeError('MockRuntime is closed')
    owner = threading.get_ident()
    if self._submission_thread is not None:
      raise RuntimeError('MockRuntime submission already active')
    self._submission_thread = owner
    try: yield self
    finally: self._submission_thread = None

  def mmio_view(self, mem, nbytes=None):
    if isinstance(mem, int) or isinstance(mem, ctypes.Array):
      addr = ctypes.addressof(mem) if isinstance(mem, ctypes.Array) else mem
      for driver in self.drivers:
        for st, en, rcb, wcb in driver.tracked_addresses:
          if st <= addr and addr + (nbytes or 1) <= en:
            data = mem if not isinstance(mem, int) else (ctypes.c_ubyte * (nbytes or (en - addr))).from_address(addr)
            return TrackedMemoryView(data, rcb, wcb)
    return None

  def open(self, path, flags=os.O_RDONLY):
    return _open(path, flags, self)

  def close(self):
    if self._closed: return
    if self._submission_thread is not None: raise RuntimeError('cannot close active MockRuntime submission')
    for fd, file in list(self.tracked_fds.items()):
      file.close(fd)
      self.tracked_fds.pop(fd, None)
    self._closed = True

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.close()

runtime = MockRuntime()
drivers = runtime.drivers
tracked_fds = runtime.tracked_fds
ioctl_bridge = runtime.ioctl_bridge

class TrackedMemoryView:
  def __init__(self, data, rcb, wcb):
    self.mv = original_memoryview(data)
    self.rcb, self.wcb = rcb, wcb

  def __getitem__(self, index):
    self.rcb(self.mv, index)
    return self.mv[index]

  def __setitem__(self, index, value):
    self.mv[index] = value
    self.wcb(self.mv, index)

  def cast(self, new_type, **kwargs):
    self.mv = self.mv.cast('B').cast(new_type, **kwargs)
    return self

  @property
  def nbytes(self): return self.mv.nbytes
  def __len__(self): return len(self.mv)
  def __repr__(self): return repr(self.mv)

def _open(path, flags, owner=None):
  owner = runtime if owner is None else owner
  for d in owner.drivers:
    for x in d.tracked_files:
      if path == x.path:
        virtfd = d.open(path, flags, 0o777, x)
        owner.tracked_fds[virtfd.fd] = virtfd
        return virtfd.fd
  return os.open(path, flags, 0o777) if os.path.exists(path) else None

class MockFileIOInterface(FileIOInterface):
  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None, runtime=None):
    self.path = path
    self._runtime = runtime if runtime is not None else globals()['runtime']
    self.fd = fd if fd is not None else _open(path, flags, self._runtime)
    self._tracked_fds = self._runtime.tracked_fds
    self._virtual = self.fd in self._tracked_fds
    self._closed = False

  def close(self):
    if self._closed: return
    self._closed = True
    fd = self.fd
    if self._virtual:
      if fd in self._tracked_fds:
        self._tracked_fds[fd].close(fd)
        self._tracked_fds.pop(fd, None)
    else:
      os.close(fd)

  def __del__(self):
    self.close()

  def ioctl(self, request, arg):
    if self.fd in self._tracked_fds:
      return self._tracked_fds[self.fd].ioctl(self.fd, request, ctypes.addressof(arg))
    return fcntl.ioctl(self.fd, request, arg)

  @property
  def ioctl_function(self): return self._runtime.ioctl_bridge.callback_ref

  @property
  def mock_runtime(self): return self._runtime

  @property
  def mmio_view(self): return self._runtime.mmio_view

  def mmap(self, start, sz, prot, flags, offset):
    if self.fd in self._tracked_fds:
      return self._tracked_fds[self.fd].mmap(start, sz, prot, flags, self.fd, offset)
    return libc.mmap(start, sz, prot, flags, self.fd, offset)

  def read(self, size=None, binary=False, offset=None):
    if self.fd in self._tracked_fds:
      if offset is not None: self._tracked_fds[self.fd].seek(offset)
      return self._tracked_fds[self.fd].read_contents(size)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file:
      if file.tell() >= os.fstat(self.fd).st_size: file.seek(0)
      return file.read(size)

  def listdir(self):
    if self.fd in self._tracked_fds:
      return self._tracked_fds[self.fd].list_contents()
    return os.listdir(self.path)

  def write(self, content, binary=False, offset=None):
    if self.fd in self._tracked_fds:
      if offset is not None: self._tracked_fds[self.fd].seek(offset)
      return self._tracked_fds[self.fd].write_contents(content)
    raise NotImplementedError('MockFileIOInterface write is unsupported for native files')
  def seek(self, offset):
    if self.fd in self._tracked_fds:
      self._tracked_fds[self.fd].seek(offset)
    else:
      os.lseek(self.fd, offset, os.SEEK_CUR)
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset):
    return FileIOInterface._mmap(start, sz, prot, flags & ~0x4a000, -1, offset)  # strip MAP_LOCKED|MAP_POPULATE|MAP_HUGETLB
  @staticmethod
  def exists(path): return _open(path, os.O_RDONLY) is not None
  @staticmethod
  def readlink(path): raise NotImplementedError('MockFileIOInterface readlink is unsupported')
  @staticmethod
  def eventfd(initval, flags=None): raise NotImplementedError('MockFileIOInterface eventfd is unsupported')
