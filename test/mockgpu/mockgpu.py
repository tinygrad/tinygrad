import ctypes, ctypes.util, time, os, builtins, fcntl
from tinygrad.runtime.support.hcq import HWInterface
from test.mockgpu.nv.nvdriver import NVDriver
from test.mockgpu.amd.amddriver import AMDDriver
from test.mockgpu.am.amdevice import AMDriver
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

drivers = [AMDriver()]
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
  for d in drivers:
    for x in d.tracked_files:
      if path == x.path:
        virtfd = d.open(path, flags, 0o777, x)
        tracked_fds[virtfd.fd] = virtfd
        return virtfd.fd
  return os.open(path, flags, 0o777) if os.path.exists(path) else None

class MockHWInterface(HWInterface):
  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path = path
    self.fd = fd or _open(path, flags)

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
    return libc.mmap(start, sz, prot, flags, self.fd, offset)

  def read(self, size=None, binary=False):
    if binary: raise NotImplementedError()
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].read_contents(size)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file:
      if file.tell() >= os.fstat(self.fd).st_size: file.seek(0)
      return file.read(size)

  def listdir(self):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].list_contents()
    return os.listdir(self.path)

  def write(self, content, binary=False):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].write(content)
    return os.write(self.fd, content)

  def seek(self, offset):
    if self.fd in tracked_fds:
      tracked_fds[self.fd].seek(offset)
    else:
      os.lseek(self.fd, offset, os.SEEK_CUR)
  @staticmethod
  def exists(path): return _open(path, os.O_RDONLY) is not None
  @staticmethod
  def readlink(path): raise NotImplementedError()
  @staticmethod
  def eventfd(initval, flags=None): NotImplementedError()
  @staticmethod
  def pci_scan(vendor_id, device_id):
    pci_devices = {}
    for d in drivers:
      for pcidev in d.pci_devs:
        if pcidev.vendor == vendor_id and pcidev.device == device_id:
          pci_devices[f"{pcidev.domain:04x}:{pcidev.bus:02x}:{pcidev.slot:02x}.{pcidev.func:d}"] = pcidev
    return pci_devices
  @staticmethod
  def pci_probe(pci_device) -> int:
    pci_device.driver.probe(pci_device)
    return 0
  @staticmethod
  def pci_enable(pci_device) -> int: 
    pci_device.driver.enable(pci_device)
    return 0
  @staticmethod
  def pci_cfg_read(pci_device, cmd) -> int:
    return pci_device.driver.cfg_read(pci_device, cmd)
  @staticmethod
  def pci_cfg_write(pci_device, cmd, value) -> int:
    return pci_device.driver.cfg_write(pci_device, cmd, value)
