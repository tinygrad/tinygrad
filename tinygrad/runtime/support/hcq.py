from __future__ import annotations
from typing import Any
import ctypes, os
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import DEV, getenv, pluralize
from tinygrad.device import Compiled
from tinygrad.uop.ops import sint
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.memory import MMIOInterface as MMIOInterface, BumpAllocator as BumpAllocator

class FileIOInterface:
  """
  Hardware Abstraction Layer for HCQ devices. The class provides a unified interface for interacting with hardware devices.
  """

  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path:str = path
    self.fd:int = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'): os.close(self.fd)
  def ioctl(self, request, arg): return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file: return file.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as file: file.write(content)
  def listdir(self): return os.listdir(self.path)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def _mmap(start, sz, prot, flags, fd, offset):
    x = libc.mmap(start, sz, prot, flags, fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, -1, offset)
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None): return FileIOInterface(fd=os.eventfd(initval, flags))  # type: ignore[attr-defined]

if DEV.interface.startswith("MOCK"): from test.mockgpu.mockgpu import MockFileIOInterface as FileIOInterface  # noqa: F401 # pylint: disable=unused-import

# **************** for HCQ Compatible Devices ****************

def hcq_filter_visible_devices(devs, device):
  assert (v:=getenv("HCQ_VISIBLE_DEVICES", "")) == "", f"HCQ_VISIBLE_DEVICES={v} is deprecated, use DEV={DEV.target(device, indices=v)} instead"
  if '-' in (idstr:=DEV.target(device).indices): ids = list(range(int(idstr.split('-')[0]), int(idstr.split('-')[1])+1))
  else: ids = [int(x) for x in idstr.split(',') if x.strip()]
  assert all(x < len(devs) for x in ids), f"invalid visibility filter: {ids} ({pluralize('device', len(devs))} available)"
  return [devs[x] for x in ids] if ids else devs

class HCQBuffer:
  def __init__(self, va_addr:sint, size:int, meta:Any=None, _base:HCQBuffer|None=None, view:MMIOInterface|None=None, owner:Any=None):
    self.va_addr, self.size, self.meta, self._base, self.view = va_addr, size, meta, _base, view
    self._devs, self.owner = ([owner] if owner is not None else []), owner
    self._mappings:dict[Compiled, HCQBuffer] = {} # mapping to the other devices

  def offset(self, offset:int=0, size:int|None=None) -> HCQBuffer:
    return HCQBuffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def base(self) -> HCQBuffer: return self._base or self

  @property
  def mappings(self): return self._mappings if self._base is None else self._base._mappings

  @property
  def mapped_devs(self): return self._devs if self._base is None else self._base._devs
