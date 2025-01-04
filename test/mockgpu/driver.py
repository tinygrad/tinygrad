from typing import Any
from dataclasses import dataclass

class VirtFileDesc:
  def __init__(self, fd): self.fd, self.off = fd, 0
  def ioctl(self, fd, req, argp): raise NotImplementedError()
  def mmap(self, st, sz, prot, flags, fd, off): raise NotImplementedError()
  def read_contents(self, size=None, offset=0): raise NotImplementedError()
  def close(self, fd): return 0

class TextFileDesc(VirtFileDesc):
  def __init__(self, fd, text):
    super().__init__(fd)
    self.content = text

  def ioctl(self, fd, req, argp): return 0
  def read_contents(self, size=None, offset=0):
    if size is None: size = len(self.content)
    return self.content[offset:offset+size]
class DirFileDesc(VirtFileDesc):
  def __init__(self, fd, child_names):
    super().__init__(fd)
    self.child_names = child_names

  def ioctl(self, fd, req, argp): return 0

  def list_contents(self):
    return self.child_names

@dataclass(frozen=True)
class VirtFile:
  path: str
  fdcls: Any # TODO: fix this Union[VirtFileDesc, functools.partial[VirtFileDesc]]

class VirtDriver:
  def __init__(self):
    self.tracked_files = []
    self.tracked_addresses = []
  def track_address(self, staddr, enaddr, rcb, wcb): self.tracked_addresses.append((staddr, enaddr, rcb, wcb))
  def open(self, name, flags, mode, fdcls): raise NotImplementedError()
