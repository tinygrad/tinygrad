import ctypes, fcntl, functools
from tinygrad.runtime.support.hcq import FileIOInterface

def _do_ioctl(idir, base, nr, struct, fd, **kwargs):
  ioctl = fd.ioctl if isinstance(fd, FileIOInterface) else functools.partial(fcntl.ioctl, fd)
  if (rc:=ioctl((idir<<30)|(ctypes.sizeof(out:=struct(**kwargs))<<16)|(base<<8)|nr, out)) != 0: raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, typ)

def CEnum(typ: type):
  class _CEnum(typ):
    _val_to_name_: dict[int,str] = {}

    @classmethod
    def from_param(cls, val): return val if isinstance(val, cls) else cls(val)
    @classmethod
    def name(cls, val): return cls._val_to_name_.get(val.value if isinstance(val, cls) else val, "unknown")
    @classmethod
    def define(cls, name, val):
      cls._val_to_name_[val] = name
      return cls(val)

    def __eq__(self, other): return self.value == other
    def __repr__(self): return self.name(self) if self.value in self.__class__._val_to_name_ else str(self.value)

  return _CEnum
