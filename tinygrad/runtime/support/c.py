import ctypes, functools, os, pathlib, re, sys, sysconfig, typing
from tinygrad.helpers import ceildiv, getenv, DEBUG, OSX, WIN
from _ctypes import _SimpleCData

def _do_ioctl(__idir, __base, __nr, __struct, __fd, *args, __payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import tinygrad.runtime.support.hcq as hcq, fcntl
  ioctl = __fd.ioctl if isinstance(__fd, hcq.FileIOInterface) else functools.partial(fcntl.ioctl, __fd)
  if (rc:=ioctl((__idir<<30)|(ctypes.sizeof(out:=(__payload or __struct(*args, **kwargs)))<<16)|(__base<<8)|__nr, out)):
    raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, typ)

def CEnum(typ: type[ctypes._SimpleCData]):
  class _CEnum(typ): # type: ignore
    _val_to_name_: dict[int,str] = {}

    @classmethod
    def from_param(cls, val): return val if isinstance(val, cls) else cls(val)
    @classmethod
    def get(cls, val, default="unknown"): return cls._val_to_name_.get(val.value if isinstance(val, cls) else val, default)
    @classmethod
    def items(cls): return cls._val_to_name_.items()
    @classmethod
    def define(cls, name, val):
      cls._val_to_name_[val] = name
      return val

    def __eq__(self, other): return self.value == other
    def __repr__(self): return self.get(self) if self.value in self.__class__._val_to_name_ else str(self.value)
    def __hash__(self): return hash(self.value)

  return _CEnum

_pending_records = []

def i2b(i:int, sz:int) -> bytes: return i.to_bytes(sz, sys.byteorder)
def b2i(b:bytes) -> int: return int.from_bytes(b, sys.byteorder)
def mv(st) -> memoryview: return memoryview(st).cast('B')

def record(cls):
  def __init__(self, *args, **kwargs):
    ctypes.Structure.__init__(self)
    for f,v in [*zip(self._real_fields_, args), *kwargs.items()]: setattr(self, f, v)
  struct = type(cls.__name__, (ctypes.Structure,), {'__init__':__init__, '_fields_': [('_mem_', ctypes.c_byte * cls.SIZE)],
                                                    '_real_fields_':tuple(cls.__annotations__.keys())})
  _pending_records.append((cls, struct, sys._getframe().f_back.f_globals))
  return struct

def init_records():
  for cls, struct, ns in _pending_records:
    for nm, t in typing.get_type_hints(cls, globalns=ns, include_extras=True).items(): setattr(struct, nm, field(t.__origin__, *t.__metadata__))
  _pending_records.clear()

def field(typ, off:int, bit_width=None, bit_off=0):
  if bit_width is not None:
    sl, set_mask = slice(off,off+(sz:=ceildiv(bit_width+bit_off, 8))), ~((mask:=(1 << bit_width) - 1) << bit_off)
    # FIXME: signedness
    return property(lambda self: (b2i(mv(self)[sl]) >> bit_off) & mask,
                    lambda self,v: mv(self).__setitem__(sl, i2b((b2i(mv(self)[sl]) & set_mask) | (v << bit_off), sz)))

  sl = slice(off, off + ctypes.sizeof(typ))
  return property(lambda self: v.value if isinstance(v:=typ.from_buffer(mv(self)[sl]), _SimpleCData) else v,
                  lambda self, v: mv(self).__setitem__(sl, bytes(v if isinstance(v, typ) else typ(v))))

class DLL(ctypes.CDLL):
  @staticmethod
  def findlib(nm:str, paths:list[str], extra_paths=[]):
    if nm == 'libc' and OSX: return '/usr/lib/libc.dylib'
    if pathlib.Path(path:=getenv(nm.replace('-', '_').upper()+"_PATH", '')).is_file(): return path
    for p in paths:
      libpaths = {"posix": ["/usr/lib64", "/usr/lib", "/usr/local/lib"], "nt": os.environ['PATH'].split(os.pathsep),
                  "darwin": ["/opt/homebrew/lib", f"/System/Library/Frameworks/{p}.framework"],
                  'linux': ['/lib', '/lib64', f"/lib/{sysconfig.get_config_var('MULTIARCH')}", "/usr/lib/wsl/lib/"]}
      if (pth:=pathlib.Path(p)).is_absolute():
        if pth.is_file(): return p
        else: continue
      for pre in (pathlib.Path(pre) for pre in ([path] if path else []) + libpaths.get(os.name, []) + libpaths.get(sys.platform, []) + extra_paths):
        if not pre.is_dir(): continue
        if WIN or OSX:
          for base in ([f"lib{p}.dylib", f"{p}.dylib", str(p)] if OSX else [f"{p}.dll"]):
            if (l:=pre / base).is_file() or (OSX and 'framework' in str(l) and l.is_symlink()): return str(l)
        else:
          for l in (l for l in pre.iterdir() if l.is_file() and re.fullmatch(f"lib{p}\\.so\\.?[0-9]*", l.name)):
            # filter out linker scripts
            with open(l, 'rb') as f:
              if f.read(4) == b'\x7FELF': return str(l)

  def __init__(self, nm:str, paths:str|list[str], extra_paths=[], emsg="", **kwargs):
    self.nm, self.emsg, self.loaded = nm, emsg, False
    if (path:= DLL.findlib(nm, paths if isinstance(paths, list) else [paths], extra_paths if isinstance(extra_paths, list) else [extra_paths])):
      if DEBUG >= 3: print(f"loading {nm} from {path}")
      try:
        super().__init__(path, **kwargs)
        self.loaded = True
      except OSError as e:
        self.emsg = str(e)
        if DEBUG >= 3: print(f"loading {nm} failed: {e}")
    elif DEBUG >= 3: print(f"loading {nm} failed: not found on system")

  @functools.cache
  def _get_func(self, name:str, args:tuple, res):
    (fn:=getattr(self, name)).argtypes, fn.restype = args, res
    return fn

  def bind(self, fn):
    restype, argtypes = None if (rt:=(hints:=typing.get_type_hints(fn)).pop('return', None)) is type(None) else rt, tuple(hints.values())
    return lambda *args: self._get_func(fn.__name__, argtypes, restype)(*args)

  def __getattr__(self, nm):
    if not self.loaded: raise AttributeError(f"failed to load library {self.nm}: " + (self.emsg or f"try setting {self.nm.upper()+'_PATH'}?"))
    return super().__getattr__(nm)
