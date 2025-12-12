import ctypes, functools, os, pathlib, re, struct, sys, sysconfig
from tinygrad.helpers import ceildiv, getenv, to_mv, DEBUG, OSX, WIN

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

  def __getattr__(self, nm):
    if not self.loaded: raise AttributeError(f"failed to load library {self.nm}: " + (self.emsg or f"try setting {self.nm.upper()+'_PATH'}?"))
    return super().__getattr__(nm)

class _CBuffer:
  __slots__, SIZE = ('_mem_',), 0

  def __init__(self, src=None):
    if src is None: self._mem_ = memoryview(bytearray(self.SIZE))
    else: self._mem_ = src if isinstance(src, memoryview) else (to_mv(src, self.SIZE) if isinstance(src, int) else memoryview(src))

  @property
  def _as_parameter_(self): return self._mem_

  @classmethod
  def from_param(cls, obj):
    if obj is None: return None
    assert isinstance(obj, cls)
    return type('_shadow', (ctypes.Structure,), {'_fields_': [('a', ctypes.c_char * cls.SIZE)]}).from_buffer(obj._mem_)

class Union(_CBuffer): pass

class Struct(_CBuffer):
  __slots__ = ('_fields_',)
  def __init__(self, *args, **kwargs):
    super().__init__()
    for f,v in [*zip(self._fields_, args), *kwargs.items()]: setattr(self, f, v)

  @staticmethod
  def from_buf(cls, src):
    ret = super().__init__(src)
    super(Struct, ret).__init__(src)
    return ret

@functools.cache
def Array(typ, cnt):
  if (prim:=isinstance(typ, str)): stride, s = struct.calcsize(typ), struct.Struct(typ)
  else: stride = typ.SIZE

  def chkidx(idx):
    if idx >= cnt or idx <= -cnt: raise IndexError(f"{idx} out of range")
    return cnt + idx if idx < 0 else idx

  class _Array(_CBuffer):
    SIZE = stride * cnt

    def __init__(self, src=None):
      if isinstance(src, (list, tuple)):
        super().__init__()
        self[:] = src
      else: super().__init__(src)

    def __len__(self): return cnt

    def __getitem__(self, idx):
      offset = chkidx(idx) * stride
      return s.unpack_from(self._mem_, offset)[0] if prim else typ(src=self._mem_[offset:offset+stride])

    def __setitem__(self, idx, v):
      if isinstance(idx, slice):
        # TODO: if prim and step == 1, we can struct.pack_into everything at once
        for i, v in zip(range(*idx.indices(cnt)), v): self[i] = v
      offset = chkidx(idx) * stride
      if prim: s.pack_into(self._mem_, offset, v)
      else: self._mem_[offset:offset+stride] = bytes(v)

    def __iter__(self):
      for i in range(cnt): yield self[i]

  _Array.__name__ = f"Array_{typ if prim else typ.__name__}_Array_{cnt}"
  return _Array

def field(off, typ, bit_width=None, bit_off=0):
  if bit_width is not None:
    sl, set_mask = slice(off,off+(sz:=ceildiv(bit_width, 8))), ~((mask:=(1 << bit_width) - 1) << bit_off)
    # FIXME: signedness
    return property(lambda self: (int.from_bytes(self._mem_[sl], sys.byteorder) >> bit_off) & mask,
                    lambda self,v: self._mem_.__setitem__(sl, ((int.from_bytes(self._mem_[sl])&set_mask)|(v << bit_off)).to_bytes(sz, sys.byteorder)))

  if isinstance(typ, str):
    s = struct.Struct(typ)
    return property(lambda self: s.unpack_from(self._mem_, off)[0], lambda self, v: s.pack_into(self._mem_, off, v))

  return property(lambda self: typ(self._mem_[off:off+typ.SIZE]), lambda self, v: self._mem_.__setitem__(slice(off,off+typ.SIZE), bytes(v)))
