from __future__ import annotations
import platform, ctypes, struct
from typing import Any
from tinygrad.helpers import OSX, unwrap, data64_le, cpu_profile
from tinygrad.device import Compiled, TinyELF, Program, HostAllocator, Buffer, BufferSpec
from tinygrad.runtime.support.c import DLL
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.support.system import RemotePCIDevice, RemoteCmd
from tinygrad.runtime.autogen import libc

class CPUProgram(Program['CPUDevice']):
  rt_lib, libm = DLL('rt', 'System' if OSX else 'gcc_s'), DLL('m', 'm')

  def _load(self, lib, base=0):
    return lib if lib[:4] != libc.ELFMAG.encode() else jit_loader(lib, base=base, link_libs=list(DLL._loaded_.values()),
      symbols={name:addr for library in self.libraries for name,addr in library.exports.items()}, exports=self.exports)

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.obj = obj
    self.dev, self.name, self.signature, self.profile_key = dev, obj.name, obj.signature, obj.profile_key
    self.lvp = obj.target.renderer == "LVP"
    self.exports:dict[str, int] = {}
    self.libraries = []
    for library in obj.libraries:
      if library.lib not in dev.libraries: dev.libraries[library.lib] = CPUProgram(dev, library)
      self.libraries.append(dev.libraries[library.lib])
    if dev.remote is not None:
      deps = struct.pack(f"<{len(self.libraries)}Q", *(lib.fxn for lib in self.libraries))
      self.fxn:Any = dev.remote.rpc(RemoteCmd.LOAD_PROG, len(obj.lib), len(self.libraries), payload=deps + obj.lib)[0]
    else:
      self.buf = Buffer(dev.device, len(obj.lib), options=BufferSpec(nolru=True), preallocate=True)
      loaded = self._load(obj.lib, self.buf._buf)
      self.buf.host[:len(loaded)] = loaded

      # RWX pages are unrepresentable on apple silicon (https://blog.svenpeter.dev/posts/m1_sprr_gxf/), so flip the written pages to R-X
      libc.mprotect(ctypes.c_void_p(self.buf._buf), len(obj.lib), libc.PROT_READ | libc.PROT_EXEC)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if 'rt' in DLL._loaded_: CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(self.buf._buf), ctypes.c_void_p(self.buf._buf + len(loaded)))
      else:
        # msync should be a universal POSIX way to do this
        libc.msync(ctypes.c_void_p(self.buf._buf), len(loaded), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.buf._buf)

  def __call__(self, *args:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               wait:bool=False, timeout:int|None=None) -> float|None:
    hdr = 12 if self.lvp else 0
    kernargs = ctypes.create_string_buffer(hdr + len(args) * 8)
    if self.lvp: struct.pack_into('<3I', kernargs, 0, *data64_le(ctypes.addressof(kernargs) + hdr), len(args) * 2)
    for a, (off, dt) in zip(args, TinyELF.iter_sig(self.signature)): struct.pack_into(f'<{dt.fmt}', kernargs, hdr + off, a)
    if (remote:=self.dev.remote) is not None:
      ret = (remote._rpc if wait else remote._post)(remote.sock, RemoteCmd.EXEC_PROG, self.fxn, len(kernargs), int(wait), payload=bytes(kernargs))
      return ret[0] / 1e9 if ret is not None else None
    with cpu_profile(self.name, self.dev.device, profile_key=self.profile_key) as prof: self.fxn(kernargs)
    return float(unwrap(prof.en) - prof.st) * 1e-6 if wait else None

class CPUDevice(Compiled):
  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    self.libraries:dict[bytes, CPUProgram] = {}
    self.remote = None
    if len(parts:=device.split(':')) == 3:
      self.remote = RemotePCIDevice("CPU", f"remote:{device[4:]}:0", RemotePCIDevice.connect(parts[1], int(parts[2])))
    super().__init__(device, HostAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram,
      arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")
    if self.remote is not None: self.peer_group = self.remote.peer_group

  def synchronize(self, timeout:int|None=None):
    if self.remote is not None: self.remote.rpc(RemoteCmd.PING)
    super().synchronize(timeout)
