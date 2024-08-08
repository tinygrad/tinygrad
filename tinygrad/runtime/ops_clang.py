import os, platform, tempfile, pathlib, subprocess, struct, ctypes, tinygrad.runtime.autogen.libc as libc
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.helpers import cpu_time_execution, cpu_objdump, OSX, DEBUG, getenv
from tinygrad.renderer.cstyle import ClangRenderer

if OSX: mac_libc = ctypes.CDLL(ctypes.util.find_library('c')) # needed for jit write protect and sys icache invalidate
MAP_JIT = 0x0800 if OSX else 0

def patchuint32(blob: memoryview, ploc: int, new: int): blob[ploc:ploc+4] = struct.pack("<I", struct.unpack("<I", blob[ploc:ploc+4])[0] | new)

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-include', 'tgmath.h', '-include', 'stdint.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror',
                              '-x', 'c', '-fPIC', '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangJITCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    args = ('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-Wall',
            '-Wno-unused-function', '-Wno-unused-command-line-argument', '-Werror', '-include', f'{os.path.dirname(__file__)}/support/tinymath.h',
            '-ffreestanding', '-nostdlib', '-ffixed-x18' if platform.machine() == "arm64" else '-Xclang=-fnative-half-type', '-', '-o', '-')
    image, _, relocs, exports = elf_loader(subprocess.check_output(args, input=src.encode('utf-8')), reserve=4)
    for ploc,tgt,r_type in [(a,b+d,c) for a,b,c,d in relocs]:
      tgt_pg, ploc_pg, rel = tgt >> 12, ploc >> 12, tgt - ploc
      if r_type in {libc.R_X86_64_PC32, libc.R_X86_64_PLT32}: patchuint32(image, ploc, 2**32+rel if rel < 0 else rel)
      elif r_type == libc.R_AARCH64_ADR_PREL_PG_HI21: patchuint32(image, ploc, ((tgt_pg-ploc_pg)&0b11)<<29 | ((tgt_pg-ploc_pg)>>2)<<5)
      elif r_type == libc.R_AARCH64_ADD_ABS_LO12_NC: patchuint32(image, ploc, (tgt&0xFFF)<<10)
      elif r_type in {libc.R_AARCH64_CALL26, libc.R_AARCH64_JUMP26}: patchuint32(image, ploc, 2**26+(rel>>2) if rel < 0 else (rel>>2))
      elif r_type == libc.R_AARCH64_LDST16_ABS_LO12_NC: patchuint32(image, ploc, (tgt&0xFFF)<<9)
      elif r_type == libc.R_AARCH64_LDST32_ABS_LO12_NC: patchuint32(image, ploc, (tgt&0xFFF)<<8)
      elif r_type == libc.R_AARCH64_LDST64_ABS_LO12_NC: patchuint32(image, ploc, (tgt&0xFFF)<<7)
      elif r_type == libc.R_AARCH64_LDST128_ABS_LO12_NC: patchuint32(image, ploc, (tgt&0xFFF)<<6)
      else: raise NotImplementedError(f"Encountered unknown relocation type {r_type:#x}")
    del exports['__truncdfhf2']
    assert len(exports) == 1, str(exports)
    _, entry = exports.popitem()
    if platform.machine() == 'arm64': image[:4] = struct.pack('<I', 0x14 << 24 | entry>>2)
    elif platform.machine() == 'x86_64': image[:5] = b'\xe9' + struct.pack('<I', entry-5)
    else: raise RuntimeError(f"Clang JIT doesn't support {platform.machine()}")
    return bytes(image)

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangJITProgram:
  def __init__(self, name:str, lib:bytes):
    self.map, self.mlen = libc.mmap(None, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0), len(lib)
    if OSX: mac_libc.pthread_jit_write_protect_np(False)
    ctypes.memmove(self.map, lib, len(lib))
    if OSX:
      mac_libc.pthread_jit_write_protect_np(True)
      mac_libc.sys_icache_invalidate(ctypes.c_void_p(self.map), ctypes.c_size_t(len(lib)))
    # TODO: clear instruction cache on linux
    self.fxn = ctypes.cast(self.map, ctypes.CFUNCTYPE(None))
  def __del__(self): libc.munmap(self.map, self.mlen)
  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # default arm abi requires stack slots to be 8 byte aligned, macos abi doesn't
    if platform.machine() == 'arm64' and OSX and len(args) > 8: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    c = ClangJITCompiler('compile_clang_object') if getenv('FASTCLANG', 1) else ClangCompiler('compile_clang')
    super().__init__(device, MallocAllocator, ClangRenderer(), c, ClangJITProgram if getenv('FASTCLANG', 1) else ClangProgram, ClangGraph)
