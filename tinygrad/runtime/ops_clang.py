import platform, tempfile, pathlib, subprocess, struct, ctypes, functools, tinygrad.runtime.autogen.libc as libc
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.helpers import cpu_time_execution, cpu_objdump, OSX, DEBUG, FASTCLANG
from tinygrad.renderer.cstyle import ClangRenderer

MAP_JIT = 0x0800 if OSX else 0
@functools.lru_cache(None)
def mac_libc(): return ctypes.CDLL(ctypes.util.find_library('c')) # needed for jit write protect and sys icache invalidate (pylint complains if gated)
def patchuint32(image: memoryview, ploc: int, new: int): image[ploc:ploc+4] = struct.pack("<I", struct.unpack("<I", image[ploc:ploc+4])[0] | new)

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes: return self.compile_jit(src) if FASTCLANG else self.compile_shared(src)
  def compile_jit(self, src:str):
    args = ('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-Wall', '-Werror',
            '-fno-math-errno', '-include', 'stdint.h', '-ffreestanding', '-nostdlib') + (('-ffixed-x18',) if platform.machine() == "arm64" else ())+\
           ('-DINFINITY=__builtin_inff()', '-DNAN=__builtin_nanf("")')
    image, _, relocs, exports = elf_loader(subprocess.check_output(args + ('-', '-o', '-'), input=src.encode('utf-8')), reserve=8)
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
    assert len(exports) == 1, str(exports)
    _, entry = exports.popitem()
    if platform.machine() == 'arm64': image[:4] = struct.pack('<I', 0x14 << 24 | entry>>2)
    elif platform.machine() == 'x86_64': image[:5] = b'\xe9' + struct.pack('<I', entry-5)
    else: raise RuntimeError(f"Clang JIT doesn't support {platform.machine()}")
    return bytes(image)
  def compile_shared(self, src:str):
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-include', 'tgmath.h', '-include', 'stdint.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror',
                              '-x', 'c', '-fPIC', '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6 and not FASTCLANG: cpu_objdump(lib) # cpu_objdump can't disassemble flat binary
    self.fxn = self.load_jit(lib) if FASTCLANG else self.load_shared(lib, name)
  def load_jit(self, shellcode: bytes):
    mem = libc.mmap(None, len(shellcode), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0)
    if OSX: mac_libc().pthread_jit_write_protect_np(False)
    ctypes.memmove(mem, shellcode, len(shellcode))
    if OSX:
      mac_libc().pthread_jit_write_protect_np(True)
      mac_libc().sys_icache_invalidate(ctypes.c_void_p(mem), ctypes.c_size_t(len(shellcode)))
    return ctypes.cast(mem, ctypes.CFUNCTYPE(None))
  def load_shared(self, lib: bytes, name: str):
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      return ctypes.CDLL(str(cached_file_path.name))[name]
  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    if FASTCLANG and platform.machine() == 'arm64' and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(f"compile_clang{'_jit' if FASTCLANG else ''}"), ClangProgram, ClangGraph)
