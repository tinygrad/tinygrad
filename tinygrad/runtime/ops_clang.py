from typing import Optional, List
import platform, struct, ctypes, subprocess, pathlib, tempfile, tinygrad.runtime.autogen.libc as libc
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG, OSX, getenv, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

MAP_JIT, global_handle = 0x0800 if OSX else 0, ctypes.CDLL(None)
def i2u(bits:int, x:int): return 2**bits+x if x < 0 else x
def patchuint32(image: memoryview, ploc: int, new: int): image[ploc:ploc+4] = struct.pack("<I", struct.unpack("<I", image[ploc:ploc+4])[0] | new)
SIMPLE_AARCH64_RELOCATIONS = [libc.R_AARCH64_ADD_ABS_LO12_NC, libc.R_AARCH64_LDST16_ABS_LO12_NC, libc.R_AARCH64_LDST32_ABS_LO12_NC,
                              libc.R_AARCH64_LDST64_ABS_LO12_NC, libc.R_AARCH64_LDST128_ABS_LO12_NC]

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-shared', '-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangJITCompiler(ClangCompiler):
  def __init__(self, key="compile_clang_jit"):
    # x18 is reserved register on at least macOS and Windows and is clobbered on context switch
    super().__init__(key, ['-c', '-march=native', f'--target={platform.machine()}-none-unknown-elf', '-fno-math-errno',
                     *(['-ffixed-x18'] if platform.machine() == 'arm64' else [])])
  def compile(self, src:str) -> bytes:
    image, _, relocs = elf_loader(super().compile(src))
    for ploc,tgt,r_type in [(pl,tgt+r_addend,r_type) for pl,tgt,r_type,r_addend in relocs]:
      rel_pg, rel = (tgt >> 12) - (ploc >> 12), tgt - ploc
      if r_type == libc.R_X86_64_PC32: patchuint32(image, ploc, i2u(32, rel))
      elif r_type == libc.R_AARCH64_ADR_PREL_PG_HI21: patchuint32(image, ploc, (rel_pg&0b11)<<29 | (rel_pg>>2)<<5)
      elif r_type in {libc.R_AARCH64_CALL26, libc.R_AARCH64_JUMP26}: patchuint32(image, ploc, i2u(26, rel>>2))
      elif r_type in SIMPLE_AARCH64_RELOCATIONS: patchuint32(image, ploc, (tgt&0xFFF)<<(10-SIMPLE_AARCH64_RELOCATIONS.index(r_type)))
      else: raise NotImplementedError(f"Encountered unknown relocation type {r_type:#x}")
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
    mem = libc.mmap(None, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0)
    if OSX:
      global_handle.pthread_jit_write_protect_np(False)
    ctypes.memmove(mem, lib, len(lib))
    if OSX:
      global_handle.pthread_jit_write_protect_np(True)
      global_handle.sys_icache_invalidate(ctypes.c_void_p(mem), ctypes.c_size_t(len(lib))) # instruction cache invalidation isn't needed on x86
    self.fxn = ctypes.cast(mem, ctypes.CFUNCTYPE(None))
  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # apple relaxes abi requirement for stack arguments to always be at least 8 byte aligned on arm64
    # https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
    if platform.machine() == 'arm64' and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(),
                     *((ClangJITCompiler(), ClangJITProgram) if getenv("FASTCLANG", 1) else (ClangCompiler(), ClangProgram)), ClangGraph)
