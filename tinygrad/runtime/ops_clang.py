import platform, struct, ctypes, subprocess, pathlib, tempfile, tinygrad.runtime.autogen.libc as libc
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.helpers import OSX, getenv, cpu_time_execution, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

MAP_JIT = 0x0800
global_handle = ctypes.CDLL(None)
def getbits(value: int, start: int, end: int): return (value >> start) & ((1 << end-start+1) - 1)
def patchuint32(image: memoryview, ploc: int, new: int): image[ploc:ploc+4] = struct.pack("<I", struct.unpack("<I", image[ploc:ploc+4])[0] | new)

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:list[str]|None=None, objdump_tool='objdump'):
    self.args = ['-march=native'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)

class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm
    args = ['-march=native', f'--target={platform.machine()}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
    arch_args = ['-ffixed-x18'] if platform.machine() == 'arm64' else []
    obj = subprocess.check_output(['clang', '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    image, _, relocs = elf_loader(obj)
    for ploc,tgt,r_type,r_addend in relocs:
      # https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.95.pdf
      if r_type == libc.R_X86_64_PC32: image[ploc:ploc+4] = struct.pack('<i', tgt+r_addend-ploc)
      # https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst for definitions of relocations
      # https://www.scs.stanford.edu/~zyedidia/arm64/index.html for instruction encodings
      elif r_type == libc.R_AARCH64_ADR_PREL_PG_HI21:
        rel_pg = ((tgt+r_addend) & ~0xFFF) - (ploc & ~0xFFF)
        patchuint32(image, ploc, (getbits(rel_pg, 12, 13) << 29) | (getbits(rel_pg, 14, 31) << 5))
      elif r_type == libc.R_AARCH64_LDST16_ABS_LO12_NC: patchuint32(image, ploc, getbits(tgt+r_addend, 1, 11) << 10)
      elif r_type == libc.R_AARCH64_LDST64_ABS_LO12_NC: patchuint32(image, ploc, getbits(tgt+r_addend, 3, 11) << 10)
      elif r_type == libc.R_AARCH64_LDST128_ABS_LO12_NC: patchuint32(image, ploc, getbits(tgt+r_addend, 4, 11) << 10)
      else: raise NotImplementedError(f"Encountered unknown relocation type {r_type:#x}")
    return bytes(image)

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangJITProgram:
  def __init__(self, name:str, lib:bytes):
    mem = libc.mmap(None, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), -1, 0)
    if OSX:
      global_handle.pthread_jit_write_protect_np(False)
    ctypes.memmove(mem, lib, len(lib))
    if OSX:
      global_handle.pthread_jit_write_protect_np(True)
      global_handle.sys_icache_invalidate(ctypes.c_void_p(mem), len(lib))
    self.fxn = ctypes.cast(mem, ctypes.CFUNCTYPE(None))

  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # apple relaxes abi requirement for stack arguments to always be at least 8 byte aligned on arm64
    # https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
    if platform.machine() == "arm64" and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    compiler, runtime = (ClangJITCompiler(), ClangJITProgram) if getenv("FASTCLANG", 1) else (ClangCompiler(), ClangProgram)
    super().__init__(device, MallocAllocator, ClangRenderer(), compiler, runtime)
