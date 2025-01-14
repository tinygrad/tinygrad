import ctypes, ctypes.util, struct, platform, tempfile, pathlib, subprocess
from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.helpers import OSX, mv_address, cpu_time_execution, cpu_objdump
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf import elf_loader, relocate
from tinygrad.renderer.cstyle import ClangRenderer

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

# Used by ops_dsp.py
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
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    args = ['-march=native', f'--target={platform.machine()}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
    arch_args = ['-ffixed-x18'] if platform.machine() == 'arm64' else []
    obj = subprocess.check_output(['clang', '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    image, _, relocs = elf_loader(obj)
    # This is needed because we have an object file, not a .so that has all internal references (like loads of constants from .rodata) resolved.
    for ploc,tgt,r_type,r_addend in relocs:
      image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], ploc, tgt+r_addend, r_type))
    return bytes(image)

  def disassemble(self, lib):
    import capstone
    match platform.machine():
      case 'x86_64': cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
      case 'aarch64' | 'arm64': cs = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
      case machine: raise NotImplementedError(f"Capstone disassembly isn't supported for {machine}")
    for instr in cs.disasm(lib, 0):
      print(f"{instr.address:#08x}: {instr.mnemonic}\t{instr.op_str}")

# CPUProgram is a jit/shellcode program that can be just mmapped and jumped to
class CPUProgram:
  helper_handle = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'gcc_s'))

  def __init__(self, name:str, lib:bytes):
    # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
    # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
    self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)

    if OSX: CPUProgram.helper_handle.pthread_jit_write_protect_np(False)
    self.mem.write(lib)
    if OSX: CPUProgram.helper_handle.pthread_jit_write_protect_np(True)

    # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
    # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
    # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
    # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
    CPUProgram.helper_handle["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))

    self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # NOTE: replace this by --target={host's triple}-elf in clang args once we only support macos sequoia and later.
    # Apple relaxes abi requirement for stack arguments to always be at least 8 byte aligned on arm64
    # https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
    # This hack is required because clang/llvm bug doesn't allow us to just use {host's triple}+'-elf' (relocation failures)
    # The bug was fixed in https://github.com/llvm/llvm-project/commit/454cc36630296262cdb6360b60f90a64a97f7f1a but was only backported to xcode 16+
    if platform.machine() == "arm64" and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangRenderer(), ClangJITCompiler(), CPUProgram)
