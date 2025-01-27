import ctypes, ctypes.util, platform
from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.helpers import OSX, cpu_time_execution, mv_address

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

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
