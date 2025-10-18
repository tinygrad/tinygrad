import subprocess
from tinygrad.runtime.autogen.autogen import Autogen

libc = Autogen("libc", "None if (libc_path:=ctypes.util.find_library('c')) is None else ctypes.CDLL(libc_path, use_errno=True)",
  lambda: ([i for i in subprocess.check_output("dpkg -L libc6-dev".split()).decode().split() if 'sys/mman.h' in i or 'sys/syscall.h' in i] +
           ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]))
