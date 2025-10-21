import subprocess
from tinygrad.runtime.autogen.autogen import Autogen

libc = Autogen("libc", "ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)",
  lambda: ([i for i in subprocess.check_output("dpkg -L libc6-dev".split()).decode().split() if 'sys/mman.h' in i or 'sys/syscall.h' in i] +
           ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]))

opencl = Autogen("opencl", "ctypes.CDLL(ctypes.util.find_library('OpenCL'))", ["/usr/include/CL/cl.h"])

cuda = Autogen("cuda", "ctypes.CDLL(ctypes.util.find_library('cuda'))", ["/usr/include/cuda.h"], args=["-D__CUDA_API_VERSION_INTERNAL"])

nvrtc = Autogen("nvrtc", "ctypes.CDLL(ctypes.util.find_library('nvrtc'))", ["/usr/local/cuda/include/nvrtc.h"])
nvjitlink = Autogen("nvjitlink", "ctypes.CDLL(ctypes.util.find_library('nvJitLink'))", ["/usr/local/cuda/include/nvJitLink.h"])

kfd = Autogen("kfd", None, ["/usr/include/linux/kfd_ioctl.h"])
