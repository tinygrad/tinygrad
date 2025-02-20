import ctypes, struct, platform, pathlib, os
from hexdump import hexdump
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import libc

def install_hook(c_function, python_function):
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  if (processor:=platform.processor()) == "aarch64":
    # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
    # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
    # 0x0000000000000008:  00 02 1F D6    br  x16
    tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
    tramp += struct.pack("Q", python_function_addr)
  elif processor == "x86_64":
    # 0x0000000000000000:  49 BB aa aa aa aa aa aa aa aa    movabs r11, <address>
    # 0x000000000000000a:  41 FF E3                         jmp    r11
    tramp = b"\x49\xBB" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE3"
  else:
    raise Exception(f"processor {processor} not supported")

  # get real function address
  fxn_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))
  page_address = (fxn_address.contents.value//0x1000)*0x1000
  print(f"** hooking function at 0x{fxn_address.contents.value:X}")

  # hook function
  ret = libc.mprotect(page_address, 0x2000, 7)
  assert ret == 0
  libc.memcpy(fxn_address.contents.value, ctypes.create_string_buffer(tramp), len(tramp))
  ret = libc.mprotect(page_address, 0x2000, 5)
  assert ret == 0

@ctypes.CFUNCTYPE(ctypes.c_uint)
def hook(flags):
  print("called cuInit")
  return 0

if __name__ == "__main__":
  cuda = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libcuda.so.550.90.07")
  #install_hook(cuda.cuInit, hook)
  install_hook(cuda.cuLaunchKernel, hook)
  #install_hook(cuda.cuLaunchKernel_ptsz, hook)
  #install_hook(cuda.cuLaunchKernelEx, hook)
  #install_hook(cuda.cuLaunchKernelEx_ptsz, hook)

  print("importing torch...")
  import torch
  print("torch", torch.__version__, torch.__file__)

  #from extra.nv_gpu_driver import nv_ioctl

  maps = pathlib.Path("/proc/self/maps").read_text()
  print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))


  a = torch.zeros(10).cuda()
  print("tensor created")
  a += 1
  print("tensor math done", a.cpu().numpy())

  install_hook(cuda.cuLaunchKernel, hook)

