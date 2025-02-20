import ctypes, struct, platform, pathlib, os
from hexdump import hexdump
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram

def _hook(fxn_address_value, tramp):
  page_address = (fxn_address_value//0x1000)*0x1000
  ret = libc.mprotect(page_address, 0x2000, 7)
  assert ret == 0
  libc.memcpy(fxn_address_value, tramp, len(tramp))
  ret = libc.mprotect(page_address, 0x2000, 5)
  assert ret == 0
  CPUProgram.rt_lib["__clear_cache"](fxn_address_value, fxn_address_value + len(tramp))

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
  tramp = ctypes.create_string_buffer(tramp)

  # get real function address
  fxn_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))
  fxn_address_value = fxn_address.contents.value
  #print(f"** hooking function at 0x{fxn_address_value}")

  orig_save = (ctypes.c_char*len(tramp))()
  libc.memcpy(orig_save, fxn_address_value, len(tramp))
  _hook(fxn_address_value, tramp)

  def original(*args):
    _hook(fxn_address_value, orig_save)
    ret = c_function(*args)
    _hook(fxn_address_value, tramp)
    return ret
  return original

hooked = {}

function_names = {}

@ctypes.CFUNCTYPE(*([cuda.cuModuleGetFunction.restype] + cuda.cuModuleGetFunction.argtypes))
def cuModuleGetFunction(hfunc, hmod, name):
  ret = hooked["cuModuleGetFunction"](hfunc, hmod, name)
  python_name = ctypes.string_at(name).decode()
  print(f"called cuModuleGetFunction {python_name}")
  function_names[ctypes.addressof(hfunc.contents.contents)] = python_name
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuMemAlloc_v2.restype] + cuda.cuMemAlloc_v2.argtypes))
def cuMemAlloc_v2(dptr, bytesize):
  print(f"called cuMemAlloc_v2 {bytesize}")
  return hooked["cuMemAlloc_v2"](dptr, bytesize)

@ctypes.CFUNCTYPE(*([cuda.cuLaunchKernel.restype] + cuda.cuLaunchKernel.argtypes))
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra):
  name = function_names[ctypes.addressof(f.contents)]
  print(f"called cuLaunchKernel {name} <<{gridDimX}, {gridDimY}, {gridDimZ}>> <<{blockDimX}, {blockDimY}, {blockDimZ}>>")
  return hooked["cuLaunchKernel"](f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)

if __name__ == "__main__":
  hooked['cuModuleGetFunction'] = install_hook(cuda.cuModuleGetFunction, cuModuleGetFunction)
  hooked['cuMemAlloc_v2'] = install_hook(cuda.cuMemAlloc_v2, cuMemAlloc_v2)
  hooked['cuLaunchKernel'] = install_hook(cuda.cuLaunchKernel, cuLaunchKernel)

  print("importing torch...")
  import torch

  # confirm cuda library is right
  #print("torch", torch.__version__, torch.__file__)
  #maps = pathlib.Path("/proc/self/maps").read_text()
  #print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))

  a = torch.zeros(10).cuda()
  print("tensor created")
  a += 1
  print("tensor math done", a.cpu().numpy())

