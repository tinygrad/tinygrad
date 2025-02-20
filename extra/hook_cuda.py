import ctypes, struct, platform, pathlib, os, binascii
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram
from tinygrad.runtime.support.elf import elf_loader

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

allocated_memory = {}
function_names = {}

seen_modules = set()

@ctypes.CFUNCTYPE(ctypes.c_int)
def dummy():
  print("dummy function hook")
  return 0

@ctypes.CFUNCTYPE(*([cuda.cuModuleLoadData.restype] + cuda.cuModuleLoadData.argtypes))
def cuModuleLoadData(module, image):
  ret = hooked["cuModuleLoadData"](module, image)
  module_address = ctypes.addressof(module.contents.contents)
  print(f"cuModuleLoadData 0x{image:x} -> 0x{module_address:x}")
  seen_modules.add(module_address)
  #images, sections, relocs = elf_loader(bytes(to_mv(image, 0x100000)))
  #for s in sections: print(s)

  #print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))

  #hexdump(to_mv(image, 0x1000))
  #image, sections, relocs = elf_loader(to_mv(image))
  #print(sections)
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuModuleGetFunction.restype] + cuda.cuModuleGetFunction.argtypes))
def cuModuleGetFunction(hfunc, hmod, name):
  ret = hooked["cuModuleGetFunction"](hfunc, hmod, name)
  python_name = ctypes.string_at(name).decode()

  # pip install git+https://github.com/wbenny/pydemangler.git
  import pydemangler
  demangled_name = pydemangler.demangle(python_name)
  if demangled_name is not None: python_name = demangled_name

  print(f"called cuModuleGetFunction 0x{ctypes.addressof(hmod.contents):X} {python_name}")
  function_names[ctypes.addressof(hfunc.contents.contents)] = python_name
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuMemAlloc_v2.restype] + cuda.cuMemAlloc_v2.argtypes))
def cuMemAlloc_v2(dptr, bytesize):
  ret = hooked["cuMemAlloc_v2"](dptr, bytesize)
  cuda_address = ctypes.addressof(dptr.contents)
  allocated_memory[cuda_address] = bytesize
  print(f"cuMemAlloc_v2 {bytesize} 0x{cuda_address:X}")
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuLaunchKernel.restype] + cuda.cuLaunchKernel.argtypes))
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra):
  name = function_names[ctypes.addressof(f.contents)]
  print(f"cuLaunchKernel <<{gridDimX:4d}, {gridDimY:4d}, {gridDimZ:4d}>> <<{blockDimX:4d}, {blockDimY:4d}, {blockDimZ:4d}>> {sharedMemBytes} {name}")

  if getenv("PARAMS"):
    i = 0
    print(f"params @ 0x{ctypes.addressof(kernelParams.contents):X}")
    while True:
      paramOffset = ctypes.c_size_t()
      paramSize = ctypes.c_size_t()
      ret = cuda.cuFuncGetParamInfo(f, i, ctypes.byref(paramOffset), ctypes.byref(paramSize))
      if ret != 0: break
      dat = to_mv(kernelParams.contents, paramOffset.value+paramSize.value)[paramOffset.value:]
      print(f"{i}: offset:{paramOffset.value:3d} size:{paramSize.value:3d}") # --", binascii.hexlify(dat).decode())
      hexdump(dat)
      i += 1

  #print(f"params 0x{ctypes.addressof(kernelParams):X}")
  #hexdump(to_mv(kernelParams, 0x100))
  #print(f"data 0x{to_mv(kernelParams, 8).cast('Q')[0]:X}")
  #hexdump(to_mv(kernelParams.contents, 0x80))
  #for i,addr in enumerate(to_mv(kernelParams.contents, 0x100).cast("Q")): print(f"{i*8:3d}: {addr:X}")
  return hooked["cuLaunchKernel"](f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)

if __name__ == "__main__":
  hooked['cuModuleGetFunction'] = install_hook(cuda.cuModuleGetFunction, cuModuleGetFunction)
  hooked['cuMemAlloc_v2'] = install_hook(cuda.cuMemAlloc_v2, cuMemAlloc_v2)
  hooked['cuLaunchKernel'] = install_hook(cuda.cuLaunchKernel, cuLaunchKernel)

  # module loading + not used module loading
  # NOTE: CUDA 12 has libraries which are more complex
  hooked['cuModuleLoadData'] = install_hook(cuda.cuModuleLoadData, cuModuleLoadData)
  hooked['cuModuleLoad'] = install_hook(cuda.cuModuleLoad, dummy)
  hooked['cuModuleLoadDataEx'] = install_hook(cuda.cuModuleLoadDataEx, dummy)
  hooked['cuModuleLoadFatBinary'] = install_hook(cuda.cuModuleLoadFatBinary, dummy)

  #hooked['cuLibraryLoadData'] = install_hook(cuda.cuLibraryLoadData, dummy)

  #from tinygrad import Tensor
  #(Tensor.zeros(6, device="CUDA").contiguous()*2).realize()
  #exit(0)

  print("importing torch...")
  import torch
  print("torch", torch.__version__, torch.__file__)


  if getenv("RESNET"):
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = model.cuda()
    model.eval()

    X = torch.rand(1, 3, 288, 288, device='cuda')
    model(X)

    print("\n\n\n****** second run ******\n")
    model(X)
  else:
    a = torch.zeros(4, 4).cuda()
    b = torch.zeros(4, 4).cuda()
    print("tensor created")
    print(f"a: 0x{a.data_ptr():X}")
    print(f"b: 0x{b.data_ptr():X}")
    a += 1
    b += 2
    a = a.exp2()
    a += b
    c = a @ b
    print("tensor math done", a.cpu().numpy())

  # confirm cuda library is right
  #maps = pathlib.Path("/proc/self/maps").read_text()
  #print('\n'.join([x for x in maps.split("\n") if 'cuda' in x]))
