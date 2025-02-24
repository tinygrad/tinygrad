import ctypes, struct, platform, pathlib, os, binascii
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram, Device
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_cuda import cu_time_execution

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
nv_devs = {}

@ctypes.CFUNCTYPE(ctypes.c_int)
def dummy():
  print("**** dummy function hook ****")
  return cuda.CUDA_SUCCESS
  
  return -1

@ctypes.CFUNCTYPE(*([cuda.cuDriverGetVersion.restype] + cuda.cuDriverGetVersion.argtypes))
def cuDriverGetVersion(version):
  version.contents.value = 12040
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuInit.restype] + cuda.cuInit.argtypes))
def cuInit(flags):
  return hooked["cuInit"](flags)
  # print("call cuInit", flags, x)
  # return 0

@ctypes.CFUNCTYPE(*([cuda.cuGetExportTable.restype] + cuda.cuGetExportTable.argtypes))
def cuGetExportTable(ppExportTable, pExportTableId):
  return 0

@ctypes.CFUNCTYPE(*([cuda.cuDeviceGetCount.restype] + cuda.cuDeviceGetCount.argtypes))
def cuDeviceGetCount(count):
  count.contents.value = 1
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuDeviceGet.restype] + cuda.cuDeviceGet.argtypes))
def cuDeviceGet(device, ordinal):
  nv_devs[ordinal] = Device[f"NV:{ordinal}"]
  device.contents.value = ordinal
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuDeviceGetName.restype] + cuda.cuDeviceGetName.argtypes))
def cuDeviceGetName(name, len, dev):
  name.value = nv_devs[0].device.encode()
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuDeviceTotalMem_v2.restype] + cuda.cuDeviceTotalMem_v2.argtypes))
def cuDeviceTotalMem_v2(bts, dev):
  bts.contents.value = (22 << 30)
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuDeviceTotalMem_v2.restype] + cuda.cuDeviceTotalMem_v2.argtypes))
def cuDeviceGetAttribute(p, attr, dev):
  p.contents.value = 0
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuMemHostAlloc.restype] + cuda.cuMemHostAlloc.argtypes))
def cuMemHostAlloc(pp, bytesize, flags):
  print(f"cuMemHostAlloc {bytesize}")
  return hooked["cuMemHostAlloc"](pp, bytesize, flags)

@ctypes.CFUNCTYPE(*([cuda.cuModuleLoadData.restype] + cuda.cuModuleLoadData.argtypes))
def cuModuleLoadData(module, image):
  ret = hooked["cuModuleLoadData"](module, image)
  module_address = ctypes.addressof(module.contents.contents)
  print(f"cuModuleLoadData 0x{image:x} -> 0x{module_address:X}")
  seen_modules.add(module_address)

  # images, sections, relocs = elf_loader(bytes(to_mv(image, 0x100000)))
  # for s in sections: print(s)

  #print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))

  # hexdump(to_mv(image, 0x1000))
  image, sections, relocs = elf_loader(to_mv(image))
  print(sections)
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
  return 0

def create_hook(func_name, restype, argtypes):
  def hook_template(*args):
    print(func_name, flush=True)
    return 0 #hooked[func_name](*args)
  return ctypes.CFUNCTYPE(restype, *argtypes)(hook_template)

if __name__ == "__main__":
  ref = []
  blacklist = ['cuInit']
  for k,v in cuda.__dict__.items():
    if not k.startswith("cu"): continue
    if not hasattr(v, "restype"): continue
    if k in blacklist: continue
    # if not k.startswith("cuInit"): continue
    # print(k, v.restype, v.argtypes)
    ref.append(create_hook(k, v.restype, v.argtypes))
    hooked[k] = install_hook(v, ref[-1])
  
  #out = cuda.CUmoduleLoadingMode()
  #print(cuda.cuModuleGetLoadingMode(ctypes.byref(out)))
  #print(out.value)

  install_hook(cuda.cuDriverGetVersion, cuDriverGetVersion)
  install_hook(cuda.cuGetExportTable, cuGetExportTable)
  install_hook(cuda.cuDeviceGetCount, cuDeviceGetCount)
  install_hook(cuda.cuDeviceGet, cuDeviceGet)
  install_hook(cuda.cuDeviceGetName, cuDeviceGetName)
  install_hook(cuda.cuDeviceTotalMem_v2, cuDeviceTotalMem_v2)

  if getenv("TINYGRAD"):
    from tinygrad import Tensor
    (Tensor.zeros(6, device="CUDA").contiguous()*2).realize()
    exit(0)

  print("importing torch...")
  import torch
  print("torch", torch.__version__, torch.__file__)

  if getenv("RESNET"):
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = model.cuda()
    model.eval()

    if getenv("COMPILE"): model = torch.compile(model)

    X = torch.rand(getenv("BS", 1), 3, 288, 288, device='cuda')
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
    b = b.exp2()
    a += b
    #c = a @ b
    print("tensor math done", a.cpu().numpy())

  # confirm cuda library is right
  #maps = pathlib.Path("/proc/self/maps").read_text()
  #print('\n'.join([x for x in maps.split("\n") if 'cuda' in x or 'nv' in x]))
