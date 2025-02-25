import ctypes, struct, platform, pathlib, os, binascii
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram, Device
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_cuda import cu_time_execution
from tinygrad.runtime.support.hcq import HCQBuffer

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
nv_allocs = {}

nv_next_module = 1
nv_modules = {}
nv_funcs = {}

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
  # return hooked["cuInit"](flags)
  # print("call cuInit", flags, x)
  return 0

# @ctypes.CFUNCTYPE(*([cuda.cuGetExportTable.restype] + cuda.cuGetExportTable.argtypes))
# def cuGetExportTable(ppExportTable, pExportTableId):
#   return 0

# @ctypes.CFUNCTYPE(*([cuda.cuDeviceGetCount.restype] + cuda.cuDeviceGetCount.argtypes))
# def cuDeviceGetCount(count):
#   count.contents.value = 1
#   return cuda.CUDA_SUCCESS

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

@ctypes.CFUNCTYPE(*([cuda.cuDeviceGetAttribute.restype] + cuda.cuDeviceGetAttribute.argtypes))
def cuDeviceGetAttribute(p, attr, dev):
  ret = hooked["cuDeviceGetAttribute"](p, attr, dev)
  # print(attr, p.contents.value)
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuMemAlloc_v2.restype] + cuda.cuMemAlloc_v2.argtypes))
def cuMemAlloc_v2(dptr, bytesize):
  # ret = hooked["cuMemAlloc_v2"](dptr, bytesize)
  # cuda_address = ctypes.addressof(dptr.contents)
  # allocated_memory[cuda_address] = bytesize
  hcq_buf = nv_devs[0].allocator.alloc(bytesize)
  nv_allocs[hcq_buf.va_addr] = hcq_buf
  dptr.contents.value = hcq_buf.va_addr
  print(f"(hcq) cuMemAlloc_v2 {bytesize} 0x{hcq_buf.va_addr:X}")
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuMemHostAlloc.restype] + cuda.cuMemHostAlloc.argtypes))
def cuMemHostAlloc(dptr, bytesize, flags):
  hcq_buf = nv_devs[0].allocator.alloc(bytesize)
  nv_allocs[hcq_buf.va_addr] = hcq_buf
  dptr.contents.value = hcq_buf.va_addr
  print(f"(hcq) cuMemHostAlloc {bytesize} 0x{hcq_buf.va_addr:X}")
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuMemcpyHtoDAsync_v2.restype] + cuda.cuMemcpyHtoDAsync_v2.argtypes))
def cuMemcpyHtoDAsync_v2(dst, src, bytesize, hStream):
  nv_devs[0].allocator._copyin(HCQBuffer(dst, bytesize), to_mv(src, bytesize))
  print(f"(hcq) cuMemcpyHtoDAsync_v2 0x{src:X} -> 0x{dst:X} {bytesize}")
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuMemcpyDtoH_v2.restype] + cuda.cuMemcpyDtoH_v2.argtypes))
def cuMemcpyDtoHAsync_v2(dst, src, bytesize):
  nv_devs[0].allocator._copyout(to_mv(dst, bytesize), HCQBuffer(src, bytesize))
  print(f"(hcq) cuMemcpyDtoHAsync_v2 0x{src:X} -> 0x{dst:X} {bytesize}")
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuStreamSynchronize.restype] + cuda.cuStreamSynchronize.argtypes))
def cuStreamSynchronize(stream):
  nv_devs[0].synchronize()
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuModuleLoadData.restype] + cuda.cuModuleLoadData.argtypes))
def cuModuleLoadData(module, image):
  # ret = hooked["cuModuleLoadData"](module, image)
  # module_address = ctypes.addressof(module.contents)
  # print(f"cuModuleLoadData 0x{image:x} -> 0x{module_address:X}")
  # seen_modules.add(module_address)

  # images, sections, relocs = elf_loader(bytes(to_mv(image, 0x100000)))
  # for s in sections: print(s)

  #print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))

  # hexdump(to_mv(image, 0x1000))
  # image, sections, relocs = elf_loader(bytes(to_mv(image, 0x100000)))
  # print(image, sections, relocs)
  global nv_next_module
  nv_modules[nv_next_module] = bytes(to_mv(image, 8 << 10))
  module.contents = ctypes.c_void_p(nv_next_module)
  nv_next_module += 1
  # print(f"cuModuleLoadData 0x{ctypes.cast(module, ctypes.c_void_p).value:X}")
  return 0

@ctypes.CFUNCTYPE(*([cuda.cuModuleGetFunction.restype] + cuda.cuModuleGetFunction.argtypes))
def cuModuleGetFunction(hfunc, hmod, name):
  global nv_next_module
  # print(ctypes.cast(hmod, ctypes.c_void_p).value)
  # hexdump(to_mv(hmod, 0x100))
  python_name = ctypes.string_at(name).decode()
  print(python_name)

  nv_funcs[nv_next_module] = nv_devs[0].runtime(python_name, nv_modules[1])
  hfunc.contents = ctypes.c_void_p(nv_next_module)
  nv_next_module += 1
  return 0


  # ret = hooked["cuModuleGetFunction"](hfunc, hmod, name)
  # python_name = ctypes.string_at(name).decode()

  # # pip install git+https://github.com/wbenny/pydemangler.git
  # import pydemangler
  # demangled_name = pydemangler.demangle(python_name)
  # if demangled_name is not None: python_name = demangled_name

  # print(f"called cuModuleGetFunction 0x{ctypes.addressof(hmod.contents):X} {python_name}")
  # function_names[ctypes.addressof(hfunc.contents.contents)] = python_name
  # return ret
  # return 1

# @ctypes.CFUNCTYPE(*([cuda.cuDevicePrimaryCtxGetState.restype] + cuda.cuDevicePrimaryCtxGetState.argtypes))
# def cuDevicePrimaryCtxGetState(device, flags, active):
#   ret = hooked["cuDevicePrimaryCtxGetState"](device, flags, active)
#   print(f"cuDevicePrimaryCtxGetState {device.contents.value} {flags} {active.contents.value}")
#   return ret


# CUresult cuLibraryGetModule ( CUmodule* pMod, CUlibrary library )
@ctypes.CFUNCTYPE(cuda.CUresult, ctypes.POINTER(cuda.CUmodule), cuda.CUmodule)
def cuLibraryGetModule(phModule, hLib):
  print(hLib)
  print("cuLibraryGetModule", phModule, hLib)
  return 0

@ctypes.CFUNCTYPE(*([cuda.cuLaunchKernel.restype] + cuda.cuLaunchKernel.argtypes))
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra):
  dev = nv_devs[0]
  prg = nv_funcs[2]
  global_size = (gridDimX, gridDimY, gridDimZ)
  local_size = (blockDimX, blockDimY, blockDimZ)
  print(f"cuLaunchKernel {prg} {global_size} {local_size}", flush=True)

  kernargs = prg.fill_kernargs([], [])
  kk = to_mv(kernargs.ptr + 0x160, 4 * 8).cast('Q')
  
  hexdump(to_mv(kernelParams, 128))
  encoded_args = to_mv(kernelParams, 10*8).cast("Q")
  kk[0] = to_mv(encoded_args[0], 8).cast("Q")[0]
  kk[1] = to_mv(encoded_args[1], 16).cast("Q")[0]
  kk[2] = to_mv(encoded_args[1], 16).cast("Q")[1]
  kk[3] = to_mv(encoded_args[2], 8).cast("I")[0]
  # for i in range(10):
  #   if encoded_args[i] > 0x7f0000000000:
  #     kk[i] = to_mv(encoded_args[i], 8).cast("Q")[0]
  #     print(f"arg {i} 0x{kk[i]:X}")
  #   else: break
  # encoded_size = to_mv(encoded_args[0], 8).cast("Q")
  # print(hex(encoded_args[0]), hex(encoded_args[1]), hex(encoded_args[3]), flush=True)
  # ctypes.memmove(kernargs.ptr, encoded_args[1], encoded_size[0])

  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()

  q.exec(prg, kernargs, global_size, local_size)

  q.signal(dev.timeline_signal, dev.timeline_value).submit(dev)
  dev.timeline_value += 1
  return 0


def create_hook(func_name, restype, argtypes):
  def hook_template(*args):
    print(func_name, flush=True)
    return hooked[func_name](*args)
  return ctypes.CFUNCTYPE(restype, *argtypes)(hook_template)

if __name__ == "__main__":
  # print(getattr(cuda._libraries['libcuda.so'], 'cuCtxCreate'))
  # print(cuda._libraries['libcuda.so'].cuCtxCreate)
  # print(dir(cuda._libraries['libcuda.so']))

  ref = []
  # blacklist = ['cuInit', 'cuGetExportTable', 'cuDeviceGetCount', 'cuDeviceGetUuid', 'cuDeviceGetAttribute', 'cuDeviceGetCount']
  blacklist = ['cuDeviceGetAttribute']
  for k,v in cuda.__dict__.items():
    if not k.startswith("cu"): continue
    if not hasattr(v, "restype"): continue
    if k in blacklist: continue
    # if not k.startswith("cuInit"): continue
    # print(k, v.restype, v.argtypes)
    if k.endswith("_v2") and k[:-3] not in cuda.__dict__:
      print(f"hooking {k[:-3]}")
      ref.append(create_hook(k[:-3], v.restype, v.argtypes))
      hooked[k] = install_hook(getattr(cuda._libraries['libcuda.so'], k[:-3]), ref[-1])
    ref.append(create_hook(k, v.restype, v.argtypes))
    hooked[k] = install_hook(v, ref[-1])
  
  #out = cuda.CUmoduleLoadingMode()
  #print(cuda.cuModuleGetLoadingMode(ctypes.byref(out)))
  #print(out.value)

  # install_hook(cuda.cuDeviceGetCount, cuDeviceGetCount)
  install_hook(cuda.cuDriverGetVersion, cuDriverGetVersion)
  # install_hook(cuda.cuGetExportTable, cuGetExportTable)
  install_hook(cuda.cuDeviceGet, cuDeviceGet)
  install_hook(cuda.cuDeviceGetName, cuDeviceGetName)
  install_hook(cuda.cuDeviceTotalMem_v2, cuDeviceTotalMem_v2)
  install_hook(cuda.cuMemAlloc_v2, cuMemAlloc_v2)
  install_hook(cuda.cuMemHostAlloc, cuMemHostAlloc)
  install_hook(cuda.cuMemcpyHtoDAsync_v2, cuMemcpyHtoDAsync_v2)
  install_hook(cuda.cuModuleGetFunction, cuModuleGetFunction)
  install_hook(cuda.cuModuleLoadData, cuModuleLoadData)
  install_hook(cuda._libraries['libcuda.so'].cuLibraryGetModule, cuLibraryGetModule)
  install_hook(cuda.cuLaunchKernel, cuLaunchKernel)
  install_hook(cuda.cuMemcpyDtoHAsync_v2, cuMemcpyDtoHAsync_v2)
  install_hook(cuda.cuStreamSynchronize, cuStreamSynchronize)

  # hooked['cuDeviceGetAttribute'] = install_hook(cuda.cuDeviceGetAttribute, cuDeviceGetAttribute)

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
    a = torch.ones(4, 4).cuda()
    b = torch.zeros(4, 4).cuda()
    print("tensor created")
    print(f"a: 0x{a.data_ptr():X}")
    print(f"b: 0x{b.data_ptr():X}")
    # a += 1
    # b += 2
    a = a.exp2()
    # b = b.exp2()
    # a += b
    # c = a @ b
    print("tensor math done", a.cpu().numpy())

  # confirm cuda library is right
  #maps = pathlib.Path("/proc/self/maps").read_text()
  #print('\n'.join([x for x in maps.split("\n") if 'cuda' in x or 'nv' in x]))
