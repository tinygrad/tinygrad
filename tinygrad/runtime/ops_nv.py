from __future__ import annotations
import os, ctypes, pathlib, re, fcntl, functools, mmap, time
from typing import Tuple, Any
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct, errno
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up, to_char_p_p
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import to_mv, getenv, round_up
from extra.nv_gpu_driver import nv_ioctl
from extra.nv_gpu_driver import esc_ioctl as nvesc
from extra.nv_gpu_driver import class_ioctl as nvcls
from extra.nv_gpu_driver import ctrl_ioctl as nvctrl
from extra.nv_gpu_driver import uvm_ioctl as nvuvm
from extra.nv_gpu_driver import nv_qcmds as nvqcmd
from hexdump import hexdump
from tinygrad.runtime.ops_cuda import check as cuda_check, _get_bytes, CUDACompiler
import tinygrad.runtime.autogen.cuda as cuda

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

def _IOWR(type, nr, size):
  return (3 << 30) | (size & 0x1FFF) << 16 | (type & 0xFF) << 8 | (nr & 0xFF)

def rm_alloc(fd, clss, root, parant, params):
  made = nvesc.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def rm_control(fd, cmd, client, obj, params):
  made = nvesc.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, params=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)), paramsSize=ctypes.sizeof(params))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_CONTROL, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def uvm_ioctl(fd, cmd, params):
  ret = fcntl.ioctl(fd, cmd, params)
  if ret != 0: raise RuntimeError(f"ioctl (uvm_control) returned {ret}")
  if params.rmStatus != 0: raise RuntimeError(f"ioctl (uvm_control) returned {params.rmStatus}")

def set_bits_in_array(arr, end_bit, start_bit, value):
  for bt in range(start_bit, end_bit+1): arr[bt // 32] |= ((value >> (bt - start_bit)) & 0x1) << (bt % 32)

def cmd_compute(program_address, constant_address, constant_len, global_size, local_size):
  arr = (ctypes.c_uint32 * 0x40)()

  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_QMD_GROUP_ID, 0x3F)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE, 1)

  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE, nvqcmd.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_MEMBAR)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SAMPLER_INDEX, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE, 0x400)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE, 3)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1A)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_QMD_MAJOR_VERSION, 3)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_REGISTER_COUNT_V, 0x10)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE, 3)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_BARRIER_COUNT, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE, 0x640)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE, 0xa)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SASS_VERSION, 0x86)

  # group
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, global_size[0])
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, global_size[1])
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, global_size[2])
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0, local_size[0])
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1, local_size[1])
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2, local_size[2])

  # program
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, program_address)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, program_address>>32)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, program_address>>8)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, program_address>>40)

  # args
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0), constant_address>>32)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0), constant_address)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0), constant_len>>4)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE_TRUE)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)
  return arr

def cmdq_push_data(dev, data):
  dev.cmdq[dev.cmdq_wptr//4] = data
  dev.cmdq_wptr += 4
def cmdq_push_data64(dev, data):
  cmdq_push_data(dev, data >> 32)
  cmdq_push_data(dev, data & 0xFFFFFFFF)
def cmdq_push_data64_le(dev, data):
  cmdq_push_data(dev, data & 0xFFFFFFFF)
  cmdq_push_data(dev, data >> 32)
def cmdq_push_method(dev, subc, mthd, size): cmdq_push_data(dev, 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2)) 

class NVCompiler(Compiler):
  compiler_opts = CompilerOptions("NV", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    NVCompiler.compiler_opts = NVCompiler.compiler_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    cuda_check(cuda.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_cuda_{self.arch}")
  def render(self, name:str, uops) -> str: return CUDARenderer(name, uops)
  def compile(self, src:str) -> bytes:
    cuda_check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0: raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, cuda_check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetCUBIN, cuda.nvrtcGetCUBINSize, cuda_check)

SHT_PROGBITS, SHF_ALLOC = 0x1, 0x2
class NVProgram:
  def __init__(self, device:NVDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

    _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]

    lib_gpu_size = round_up(max(sh[5]+sh[3] for sh in sections if sh[1] == SHT_PROGBITS), 0x200000)
    lib_gpu_mem_handle = self.device._gpu_alloc(lib_gpu_size, huge_page=True, contig=True)
    self.lib_gpu = self.device._gpu_uvm_map(lib_gpu_mem_handle, lib_gpu_size, cpu_visible=True, fixed_address=self.device.next_lib_address)
    lib_gpu_view = to_mv(self.lib_gpu, lib_gpu_size)
    self.device.next_lib_address += lib_gpu_size

    ld_loc = 0
    for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, _, _ in sections:
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC:
        if sh_flags == 66: self.constants = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size]))
        elif sh_flags == 6:
          lib_gpu_view[ld_loc:ld_loc+sh_size] = self.lib[sh_offset:sh_offset+sh_size]
          ld_loc += sh_size

    self.handle = self.lib_gpu

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
    
    ctypes.memmove(self.device.kernargs_ptr, from_mv(self.constants), self.constants.nbytes)
    args_st = self.args_struct_t.from_address(self.device.kernargs_ptr + self.constants.nbytes - ctypes.sizeof(self.args_struct_t))
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i])
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

    cmd = cmd_compute(self.handle, self.device.kernargs_ptr, round_up(self.constants.nbytes, 16), global_size, local_size)
    self.device.kernargs_ptr += round_up(self.constants.nbytes, 16)

    cmdq_start_wptr = self.device.cmdq_wptr
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2)
    cmdq_push_data64(self.device, 0x00007FFFF4000000)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 2)
    cmdq_push_data64(self.device, 0x004B0000)

    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2)
    cmdq_push_data64(self.device, self.device.qmd>>8)

    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LOAD_INLINE_QMD_DATA(0), 0x40)
    for i in range(0x40): cmdq_push_data(self.device, cmd[i])

    self.device._cmdq_insert_progress_semaphore()
    packets_written = (self.device.cmdq_wptr - cmdq_start_wptr) // 4
    # hexdump(to_mv(self.device.cmdq_addr, packets_written * 4))
    self.device.gpu_ring[self.device.gpu_ring_controls.GPPut] = (((self.device.cmdq_addr+cmdq_start_wptr)//4) << 2) | (packets_written << 42) | (1 << 63)
    self.device.gpu_ring_controls.GPPut += 1
    self.device._cmdq_ring_doorbell()
    self.device.synchronize() # TODO: remove

class NVAllocator(LRUAllocator):
  def __init__(self, device:NVDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    if options.host: return self.device._gpu_host_alloc(size)
    else:
      size = round_up(size, 2 << 20)
      mem_handle = self.device._gpu_alloc(size)
      return self.device._gpu_uvm_map(mem_handle, size)

  def copyin(self, dest, src: memoryview):
    # TODO: free host_mem
    host_mem = self.alloc(src.nbytes, BufferOptions(host=True))
    ctypes.memmove(host_mem, from_mv(src), src.nbytes)
    self.device._cmdq_dma_copy(dest, host_mem, src.nbytes)

  def copyout(self, dest:memoryview, src):
    # TODO: free host_mem
    host_mem = self.alloc(dest.nbytes, BufferOptions(host=True))
    self.device._cmdq_dma_copy(host_mem, src, dest.nbytes)
    ctypes.memmove(from_mv(dest), host_mem, dest.nbytes)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDevice(Compiled):
  root = None
  fd_ctl:int = -1
  fd_uvm:int = -1
  fd_uvm_2:int = -1

  def _gpu_alloc(self, size:int, coherent=False, huge_page=False, contig=False, system=False):
    attr, attr2, flags, alignment = 0, nvesc.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC, 0, 4<<10

    flags |= nvesc.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nvesc.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nvesc.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED

    if coherent:
      attr |= nvesc.NVOS32_ATTR_LOCATION_PCI << 25
      attr2 |= nvesc.NVOS32_ATTR2_GPU_CACHEABLE_NO << 2
    else:
      attr2 |= nvesc.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2
      flags |= nvesc.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM

    if contig: attr |= nvesc.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << 27
    else: attr |= nvesc.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27

    if huge_page:
      attr |= nvesc.NVOS32_ATTR_PAGE_SIZE_HUGE << 23
      attr2 |= nvesc.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20
      flags |= nvesc.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE
      alignment = 2 << 20
    else:
      attr |= nvesc.NVOS32_ATTR_PAGE_SIZE_4KB << 23

    size = round_up(size, alignment)
    alloc_params = nvesc.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, flags=flags, attr=attr, attr2=attr2, format=6, size=size, alignment=alignment, offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nvcls.NV1_MEMORY_SYSTEM if system else nvcls.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew
    return mem_handle

  def _new_gpu_fd(self):
    fd_dev0 = os.open(f"/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
    made = nvesc.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl)
    ret = fcntl.ioctl(fd_dev0, _IOWR(ord('F'), nvesc.NV_ESC_REGISTER_FD, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return fd_dev0

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0):
    fd_dev0 = self._new_gpu_fd()
    made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
      params=nvesc.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    ret = fcntl.ioctl(self.fd_ctl, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
    res = libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev0, 0)
    # print(hex(res))
    return res

  def _gpu_uvm_map(self, mem_handle, size:int, cpu_visible=False, fixed_address=None, map_flags=0):
    assert size % (2<<20) == 0
    if cpu_visible: va_base = self._gpu_map_to_cpu(mem_handle, size, target=fixed_address, flags=map_flags)
    else: va_base = libc.mmap(0, size, 0, 34, -1, 0) # gpu address
    return self._gpu_uvm_vis(va_base, size, mem_handle)

  def _gpu_host_alloc(self, size):
    va_base = libc.mmap(0, size + (2<<20), mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0) # gpu address
    va_base = round_up(va_base, 2<<20)
    size = round_up(size, 2<<20)
    return self._gpu_map_to_gpu(va_base, size)

  def _gpu_map_to_gpu(self, va_base, size):
    assert size % (2<<20) == 0
    fd_dev0 = self._new_gpu_fd()
    # hClass=113 - NV01_MEMORY_SYSTEM_OS_DESCRIPTOR
    # TODO: flags
    self.host_mem_object_enumerator += 1
    made = nvesc.nv_ioctl_nvos02_parameters_with_fd(fd=-1,
      params=nvesc.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hObjectNew=self.host_mem_object_enumerator, hClass=113, flags=1073745936, pMemory=va_base, limit=size-1))
    ret = fcntl.ioctl(fd_dev0, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.params.status}")
    # print(made.params.hObjectNew)
    return self._gpu_uvm_vis(va_base, size, made.params.hObjectNew)

  def _gpu_uvm_vis(self, va_base, size, mem_handle):
    creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)

    map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size, rmCtrlFd=self.fd_ctl, hClient=self.root, hMemory=mem_handle,
                                                              gpuAttributesCount=1)
    map_ext_params.perGpuAttributes[0].gpuUuid = nvuvm.struct_nv_uuid(uuid=self.gpu_uuid)
    map_ext_params.perGpuAttributes[0].gpuMappingType = 1
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_MAP_EXTERNAL_ALLOCATION[2]), map_ext_params)

    return va_base

  def __init__(self, device:str=""):
    if NVDevice.root is None:
      NVDevice.fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm_2 = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.root = rm_alloc(self.fd_ctl, nvesc.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew
      uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_INITIALIZE), nvuvm.UVM_INITIALIZE_PARAMS())
      uvm_ioctl(self.fd_uvm_2, int(nvuvm.UVM_MM_INITIALIZE[2]), nvuvm.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm))

    # TODO: Get classes from NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    self.host_mem_object_enumerator = 0x1000 # start 

    device_params = nvcls.NV0080_ALLOC_PARAMETERS(deviceId=0x0, hClientShare=self.root, vaMode=nvesc.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.device = rm_alloc(self.fd_ctl, nvcls.NV01_DEVICE_0, self.root, self.root, device_params).hObjectNew
    self.subdevice = rm_alloc(self.fd_ctl, nvcls.NV20_SUBDEVICE_0, self.root, self.device, None).hObjectNew
    self.usermode = rm_alloc(self.fd_ctl, nvcls.TURING_USERMODE_A, self.root, self.subdevice, None).hObjectNew
    gpu_mmio_ptr = self._gpu_map_to_cpu(self.usermode, 0x10000, flags=2)

    vaspace_params = nvesc.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nvesc.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nvesc.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = rm_alloc(self.fd_ctl, nvcls.FERMI_VASPACE_A, self.root, self.device, vaspace_params).hObjectNew

    gpu_uuid_params = nvctrl.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nvctrl.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
    rm_control(self.fd_ctl, nvctrl.NV2080_CTRL_CMD_GPU_GET_GID_INFO, self.root, self.subdevice, gpu_uuid_params)
    self.gpu_uuid = (ctypes.c_ubyte*16)()
    for i in range(16): self.gpu_uuid[i] = gpu_uuid_params.data[i]

    register_gpu = nvuvm.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid))
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_GPU[2]), register_gpu)

    register_vaspace = nvuvm.UVM_REGISTER_GPU_VASPACE_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid), 
      rmCtrlFd=self.fd_ctl, hClient=self.root, hVaSpace=vaspace)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_GPU_VASPACE[2]), register_vaspace)

    channel_params = nvesc.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nvcls.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.fd_ctl, nvcls.KEPLER_CHANNEL_GROUP_A, self.root, self.device, channel_params).hObjectNew

    gpfifo_mem_handle = self._gpu_alloc(0x200000, huge_page=True, contig=True)
    gpfifo_addr = self._gpu_uvm_map(gpfifo_mem_handle, 0x200000, cpu_visible=True, fixed_address=0x200400000)

    notifier = self._gpu_alloc(0x1000, coherent=True, system=True)

    ctxshare_params = nvesc.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nvesc.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.fd_ctl, nvcls.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params).hObjectNew

    gpfifo_params = nvesc.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier, hObjectBuffer=gpfifo_mem_handle, gpFifoOffset=gpfifo_addr,
      gpFifoEntries=0x400, hContextShare=ctxshare, hUserdMemory=(ctypes.c_uint32*8)(gpfifo_mem_handle), userdOffset=(ctypes.c_uint64*8)(0x2000))
    gpfifo = rm_alloc(self.fd_ctl, nvcls.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, gpfifo_params).hObjectNew
    compute = rm_alloc(self.fd_ctl, nvcls.ADA_COMPUTE_A, self.root, gpfifo, None).hObjectNew
    dma = rm_alloc(self.fd_ctl, nvcls.AMPERE_DMA_COPY_B, self.root, gpfifo, None).hObjectNew

    ws_token_params = nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    rm_control(self.fd_ctl, nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, self.root, gpfifo, ws_token_params)
    assert ws_token_params.workSubmitToken != -1

    register_channel_params = nvuvm.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl, hClient=self.root,
      hChannel=gpfifo, base=0x203600000, length=0x2c1a000)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_CHANNEL[2]), register_channel_params)

    en_fifo_params = nvctrl.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    rm_control(self.fd_ctl, nvctrl.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, self.root, channel_group, en_fifo_params)

    self.gpu_ring = to_mv(gpfifo_addr, 0x2000).cast("Q")
    self.gpu_ring_controls = nvcls.AmpereAControlGPFifo.from_address(gpfifo_addr + 0x2000)
    self.gpu_mmio = to_mv(gpu_mmio_ptr, 0x10000).cast("Q")
    self.gpfifo_token = ws_token_params.workSubmitToken

    cmdq_mem_handle = self._gpu_alloc(0x200000, huge_page=True, contig=True)
    self.cmdq_addr = self._gpu_uvm_map(cmdq_mem_handle, 0x200000, cpu_visible=True, fixed_address=0x200700000)
    self.cmdq = to_mv(self.cmdq_addr, 0x200000).cast("I")
    self.cmdq_wptr = 0 # in bytes

    semaphores_mem_handle = self._gpu_alloc(0x200000, huge_page=True, contig=True)
    self.semaphores_addr = self._gpu_uvm_map(semaphores_mem_handle, 0x200000, cpu_visible=True, fixed_address=0x200900000)
    self.semaphores = to_mv(self.semaphores_addr, 0x200000).cast("Q")

    kernargs_mem_handle = self._gpu_alloc(0x200000, huge_page=True, contig=True)
    self.kernargs_base_addr = self._gpu_uvm_map(kernargs_mem_handle, 0x200000, cpu_visible=True, fixed_address=0x300a00000)
    self.kernargs_ptr = self.kernargs_base_addr

    qmd_mem_handle = self._gpu_alloc(0x200000, huge_page=True, contig=True)
    self.qmd = self._gpu_uvm_map(qmd_mem_handle, 0x200000, cpu_visible=True, fixed_address=0x300c00000)

    self.arch = 'sm_90' # TODO: fix
    self.next_lib_address = 0x300e00000 # TODO: remove with better alloc
    
    super().__init__(device, NVAllocator(self), NVCompiler(self.arch), functools.partial(NVProgram, self))

  def synchronize(self):
    sem_value = self.semaphores[0]
    while sem_value != self.gpu_ring_controls.GPPut: sem_value = self.semaphores[0]

  def _cmdq_insert_progress_semaphore(self):
    cmdq_push_method(self, 0, nvcls.NVC56F_SEM_ADDR_LO, 5)
    cmdq_push_data64_le(self, self.semaphores_addr)
    cmdq_push_data64_le(self, 0x1)
    cmdq_push_data(self, 0x6 | (1 << 24) | (5 << 27))
  
  def _cmdq_dma_copy(self, dst, src, sz):
    cmdq_start_wptr = self.cmdq_wptr
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_OFFSET_IN_UPPER, 4)
    cmdq_push_data64(self, src)
    cmdq_push_data64(self, dst)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LINE_LENGTH_IN, 1)
    cmdq_push_data(self, sz)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LAUNCH_DMA, 1)
    cmdq_push_data(self, 0x00000182 | (1 << 25)) # TODO: flags...
    self._cmdq_insert_progress_semaphore()
    packets_written = (self.cmdq_wptr - cmdq_start_wptr) // 4
    self.gpu_ring[self.gpu_ring_controls.GPPut] = ((self.cmdq_addr+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 63)
    self.gpu_ring_controls.GPPut += 1
    self._cmdq_ring_doorbell()
    self.synchronize() # TODO: remove

  def _cmdq_ring_doorbell(self): self.gpu_mmio[0x90 // 8] = self.gpfifo_token # TODO: this is bad...
