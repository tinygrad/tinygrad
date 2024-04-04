from __future__ import annotations
import os, ctypes, pathlib, re, fcntl, functools, mmap, time
from typing import Tuple, Any
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct, errno
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import to_mv, getenv, round_up
from extra.nv_gpu_driver import nv_ioctl
from extra.nv_gpu_driver import esc_ioctl as nvesc
from extra.nv_gpu_driver import class_ioctl as nvcls
from extra.nv_gpu_driver import ctrl_ioctl as nvctrl
from extra.nv_gpu_driver import uvm_ioctl as nvuvm
from extra.nv_gpu_driver import nv_qcmds as nvqcmd
from hexdump import hexdump

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

# class NVProgram:
#   def __init__(self, device:NVDevice, name:str, lib:bytes):
#     # TODO; this API needs the type signature of the function and global_size/local_size
#     self.device, self.name, self.lib = device, name, lib

#     _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
#     sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]

#     lib_gpu_size = round_up(max(sh[5]+sh[3] for sh in sections if sh[1] == SHT_PROGBITS), 0x1000)
#     self.lib_gpu = self.device._gpu_alloc(lib_gpu_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
#     lib_gpu_view = to_mv(self.lib_gpu.va_addr, lib_gpu_size)

#     for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, _, _ in sections:
#       if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC: lib_gpu_view[sh_addr:sh_addr+sh_size] = self.lib[sh_offset:sh_offset+sh_size]

#     # self.device._submit_cache_inv(gli=2)

#     # entry_point = min(sh[3] for sh in sections if sh[1] == SHT_PROGBITS and sh[2] & SHF_ALLOC)
#     # self.handle = self.lib_gpu.va_addr + entry_point
#     # self.group_segment_size = lib_gpu_view.cast("I")[entry_point//4]
#     # self.private_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 1]
#     # self.kernargs_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 2]
#     # assert self.private_segment_size <= self.device.max_private_segment_size, \
#     #   f"{self.private_segment_size=} > {self.device.max_private_segment_size=}"

#   def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
#     if not hasattr(self, "args_struct_t"):
#       self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
#                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
#       if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
#         raise RuntimeError(f"HSAProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")
#     args_st = self.args_struct_t.from_address(self.device.kernargs.va_addr)
#     for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].va_addr)
#     for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

#     self.device.completion_signal.value = 1 # reset the signal before call
#     packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.device.aql_ring.va_addr +
#                                                            (self.device.aql_doorbell_value*AQL_PACKET_SIZE) % self.device.aql_ring.size)
#     packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z = local_size
#     packet.reserved0 = 0
#     packet.grid_size_x, packet.grid_size_y, packet.grid_size_z = tuple(g*l for g,l in zip(global_size, local_size))
#     packet.kernel_object = self.handle
#     packet.kernarg_address = self.device.kernargs.va_addr
#     packet.group_segment_size = self.group_segment_size
#     packet.private_segment_size = self.private_segment_size   # what it this and why doesn't it work? (see TestOps.test_dilated_conv_transpose2d)
#     packet.reserved2 = 0
#     packet.completion_signal = hsa.hsa_signal_t(ctypes.addressof(self.device.completion_signal))
#     packet.setup = DISPATCH_KERNEL_SETUP
#     packet.header = DISPATCH_KERNEL_HEADER

#     # one pending packet + ring doorbell
#     self.device.amd_aql_queue.write_dispatch_id = self.device.aql_doorbell_value + 1
#     self.device.aql_doorbell[0] = self.device.aql_doorbell_value
#     self.device.aql_doorbell_value += 1

#     evt_arr = (kfd.struct_kfd_event_data * 1)()
#     evt_arr[0].event_id = self.device.completion_signal.event_id
#     kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

#     assert (wp:=self.device.amd_aql_queue.write_dispatch_id) == (rp:=self.device.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"
#     if wait: return (self.device.completion_signal.end_ts-self.device.completion_signal.start_ts)/1e9

class NVAllocator(LRUAllocator):
  def __init__(self, device:NVDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    if options.host:
      mem_handle = self.device._gpu_alloc(size)
      return self.device._gpu_map_to_cpu(mem_handle, size)
    else:
      mem_handle = self.device._gpu_alloc(size)
      return self.device._gpu_uvm_map(mem_handle, size)

  def copyin(self, dest, src: memoryview):
    pass
    # self.device._map_userptr_to_gpu(ctypes.addressof(from_mv(src).contents), src.nbytes)
    # self.device.completion_signal.value = 1
    # self.device._submit_sdma(dest.va_addr, ctypes.addressof(from_mv(src).contents), src.nbytes, completion_signal=self.device.completion_signal)
    # evt_arr = (kfd.struct_kfd_event_data * 1)()
    # evt_arr[0].event_id = self.device.completion_signal.event_id
    # kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

  def copyout(self, dest:memoryview, src):
    pass
    # self.device._map_userptr_to_gpu(ctypes.addressof(from_mv(dest).contents), dest.nbytes)
    # self.device.completion_signal.value = 1
    # self.device._submit_sdma(ctypes.addressof(from_mv(dest).contents), src.va_addr, dest.nbytes, completion_signal=self.device.completion_signal)
    # evt_arr = (kfd.struct_kfd_event_data * 1)()
    # evt_arr[0].event_id = self.device.completion_signal.event_id
    # kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

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

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0):
    fd_dev0 = os.open(f"/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
    made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
      params=nvesc.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    ret = fcntl.ioctl(self.fd_ctl, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
    res = libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev0, 0)
    # print(hex(res))
    return res

  # def _gpu_uvm_map(self, mem_handle, size:int, cpu_visible=False, fixed_address=None, map_flags=0):

  
  def _gpu_uvm_map(self, mem_handle, size:int, cpu_visible=False, fixed_address=None, map_flags=0):
    if cpu_visible: va_base = self._gpu_map_to_cpu(mem_handle, size, target=fixed_address, flags=map_flags)
    else: va_base = libc.mmap(0, size, 0, 34, -1, 0) # gpu address
    return self._gpu_uvm_vis(va_base, size, mem_handle)

  def _gpu_host_alloc(self, size):
    libc.mmap(139747733278720, 536866816, 0, 43, -1, 0)
    va_base = libc.mmap(139747745857536, size, 3, 49, -1, 0) # gpu address
    print(hex(va_base))
    made = nvesc.nv_ioctl_nvos02_parameters_with_fd(fd=-1,
      params=nvesc.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hClass=113, flags=1073745936, pMemory=va_base, limit=size-1))
    ret = fcntl.ioctl(self.fd_dev, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.params.status}")
    return self._gpu_uvm_vis(self, va_base, size, mem_handle)

  def _gpu_uvm_vis(self, va_base, size, mem_handle):
    creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)

    map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size, rmCtrlFd=NVDevice.fd_ctl, hClient=self.root, hMemory=mem_handle,
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

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)

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

    # super().__init__(device, NVAllocator(self), NVCompiler(self.arch), functools.partial(NVProgram, self))
