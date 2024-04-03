import os, ctypes, pathlib, re, fcntl, functools, mmap, time
from tinygrad.helpers import to_mv, getenv, round_up
from extra.nv_gpu_driver import nv_ioctl
from extra.nv_gpu_driver import esc_ioctl as nvesc
from extra.nv_gpu_driver import class_ioctl as nvcls
from extra.nv_gpu_driver import ctrl_ioctl as nvctrl
from extra.nv_gpu_driver import uvm_ioctl as nvuvm
from extra.nv_gpu_driver import nv_qcmds as nvqcmd
from hexdump import hexdump

NV_RUNLIST_INTERNAL_DOORBELL = 0x090
NV_RUNLIST_INFO = 0x108

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
MAP_FIXED, MAP_NORESERVE = 0x10, 0x400

ring_cmd = 0x0
ring_off = 0x0

def PUSH_DATA(data):
  global ring_cmd, ring_off
  ring_cmd[ring_off//4] = data
  ring_off += 4
def PUSH_DATA64(data):
  PUSH_DATA(data >> 32)
  PUSH_DATA(data & 0xFFFFFFFF)
def NVC0_FIFO_PKHDR_SQ(subc, mthd, size): return 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2)
def BEGIN_NVC0(subc, mthd, size): PUSH_DATA(NVC0_FIFO_PKHDR_SQ(subc, mthd, size)) 
def NVC0_FIFO_PKHDR_NI(subc, mthd, size): return 0x60000000 | (size << 16) | (subc << 13) | (mthd >> 2)
def BEGIN_NIC0(subc, mthd, size): PUSH_DATA(NVC0_FIFO_PKHDR_NI(subc, mthd, size)) 

def set_bits_in_array(arr, end_bit, start_bit, value):
  # Calculate the start and end indices in the array
  start_index = start_bit // 32
  end_index = end_bit // 32

  # Calculate the number of bits to set
  num_bits = end_bit - start_bit + 1

  # Create a mask for the bits to be set
  mask = (1 << num_bits) - 1

  # Clear the bits in the specified range
  for i in range(start_index, end_index + 1):
    arr[i] &= ~(mask << (start_bit - i * 32))

  # Set the bits in the specified range to the new value
  for i in range(start_index, end_index + 1):
    arr[i] |= (value & mask) << (start_bit - i * 32)
    value >>= 32 - (start_bit - i * 32)

def cmd_dma_copy(dst, src, sz):
  BEGIN_NVC0(4, nvqcmd.NVC6B5_OFFSET_IN_UPPER, 4)
  PUSH_DATA64(src)
  PUSH_DATA64(dst)
  BEGIN_NVC0(4, nvqcmd.NVC6B5_LINE_LENGTH_IN, 1)
  PUSH_DATA(sz)
  BEGIN_NVC0(4, nvqcmd.NVC6B5_LAUNCH_DMA, 1)
  PUSH_DATA(0x00000182)

def cmd_memcpy(dst, src, sz):
  BEGIN_NVC0(1, nvqcmd.NVC6C0_OFFSET_OUT_UPPER, 2)
  PUSH_DATA64(dst)
  BEGIN_NVC0(1, nvqcmd.NVC6C0_LINE_LENGTH_IN, 2)
  PUSH_DATA(sz)
  PUSH_DATA(1)
  BEGIN_NVC0(1, nvqcmd.NVC6C0_LAUNCH_DMA, 1)
  PUSH_DATA(0x41)
  BEGIN_NIC0(1, nvqcmd.NVC6C0_LOAD_INLINE_DATA, sz//4)
  for i in range(sz//4): PUSH_DATA(src[i])

def cmd_compute(qmd, prog, params, params_len):
  arr = (ctypes.c_uint32 * 0x40)()

  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_QMD_GROUP_ID, 0x3F)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE, 1)

  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CWD_MEMBAR_TYPE, NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_SAMPLER_INDEX, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_SHARED_MEMORY_SIZE, 0x400)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE, 3)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1A)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_QMD_MAJOR_VERSION, 3)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_REGISTER_COUNT_V, 0x10)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE, 3)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_BARRIER_COUNT, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE, 0x640)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE, 0xa)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_SASS_VERSION, 0x86)

  # group
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_RASTER_WIDTH, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_RASTER_HEIGHT, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_RASTER_DEPTH, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_THREAD_DIMENSION0, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_THREAD_DIMENSION1, 1)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CTA_THREAD_DIMENSION2, 1)

  # program
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, program_address)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, program_address>>32)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, program_address>>8)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, program_address>>40)

  # args
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0), constant_address>>32)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0), constant_address)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0), constant_length>>4)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE_TRUE)
  set_bits_in_array(arr, *nvqcmd.C6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)

  BEGIN_NVC0(1, nvqcmd.NVC6C0_LOAD_INLINE_QMD_DATA(0), 0x40)
  for i in range(0x40): PUSH_DATA(dat[i])

  BEGIN_NVC0(1, nvqcmd.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2)
  PUSH_DATA64(qmd>>8)

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

def mmap_object(fd, fd_dev0, client, device, memory, length, target=None, flags=0):
  fd_dev0 = os.open(f"/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
  made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
    params=nvesc.NVOS33_PARAMETERS(hClient=client, hDevice=device, hMemory=memory, length=length, flags=flags))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
  return libc.mmap(target, length, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev0, 0)

def heap_alloc(fd_ctl, fd_uvm, fd_dev0, root, device, subdevice, addr, length, flags, mmap_flags, typ, gpu_uuid):
  made = nvesc.NVOS32_PARAMETERS(hRoot=root, hObjectParent=device, function=nvesc.NVOS32_FUNCTION_ALLOC_SIZE,
    data=nvesc.union_c__SA_NVOS32_PARAMETERS_data(AllocSize=nvesc.struct_c__SA_NVOS32_PARAMETERS_0_AllocSize(owner=root,type=typ,flags=flags, size=length)))
  ret = fcntl.ioctl(fd_ctl, _IOWR(ord('F'), nvesc.NV_ESC_RM_VID_HEAP_CONTROL, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"heap_alloc returned {made.status}")
  mem_handle = made.data.AllocSize.hMemory
  local_ptr = mmap_object(fd_ctl, fd_dev0, root, subdevice, mem_handle, length, addr, mmap_flags)
  # print(local_ptr, mem_handle)

  if typ != 0: return mem_handle

  creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=local_ptr, length=length)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)

  map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=local_ptr, length=length, rmCtrlFd=fd_ctl, hClient=root, hMemory=mem_handle,
                                                            gpuAttributesCount=1)
  map_ext_params.perGpuAttributes[0].gpuUuid = nvuvm.struct_nv_uuid(uuid=gpu_uuid)
  map_ext_params.perGpuAttributes[0].gpuMappingType = 1
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_MAP_EXTERNAL_ALLOCATION[2]), map_ext_params)
  return mem_handle

gpu_base_address = 0x500000000
def _gpu_alloc(root, device, fd_ctl, gpu_uuid, size:int, fixed_address=None):
  global gpu_base_address, gpu_va_base
  size = round_up(size, 2<<20)
  alloc_params = nvesc.NV_MEMORY_ALLOCATION_PARAMS(owner=root, flags=114945, attr=293601280, attr2=1048581, format=6, size=size, alignment=2<<20, offset=gpu_base_address, limit=2097151)
  mem_handle = rm_alloc(fd_ctl, nvcls.NV1_MEMORY_USER, root, device, alloc_params).hObjectNew
  gpu_base_address += size

  gpu_va_base = libc.mmap(0, size, 0, 34, -1, 0) #if fixed_address else 
  
  creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=gpu_va_base, length=size)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)

  map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=gpu_va_base, length=size, rmCtrlFd=fd_ctl, hClient=root, hMemory=mem_handle,
                                                            gpuAttributesCount=1)
  map_ext_params.perGpuAttributes[0].gpuUuid = nvuvm.struct_nv_uuid(uuid=gpu_uuid)
  map_ext_params.perGpuAttributes[0].gpuMappingType = 1
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_MAP_EXTERNAL_ALLOCATION[2]), map_ext_params)

  return gpu_va_base

if __name__ == "__main__":
  device_id = 0
  fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
  fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
  fd_uvm_2 = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
  fd_dev0 = os.open(f"/dev/nvidia{device_id}", os.O_RDWR | os.O_CLOEXEC)

  root = rm_alloc(fd_ctl, nvesc.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew

  device_params = nvcls.NV0080_ALLOC_PARAMETERS(deviceId=0x0, hClientShare=root, vaMode=nvesc.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
  device = rm_alloc(fd_ctl, nvcls.NV01_DEVICE_0, root, root, device_params).hObjectNew
  subdevice = rm_alloc(fd_ctl, nvcls.NV20_SUBDEVICE_0, root, device, None).hObjectNew
  usermode = rm_alloc(fd_ctl, nvcls.TURING_USERMODE_A, root, subdevice, None).hObjectNew
  gpu_mmio_ptr = mmap_object(fd_ctl, fd_dev0, root, subdevice, usermode, 0x10000, None, 2)
  gpu_mmio = to_mv(gpu_mmio_ptr, 0x10000).cast("Q") # turing regs from 

  vaspace_params = nvesc.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
    flags=nvesc.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING|nvesc.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
  vaspace = rm_alloc(fd_ctl, nvcls.FERMI_VASPACE_A, root, device, vaspace_params).hObjectNew

  # vaspace_params = nvesc.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=83886080, vaSize=562949869535232, flags=0)
  # vaspace2 = rm_alloc(fd_ctl, nvcls.FERMI_VASPACE_A, root, device, vaspace_params).hObjectNew

  gpu_uuid_params = nvctrl.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nvctrl.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
  rm_control(fd_ctl, nvctrl.NV2080_CTRL_CMD_GPU_GET_GID_INFO, root, subdevice, gpu_uuid_params)
  gpu_uuid = (ctypes.c_ubyte*16)()
  for i in range(16): gpu_uuid[i] = gpu_uuid_params.data[i]
  
  # register uvm
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_INITIALIZE), nvuvm.UVM_INITIALIZE_PARAMS())
  uvm_ioctl(fd_uvm_2, int(nvuvm.UVM_MM_INITIALIZE[2]), nvuvm.UVM_MM_INITIALIZE_PARAMS(uvmFd=fd_uvm))
  # uvm_ioctl(fd_uvm, int(nvuvm.UVM_PAGEABLE_MEM_ACCESS[2]), nvuvm.UVM_PAGEABLE_MEM_ACCESS_PARAMS(pageableMemAccess=0))

  register_gpu = nvuvm.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=nvuvm.struct_nv_uuid(uuid=gpu_uuid))
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_REGISTER_GPU[2]), register_gpu)

  # create_group = nvuvm.UVM_CREATE_RANGE_GROUP_PARAMS(rangeGroupId=0)
  # uvm_ioctl(fd_uvm, int(nvuvm.UVM_CREATE_RANGE_GROUP[2]), create_group)

  register_vaspace = nvuvm.UVM_REGISTER_GPU_VASPACE_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=gpu_uuid), rmCtrlFd=fd_ctl, hClient=root, hVaSpace=vaspace)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_REGISTER_GPU_VASPACE[2]), register_vaspace)

  # register fifo
  channel_params = nvesc.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nvcls.NV2080_ENGINE_TYPE_GRAPHICS)
  channel_group = rm_alloc(fd_ctl, nvcls.KEPLER_CHANNEL_GROUP_A, root, device, channel_params).hObjectNew

  fifo_buffer = heap_alloc(fd_ctl, fd_uvm, fd_dev0, root, device, subdevice, 0x200400000, 0x200000,
    nvesc.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nvesc.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE |
    nvesc.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nvesc.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nvesc.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM,
    0xc0000, nvesc.NVOS32_TYPE_IMAGE, gpu_uuid)
  err_buffer = heap_alloc(fd_ctl, fd_uvm, fd_dev0, root, device, subdevice, 0x7ffff7ffb000, 0x1000, 0xc001, 0, nvesc.NVOS32_TYPE_NOTIFIER, gpu_uuid)

  ctxshare_params = nvesc.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nvesc.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
  ctxshare = rm_alloc(fd_ctl, nvcls.FERMI_CONTEXT_SHARE_A, root, channel_group, ctxshare_params).hObjectNew

  gpfifo_params = nvesc.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=err_buffer, hObjectBuffer=fifo_buffer, gpFifoOffset=0x200400000,
    gpFifoEntries=0x400, hContextShare=ctxshare, hUserdMemory=(ctypes.c_uint32*8)(fifo_buffer), userdOffset=(ctypes.c_uint64*8)(0x2000))
  gpfifo = rm_alloc(fd_ctl, nvcls.AMPERE_CHANNEL_GPFIFO_A, root, channel_group, gpfifo_params).hObjectNew
  compute = rm_alloc(fd_ctl, nvcls.ADA_COMPUTE_A, root, gpfifo, None).hObjectNew

  ws_token_params = nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
  rm_control(fd_ctl, nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, root, gpfifo, ws_token_params)
  assert ws_token_params.workSubmitToken != -1

  register_channel_params = nvuvm.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=gpu_uuid), rmCtrlFd=fd_ctl, hClient=root,
    hChannel=gpfifo, base=0x203600000, length=0x2c1a000)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_REGISTER_CHANNEL[2]), register_channel_params)

  en_fifo_params = nvctrl.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
  rm_control(fd_ctl, nvctrl.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, channel_group, en_fifo_params)

  gpu_base = 0x200500000
  cmdq = gpu_base+0x6000
  ring_cmd = to_mv(cmdq, 0x1000).cast("I")
  cmd_memcpy(gpu_base+4, (ctypes.c_uint32*1).from_buffer_copy(b'\xaa\xbb\xcc\xdd'), 4)

  gpu_ring = to_mv(0x200400000, 0x2000).cast("Q")
  gpu_ring[0] = cmdq | (ring_off << 40)
  
  gpu_ring_controls = nvcls.AmpereAControlGPFifo.from_address(0x200400000 + 0x2000)
  gpu_ring_controls.GPPut = 1

  # these are open-gpu-kernel-modules/src/common/inc/swref/published/turing/tu102/dev_vm.h (which are not priv)
  gpu_mmio_ptr_view = to_mv(gpu_mmio_ptr, 0x1000).cast("I")
  gpu_mmio_ptr_view[0x90//4] = ws_token_params.workSubmitToken

  get_val = gpu_ring_controls.GPGet
  while get_val != 1: get_val = gpu_ring_controls.GPGet

  hexdump(to_mv(gpu_base, 0x100))

  _gpu_alloc(root, device, fd_ctl, gpu_uuid, 64)

  # ring_off

  # these are open-gpu-kernel-modules/src/common/inc/swref/published/turing/tu102/dev_vm.h (which are not priv)
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])

  print('finish')