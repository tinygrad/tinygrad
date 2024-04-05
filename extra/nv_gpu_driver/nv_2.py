import subprocess
import os, ctypes, pathlib, re, fcntl, functools, mmap, time
from tinygrad.helpers import to_mv, getenv, round_up
from tinygrad.runtime.ops_nv import NVDevice, NVAllocator, NVCompiler, NVProgram
from tinygrad.runtime.ops_cuda import CUDACompiler
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
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
def PUSH_DATA64LE(data):
  PUSH_DATA(data & 0xFFFFFFFF)
  PUSH_DATA(data >> 32)
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

def cmd_memcpy(dst, src, sz, signal_addr):
  BEGIN_NVC0(1, nvcls.NVC56F_SEM_ADDR_LO, 5)
  PUSH_DATA64LE(signal_addr)
  PUSH_DATA64LE(0x0)
  PUSH_DATA(0x2 | (1 << 12)| (1 << 24))

  BEGIN_NVC0(0, nvcls.NVC56F_NON_STALL_INTERRUPT, 1)
  PUSH_DATA(0x333)
  
  BEGIN_NVC0(1, nvqcmd.NVC6C0_OFFSET_OUT_UPPER, 2)
  PUSH_DATA64(dst)
  BEGIN_NVC0(1, nvqcmd.NVC6C0_LINE_LENGTH_IN, 2)
  PUSH_DATA(sz)
  PUSH_DATA(1)
  BEGIN_NVC0(1, nvqcmd.NVC6C0_LAUNCH_DMA, 1)
  PUSH_DATA(0x41)
  BEGIN_NIC0(1, nvqcmd.NVC6C0_LOAD_INLINE_DATA, sz//4)
  for i in range(sz//4): PUSH_DATA(src[i])

  BEGIN_NVC0(0, nvcls.NVC56F_SEM_ADDR_LO, 5)
  PUSH_DATA64LE(signal_addr)
  PUSH_DATA64LE(0xdeadbeefdeadbeef)
  PUSH_DATA(0x1 | (1 << 24) | (1 << 25))

  BEGIN_NVC0(0, nvcls.NVC56F_NON_STALL_INTERRUPT, 1)
  PUSH_DATA(0x333)

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

def _gpu_map_to_cpu(fd, fd_dev0, client, device, memory, length, target=None, flags=0):
  # fd_dev0 = os.open(f"/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
  # made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
  #   params=nvesc.NVOS33_PARAMETERS(hClient=client, hDevice=device, hMemory=memory, length=length, flags=flags))
  # ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
  # if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  # if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")

  # address = made.params.pLinearAddress
  # if target is not None:
  #   remap = nvesc.NVOS56_PARAMETERS(hClient=client, hDevice=device, hMemory=memory, pOldCpuAddress=address, pNewCpuAddress=target)
  #   ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO, ctypes.sizeof(remap)), made)
  #   if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  #   if remap.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {remap.status}")
  #   address = target
  # return address
  fd_dev0 = os.open(f"/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
  made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
    params=nvesc.NVOS33_PARAMETERS(hClient=client, hDevice=device, hMemory=memory, length=length, flags=flags))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
  return libc.mmap(target, length, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev0, 0)

def _gpu_alloc(root, device, fd_ctl, size:int, coherent=False, huge_page=False, contig=False, system=False):
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
  alloc_params = nvesc.NV_MEMORY_ALLOCATION_PARAMS(owner=root, flags=flags, attr=attr, attr2=attr2, format=6, size=size, alignment=alignment, offset=0, limit=size-1)
  mem_handle = rm_alloc(fd_ctl, nvcls.NV1_MEMORY_SYSTEM if system else nvcls.NV1_MEMORY_USER, root, device, alloc_params).hObjectNew
  return mem_handle

def _gpu_uvm_map(root, device, fd_ctl, fd_dev0, gpu_uuid, mem_handle, size:int, cpu_visible=False, fixed_address=None):
  if cpu_visible: va_base = _gpu_map_to_cpu(fd_ctl, fd_dev0, root, device, mem_handle, size, target=fixed_address, flags=0)
  else: va_base = libc.mmap(0, size, 0, 34, -1, 0) # allocate address

  creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)

  map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size, rmCtrlFd=fd_ctl, hClient=root, hMemory=mem_handle,
                                                            gpuAttributesCount=1)
  map_ext_params.perGpuAttributes[0].gpuUuid = nvuvm.struct_nv_uuid(uuid=gpu_uuid)
  map_ext_params.perGpuAttributes[0].gpuMappingType = 1
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_MAP_EXTERNAL_ALLOCATION[2]), map_ext_params)

  return va_base

code = """
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void E_3(int* data0) {
  // int lidx0 = threadIdx.x; /* 3 */
  data0[0] = 0xdeadbeef;
}
"""

if __name__ == "__main__":
  device_id = 0
  dev = NVDevice("NV:0")
  alloctor = NVAllocator(dev)
  compiler = NVCompiler(arch='sm_90')
  ccompiler = CUDACompiler(arch='sm_90')

  binary = compiler.compile(code)
  ptx = ccompiler.compile(code)
  try:
    with open("/home/nimlgen/tmp/tmp.ptx", "wb") as f: f.write(ptx)
    lb = subprocess.check_output(["ptxas", f"-arch=sm_90", "-o", "/home/nimlgen/tmp/tmp.ptx.o", "/home/nimlgen/tmp/tmp.ptx"])
  except Exception as e: print("failed to generate SASS", str(e))
  # print(lb)
  # hexdump("dai", lb)
  # with open("/home/nimlgen/cubin.elf", 'wb') as file:
  #   file.write(binary)

  # from io import BytesIO
  # from elftools.elf.elffile import ELFFile
  # from elftools.elf.sections import NoteSection
  # with BytesIO(binary) as f:
  # with open("/home/nimlgen/cuda_ioctl_sniffer/out/simple.o", "rb") as f:
  #   elf = ELFFile(f)
  #   print(elf.header['e_type'])

  #   # Print all segments
  #   print("Segments:")
  #   for i, segment in enumerate(elf.iter_segments()):
  #     print(f"  Segment {i}:")
  #     print(f"    Type: {segment['p_type']}")
  #     print(f"    Offset: {segment['p_offset']}")
  #     print(f"    Virtual Address: {segment['p_vaddr']}")
  #     print(f"    Physical Address: {segment['p_paddr']}")
  #     print(f"    Size in File: {segment['p_filesz']}")
  #     print(f"    Size in Memory: {segment['p_memsz']}")
  #     print(f"    Flags: {segment['p_flags']}")
  #     print(f"    Alignment: {segment['p_align']}")

  #   print("\nSections:")
  #   for i, section in enumerate(elf.iter_sections()):
  #     print(f"  Section {i}: {section.name}")
  #     print(f"    Type: {section['sh_type']}")
  #     print(f"    Address: {section['sh_addr']}")
  #     print(f"    Offset: {section['sh_offset']}")
  #     print(f"    Size: {section['sh_size']}")
  #     print(f"    Flags: {section['sh_flags']}")
  #     print(f"    Link: {section['sh_link']}")
  #     print(f"    Info: {section['sh_info']}")
  #     print(f"    Address Alignment: {section['sh_addralign']}")
  #     print(f"    Entry Size: {section['sh_entsize']}")

    # dynsym = elffile.get_section_by_name('.dynsym')
    # if not dynsym:
    #   print("Dynamic symbol table not found.")

    # kern_info = None
    # for section in elffile.iter_sections():
    #   if isinstance(section, NoteSection):

  prog = NVProgram(dev, "E_3", binary)
  gpu_buf_addr = alloctor.alloc(2<<20)
  # gpu_buf_addr2 = 0x200400000 + 0x107000 #alloctor.alloc(2<<20)
  # alloctor.copyin(gpu_buf_addr2, memoryview(bytearray(b'\xff\x00\xdd\xee')))
  # dev.synchronize()
  

  # for i in range(10): 
  prog(gpu_buf_addr)
  dev.synchronize()
  dev._cmdq_dma_copy(gpu_buf_addr+0x18, gpu_buf_addr, 8)
  dev.synchronize()
  # print("wait done called")
  # hexdump(to_mv(gpu_buf_addr, 0x40))
  hexdump(to_mv(dev.qmd, 0x40))
  # hexdump(to_mv(prog.handle, 0x100))
  # hexdump(to_mv(prog.kernarg_address, 0x200))
  res = memoryview(bytearray(0x4))
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  alloctor.copyout(res, gpu_buf_addr)
  print(res)
  for byte in res.tobytes(): print(byte)

  exit(0)
  gpu_base = 0x200400000 + 0x100000
  cmdq = dev.cmdq_addr
  signal_addr = gpu_base+0x7000
  ring_cmd = to_mv(cmdq, 0x1000).cast("I")
  cmd_memcpy(gpu_base+4, (ctypes.c_uint32*1).from_buffer_copy(b'\xaa\xbb\xcc\xdd'), 4, signal_addr)
  buf = (ctypes.c_uint32*1).from_buffer_copy(bytearray(b'\xff\x00\xdd\xee'))

  # addr = dev._gpu_map_to_gpu(ctypes.addressof(buf), (2<<20))
  # print(addr)
  local_buf_addr = alloctor.alloc(4, BufferOptions(host=True))
  gpu_buf_addr = alloctor.alloc(2<<20)
  ctypes.memmove(local_buf_addr, buf, 4)


  alloctor.copyin(gpu_buf_addr, memoryview(bytearray(b'\xff\x00\xdd\xee')))
  res = memoryview(bytearray(0x4))
  alloctor.copyout(res, gpu_buf_addr)
  dev.synchronize()
  for byte in res.tobytes(): print(byte)

  cmd_dma_copy(gpu_base+8, gpu_base+4, 4)
  cmd_dma_copy(gpu_buf_addr, local_buf_addr, 4)
  cmd_dma_copy(gpu_base+12, gpu_buf_addr, 4)

  dev.synchronize()

  gpu_ring = dev.gpu_ring

  assert cmdq == ((cmdq >> 2) << 2)
  assert ring_off == ((ring_off >> 2) << 2)
  gpu_ring[0] = cmdq | (ring_off << 40) | (1 << 63) # sync them
  
  dev.gpu_ring_controls.GPPut = 1

  # these are open-gpu-kernel-modules/src/common/inc/swref/published/turing/tu102/dev_vm.h (which are not priv)
  dev.gpu_mmio[0x90//8] = dev.gpfifo_token

  signal_addr_mv = to_mv(signal_addr, 16).cast("Q")
  get_val = signal_addr_mv[0]
  # while get_val != 0xdeadbeefdeadbeef: get_val = signal_addr_mv[0]

  # while 

  import time
  time.sleep(1)

  hexdump(to_mv(gpu_base, 0x20))

  print()
  hexdump(to_mv(signal_addr, 0x20))

  # _gpu_alloc(root, device, fd_ctl, gpu_uuid, 64)

  # ring_off

  # these are open-gpu-kernel-modules/src/common/inc/swref/published/turing/tu102/dev_vm.h (which are not priv)
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])
  # print("time=", gpu_mmio[0x80//8])

  print('finish')