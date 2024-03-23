import os, fcntl, ctypes, mmap
import time
import pathlib,re
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import to_mv, from_mv
from extra.hip_gpu_driver import hip_ioctl

libc = ctypes.CDLL("libc.so.6")
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
def ioctls_from_header():
  hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return type("KIO", (object, ), {name:(getattr(kfd, "struct_"+sname), int(nr, 0x10), idirs[idir]) for name, idir, nr, sname in matches})
  #return {int(nr, 0x10):(name, getattr(kfd, "struct_"+sname)) for name, nr, sname in matches}
kio = ioctls_from_header()
#for kk in dir(kio): print(getattr(kio, kk))

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

def ioctl_nr(idir, size, itype, nr):
  return (idir<<30) | (size<<16) | (itype<<8) | nr

def kfd_ioctl(tt, st):
  ret = fcntl.ioctl(fd, ioctl_nr(tt[2], ctypes.sizeof(st), ord('K'), tt[1]), st)
  assert ret == 0

# 1. Allocate GPU memory
# 2. Execute a GPU kernel to fill GPU memory
# 3. Read GPU memory and confirm kernel filled GPU memory
# 4. Do fun tests
# 5. Put in tinygrad?

# large graph = GPU command queue
# memory scheduling is separate
# each node in the graph has a program

if __name__ == "__main__":
  fd = os.open("/dev/kfd", os.O_RDWR)
  drm_fd = os.open("/dev/dri/renderD128", os.O_RDWR)
  print(kio.AMDKFD_IOC_GET_VERSION)
  kfd_ioctl(kio.AMDKFD_IOC_GET_VERSION, st:=kio.AMDKFD_IOC_GET_VERSION[0]())

  # /sys/devices/virtual/kfd/kfd/topology/nodes/1/gpu_id = 0xBFE4
  GPU_ID = 0xBFE4

  st = kio.AMDKFD_IOC_ACQUIRE_VM[0]()
  st.drm_fd = drm_fd
  st.gpu_id = GPU_ID
  kfd_ioctl(kio.AMDKFD_IOC_ACQUIRE_VM, st)

  # "Enable the AGP aperture. This provides an aperture in the GPU's internal address space for direct access to system memory. Note that these accesses are non-snooped, so they are only used for access to uncached memory."

  #mmap(NULL, 12288, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  #MAP_NORESERVE =
  #mmap.MAP_NORESERVE
  #addr = mmap.mmap(-1, 12288, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, prot=0)
  #print(ctypes.addressof(addr))
  #print(type(addr), dir(addr))
  #print(ctypes.c_int.from_buffer(addr).value)

  # mbind(0x752be242a000, 4096, MPOL_DEFAULT, NULL, 0, 0) = 0

  #st = kio.AMDKFD_IOC_SET_MEMORY_POLICY[0]()
  #st.alternate_aperture_base = 0x200000
  #st.alternate_aperture_size = 0x7FFFFFE00000
  #st.gpu_id = GPU_ID
  #st.default_policy = 1
  #st.alternate_policy = 0
  #kfd_ioctl(kio.AMDKFD_IOC_SET_MEMORY_POLICY, st)

  MAP_NORESERVE = 0x4000
  MAP_FIXED = 0x10
  #addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  buf = addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)
  print(hex(addr))
  #libc.munmap(addr, 0x1000)

  # GPU ram is on the drm device
  # 90.11 ms +   0.02 ms :  0 = AMDKFD_IOC_ALLOC_MEMORY_OF_GPU           va_addr:0x76ADC8000000 size:0x1000000 handle:0xBFE40000000F mmap_offset:0x10EE1C000 gpu_id:0xBFE4 flags:0xF0000001
  # mmap(0x76adc8000000, 16777216, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FIXED, 7, 0x10ee1c000) = 0x76adc8000000

  st = kio.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU[0]()
  st.va_addr = addr #+0x1000
  st.size = 0x1000
  st.gpu_id = GPU_ID
  # 0xF0000001 = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # VRAM = video memory
  # 0xD6000002 = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # GPU accessible system memory, mapped into the GPU’s virtual address space via gart
  # 0xD6000004 = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0x94000010 = KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  #st.flags = 0xF0000001
  st.flags = 0xD6000002
  #st.mmap_offset = addr

  #st.flags = 0x94000010
  kfd_ioctl(kio.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, st)
  buf = libc.mmap(st.va_addr, st.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, st.mmap_offset)
  assert buf != 0xffffffffffffffff
  print(hex(buf))

  stm = kio.AMDKFD_IOC_MAP_MEMORY_TO_GPU[0]()
  stm.handle = st.handle
  arr = (ctypes.c_int32 * 1)()
  arr[0] = GPU_ID
  stm.device_ids_array_ptr = ctypes.addressof(arr)
  #ctypes.cast(ctypes.c_uint64, ctypes.pointer(ctypes.c_int32(GPU_ID)))
  stm.n_devices = 1
  kfd_ioctl(kio.AMDKFD_IOC_MAP_MEMORY_TO_GPU, stm)
  assert stm.n_success == 1

  st = kio.AMDKFD_IOC_CREATE_QUEUE[0]()
  st.ring_base_address = buf
  st.ring_size = 0x1000
  st.gpu_id = GPU_ID
  st.queue_type = kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL
  st.queue_percentage = kfd.KFD_MAX_QUEUE_PERCENTAGE
  st.queue_priority = kfd.KFD_MAX_QUEUE_PRIORITY

  st.write_pointer_address = buf+8
  st.read_pointer_address = buf+0x10

  kfd_ioctl(kio.AMDKFD_IOC_CREATE_QUEUE, st)


  mv = to_mv(buf, 0x1000)
  print(mv[0], mv[1], mv[2])
  mv[0] = 101
  print(mv[0], mv[1], mv[2])

  #ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))


  time.sleep(2)

  #print(format_struct(st))
