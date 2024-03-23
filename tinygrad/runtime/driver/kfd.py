import os, ctypes, pathlib, re, fcntl, functools, mmap
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import to_mv
from extra.hip_gpu_driver import hip_ioctl

libc = ctypes.CDLL("libc.so.6")
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
MAP_NORESERVE = 0x4000
MAP_FIXED = 0x10

def kfd_ioctl(idir, nr, user_struct, fd, **kwargs):
  made = user_struct(**kwargs)
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(user_struct)<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
def ioctls_from_header():
  hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)

  fxns = {}
  for name, idir, nr, sname in matches:
    fxns[name.replace("AMDKFD_IOC_", "").lower()] = functools.partial(kfd_ioctl, idirs[idir], int(nr, 0x10), getattr(kfd, "struct_"+sname))
  return type("KIO", (object, ), fxns)
kio = ioctls_from_header()

if __name__ == "__main__":
  fd = os.open("/dev/kfd", os.O_RDWR)
  drm_fd = os.open("/dev/dri/renderD128", os.O_RDWR)
  GPU_ID = 0xBFE4

  ver = kio.get_version(fd)
  st = kio.acquire_vm(fd, drm_fd=drm_fd, gpu_id=GPU_ID)

  # 0xF0000001 = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000002 = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000004 = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0x94000010 = KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
  #mem = kio.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xD6000004)

  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)

  arr = (ctypes.c_int32 * 1)(GPU_ID)
  stm = kio.map_memory_to_gpu(fd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1

  nq = kio.create_queue(fd, ring_base_address=buf, ring_size=0x1000, gpu_id=GPU_ID,
                        queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE,
                        queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY, write_pointer_address=buf+8, read_pointer_address=buf+0x10)
  print(nq)
  #mv = to_mv(buf, 0x1000)

  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)

  #print('\n'.join(format_struct(ver)))
  #print('\n'.join(format_struct(st)))
