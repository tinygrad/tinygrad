from __future__ import annotations
from typing import Tuple
import os, fcntl, ctypes, functools, re, pathlib, mmap
from tinygrad.device import Compiled, LRUAllocator, Compiler
from tinygrad.helpers import getenv, from_mv
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.driver.hip_comgr import compile_hip
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401

class KFDCompiler(Compiler):
  linearizer_opts = LinearizerOptions("KFD", has_tensor_cores=True, shared_max=65536)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

libc = ctypes.CDLL("libc.so.6")
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

def kfd_ioctl(idir, nr, user_struct, fd, **kwargs):
  made = user_struct(**kwargs)
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(user_struct)<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def ioctls_from_header():
  hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
  fxns = {name.replace("AMDKFD_IOC_", "").lower():
          functools.partial(kfd_ioctl, idirs[idir], int(nr, 0x10), getattr(kfd, "struct_"+sname))
          for name, idir, nr, sname in matches}
  return type("KIO", (object, ), fxns)
kio = ioctls_from_header()

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
def gpu_alloc(fd:int, drm_fd:int, gpu_id:int, size:int, flags:int):
  if flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR:
    buf = addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
  else:
    buf, addr = 0, libc.mmap(0, size, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  assert addr != 0xffffffffffffffff
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=size, gpu_id=gpu_id, flags=flags, mmap_offset=buf)
  if not (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR):
    buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)
    assert buf != 0xffffffffffffffff
    assert addr == buf == mem.va_addr
  arr = (ctypes.c_int32 * 1)(gpu_id)
  stm = kio.map_memory_to_gpu(fd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  return mem

class KFDProgram:
  def __init__(self, device:KFDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    #self.device._submit_packet()
    pass

class KFDAllocator(LRUAllocator):
  def __init__(self, device:KFDDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    return gpu_alloc(KFDDevice.kfd, self.device.drm_fd, self.device.gpu_id, size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

  # obviously slow
  def copyin(self, dest, src: memoryview):
    ctypes.memmove(dest.va_addr, from_mv(src), dest.size)
  def copyout(self, dest:memoryview, src):
    ctypes.memmove(from_mv(dest), src.va_addr, src.size)

class KFDDevice(Compiled):
  kfd:int = -1
  def __init__(self, device:str=""):
    if KFDDevice.kfd == -1: KFDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.drm_fd = os.open(f"/dev/dri/renderD{128+self.device_id}", os.O_RDWR)
    with open(f"/sys/devices/virtual/kfd/kfd/topology/nodes/{1+self.device_id}/gpu_id", "r") as f:
      self.gpu_id = int(f.read())
    self.arch = "gfx1100"
    kio.acquire_vm(KFDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    self.eop_buffer = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.aql_ring = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.gart = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

    self.aql_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.aql_ring.va_addr, ring_size=self.aql_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=self.eop_buffer.va_addr, eop_buffer_size=self.eop_buffer.size,
      write_pointer_address=self.gart.va_addr+0, read_pointer_address=self.gart.va_addr+0x8)
    self.doorbell_page = libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, KFDDevice.kfd, self.aql_queue.doorbell_offset)
    super().__init__(device, KFDAllocator(self), KFDCompiler(self.arch), functools.partial(KFDProgram, self))
