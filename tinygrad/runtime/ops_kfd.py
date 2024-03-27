from __future__ import annotations
from typing import Tuple
import os, fcntl, ctypes, functools, re, pathlib, mmap
from tinygrad.device import Compiled, LRUAllocator, Compiler
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up
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

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

class KFDProgram:
  def __init__(self, device:KFDDevice, name:str, lib:bytes):
    print("here")
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib
    self.lib_gpu = gpu_alloc(KFDDevice.kfd, self.device.drm_fd, self.device.gpu_id, round_up(len(lib), 0x1000),
                             kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                             kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                             kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
    lib_mv = to_mv(self.lib_gpu.va_addr, len(lib))
    lib_mv[:] = lib
    assert lib_mv.cast("I")[0x550//4] == 0x10c0
    lib_mv.cast("I")[0x550//4] -= 0x1000
    self.handle = self.lib_gpu.va_addr + 0x540

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    print("call")
    self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
    args_st = self.args_struct_t.from_address(self.device.kernargs.va_addr)
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].va_addr)
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])
    """
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
        raise RuntimeError(f"HSAProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")
    """
    print("there")
    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.device.aql_ring.va_addr)
    print("there 2")
    packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z = local_size
    packet.reserved0 = 0
    packet.grid_size_x, packet.grid_size_y, packet.grid_size_z = tuple(g*l for g,l in zip(global_size, local_size))

    #from hexdump import hexdump
    #hexdump(to_mv(self.lib_gpu.va_addr + 0x540, 0x1000))

    packet.kernel_object = self.handle
    packet.kernarg_address = self.device.kernargs.va_addr
    packet.private_segment_size = 0
    packet.group_segment_size = 0

    packet.reserved2 = 0
    packet.completion_signal = hsa.hsa_signal_t(ctypes.addressof(self.device.completion_signal))
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER

    # one pending packet + ring doorbell
    self.device.mgart[0] = 1
    self.device.doorbell[0] = 0

    #self.device._submit_packet()

    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = self.device.completion_signal.event_id
    kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

    print(self.device.mgart[0], self.device.mgart[1])

class KFDAllocator(LRUAllocator):
  def __init__(self, device:KFDDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    return gpu_alloc(KFDDevice.kfd, self.device.drm_fd, self.device.gpu_id, size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)

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

    self.event_page = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.sync_event = kio.create_event(KFDDevice.kfd, event_page_offset=self.event_page.handle, auto_reset=1)
    self.eop_buffer = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.aql_ring = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.signals_page = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                                  kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                  kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.gart = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.mgart = to_mv(self.gart.va_addr, 0x1000).cast("Q")
    self.kernargs = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                              kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    self.ctx_save_restore_address = gpu_alloc(KFDDevice.kfd, self.drm_fd, self.gpu_id, 0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                      kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                                      kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

    self.completion_signal = hsa.amd_signal_t.from_address(self.signals_page.va_addr)
    self.completion_signal.kind = hsa.AMD_SIGNAL_KIND_USER
    self.completion_signal.event_mailbox_ptr = self.event_page.va_addr + self.sync_event.event_slot_index*8
    self.completion_signal.event_id = self.sync_event.event_id

    self.aql_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.aql_ring.va_addr, ring_size=self.aql_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=self.eop_buffer.va_addr, eop_buffer_size=self.eop_buffer.size,
      ctx_save_restore_address=self.ctx_save_restore_address.va_addr, ctx_save_restore_size=self.ctx_save_restore_address.size,
      ctl_stack_size = 0xa000,
      write_pointer_address=self.gart.va_addr+0, read_pointer_address=self.gart.va_addr+0x8)
    self.doorbell = to_mv(libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED,
                                    KFDDevice.kfd, self.aql_queue.doorbell_offset), 8192).cast("I")
    super().__init__(device, KFDAllocator(self), KFDCompiler(self.arch), functools.partial(KFDProgram, self))
