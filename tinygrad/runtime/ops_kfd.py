from __future__ import annotations
from typing import Tuple
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions
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

AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

SHT_PROGBITS = 0x1
SHF_ALLOC = 0x2

class KFDProgram:
  def __init__(self, device:KFDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

    e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, e_shoff + i * e_shentsize) for i in range(e_shnum)]

    lib_gpu_size = round_up(max(sh[5]+sh[3] for sh in sections if sh[1] == SHT_PROGBITS), 0x1000)
    self.lib_gpu = self.device._gpu_alloc(lib_gpu_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    lib_gpu_view = to_mv(self.lib_gpu.va_addr, lib_gpu_size)

    for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, _, _ in sections:
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC: lib_gpu_view[sh_addr:sh_addr+sh_size] = self.lib[sh_offset:sh_offset+sh_size]

    entry_point = min(sh[3] for sh in sections if sh[1] == SHT_PROGBITS and sh[2] & SHF_ALLOC)
    self.handle = self.lib_gpu.va_addr + entry_point
    self.group_segment_size = lib_gpu_view.cast("I")[entry_point//4]
    self.private_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 1]
    self.kernargs_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 2]
    assert self.private_segment_size <= self.device.max_private_segment_size, f"{self.private_segment_size=} > {self.device.max_private_segment_size=}"

    #from hexdump import hexdump
    #hexdump(to_mv(self.handle, 0x100))

  # NOTE: no programs are ever freed
  def __del__(self): kio.free_memory_of_gpu(KFDDevice.kfd, handle=self.lib_gpu.handle)

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
        raise RuntimeError(f"HSAProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")
    args_st = self.args_struct_t.from_address(self.device.kernargs.va_addr)
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].va_addr)
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

    self.device.completion_signal.value = 1 # reset the signal before call
    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.device.aql_ring.va_addr +
                                                           (self.device.doorbell_value*AQL_PACKET_SIZE) % self.device.aql_ring.size)
    packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z = local_size
    packet.reserved0 = 0
    packet.grid_size_x, packet.grid_size_y, packet.grid_size_z = tuple(g*l for g,l in zip(global_size, local_size))
    packet.kernel_object = self.handle
    packet.kernarg_address = self.device.kernargs.va_addr
    packet.group_segment_size = self.group_segment_size
    packet.private_segment_size = self.private_segment_size   # what it this and why doesn't it work? (see TestOps.test_dilated_conv_transpose2d)
    packet.reserved2 = 0
    packet.completion_signal = hsa.hsa_signal_t(ctypes.addressof(self.device.completion_signal))
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER

    # one pending packet + ring doorbell
    self.device.amd_aql_queue.write_dispatch_id = self.device.doorbell_value+1
    self.device.doorbell[0] = self.device.doorbell_value
    self.device.doorbell_value += 1

    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = self.device.completion_signal.event_id
    kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

    assert (wp:=self.device.amd_aql_queue.write_dispatch_id) == (rp:=self.device.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"
    if wait: return (self.device.completion_signal.end_ts-self.device.completion_signal.start_ts)/1e9

class KFDAllocator(LRUAllocator):
  def __init__(self, device:KFDDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)

  # obviously slow
  def copyin(self, dest, src: memoryview):
    ctypes.memmove(dest.va_addr, from_mv(src), dest.size)
  def copyout(self, dest:memoryview, src):
    ctypes.memmove(from_mv(dest), src.va_addr, src.size)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class KFDDevice(Compiled):
  kfd:int = -1

  def _gpu_alloc(self, size:int, flags:int, uncached=False, public=False, map_to_gpu=True):
    flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
    if uncached: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED
    if public: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
    if flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR:
      buf = addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
    else:
      buf, addr = 0, libc.mmap(0, size, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
    assert addr != 0xffffffffffffffff
    mem = kio.alloc_memory_of_gpu(self.kfd, va_addr=addr, size=size, gpu_id=self.gpu_id, flags=flags, mmap_offset=buf)
    if not (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR):
      buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, self.drm_fd, mem.mmap_offset)
      assert buf != 0xffffffffffffffff
      assert addr == buf == mem.va_addr
    if map_to_gpu:
      arr = (ctypes.c_int32 * 1)(self.gpu_id)
      stm = kio.map_memory_to_gpu(self.kfd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
      assert stm.n_success == 1
    return mem

  def __init__(self, device:str=""):
    if KFDDevice.kfd == -1: KFDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.drm_fd = os.open(f"/dev/dri/renderD{128+self.device_id}", os.O_RDWR)
    with open(f"/sys/devices/virtual/kfd/kfd/topology/nodes/{1+self.device_id}/gpu_id", "r") as f:
      self.gpu_id = int(f.read())
    self.arch = "gfx1100"
    kio.acquire_vm(KFDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    self.event_page = self._gpu_alloc(0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.sync_event = kio.create_event(KFDDevice.kfd, event_page_offset=self.event_page.handle, auto_reset=1)
    self.eop_buffer = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.aql_ring = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.signals_page = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.gart = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    # self.mgart = to_mv(self.gart.va_addr, 0x1000).cast("Q")
    self.kernargs = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.ctx_save_restore_address = self._gpu_alloc(0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    self.completion_signal = hsa.amd_signal_t.from_address(self.signals_page.va_addr)
    self.completion_signal.value = 1
    self.completion_signal.kind = hsa.AMD_SIGNAL_KIND_USER
    self.completion_signal.event_mailbox_ptr = self.event_page.va_addr + self.sync_event.event_slot_index*8
    self.completion_signal.event_id = self.sync_event.event_id

    # Queue
    self.amd_aql_queue = hsa.amd_queue_t.from_address(self.gart.va_addr)
    self.amd_aql_queue.write_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id_field_base_byte_offset = getattr(hsa.amd_queue_t, 'read_dispatch_id').offset
    self.amd_aql_queue.queue_properties = hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING

    self.amd_aql_queue.max_cu_id = 95 # TODO: hardcoded for 7900xtx
    self.amd_aql_queue.max_wave_id = 31

    # scratch setup
    self.max_private_segment_size = 256
    self.scratch_len = self.max_private_segment_size * (self.amd_aql_queue.max_cu_id + 1) * (self.amd_aql_queue.max_wave_id + 1)
    self.scratch = self._gpu_alloc(self.scratch_len, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.amd_aql_queue.scratch_backing_memory_location = self.scratch.va_addr
    self.amd_aql_queue.scratch_backing_memory_byte_size = self.scratch_len
    self.amd_aql_queue.scratch_wave64_lane_byte_size = self.max_private_segment_size * (self.amd_aql_queue.max_wave_id + 1) // 64
    self.amd_aql_queue.scratch_resource_descriptor[0] = self.scratch.va_addr & 0xFFFFFFFF
    self.amd_aql_queue.scratch_resource_descriptor[1] = ((self.scratch.va_addr >> 32) & 0xFFFF) | (1 << 30) # va_hi | SWIZZLE_ENABLE
    self.amd_aql_queue.scratch_resource_descriptor[2] = self.scratch_len & 0xFFFFFFFF
    self.amd_aql_queue.scratch_resource_descriptor[3] = 0x20814fac # FORMAT=BUF_FORMAT_32_UINT, OOB_SELECT=2, ADD_TID_ENABLE=1, TYPE=SQ_RSRC_BUF, SQ_SELs

    wave_scratch = (((self.amd_aql_queue.max_wave_id + 1) * self.max_private_segment_size + 255) // 256)
    self.amd_aql_queue.compute_tmpring_size = wave_scratch << 12 | (self.amd_aql_queue.max_cu_id + 1)

    self.aql_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.aql_ring.va_addr, ring_size=self.aql_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=self.eop_buffer.va_addr, eop_buffer_size=self.eop_buffer.size,
      ctx_save_restore_address=self.ctx_save_restore_address.va_addr, ctx_save_restore_size=self.ctx_save_restore_address.size,
      ctl_stack_size = 0xa000,
      write_pointer_address=self.gart.va_addr + getattr(hsa.amd_queue_t, 'write_dispatch_id').offset,
      read_pointer_address=self.gart.va_addr + getattr(hsa.amd_queue_t, 'read_dispatch_id').offset)
    self.doorbell = to_mv(libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED,
                                    KFDDevice.kfd, self.aql_queue.doorbell_offset), 8192).cast("I")
    self.doorbell_value = 0
    super().__init__(device, KFDAllocator(self), KFDCompiler(self.arch), functools.partial(KFDProgram, self))
