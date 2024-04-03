from __future__ import annotations
from typing import Tuple, Any
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct, errno
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.driver.hip_comgr import compile_hip
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
import tinygrad.runtime.autogen.amd_gpu as amd_gpu
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401

libc = ctypes.CDLL("libc.so.6")
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

def node_sysfs_path(node_id, file): return f"/sys/devices/virtual/kfd/kfd/topology/nodes/{node_id}/{file}"

def kfd_ioctl(idir, nr, user_struct, fd, made_struct=None, **kwargs):
  made = made_struct or user_struct(**kwargs)
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(made)<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def ioctls_from_header():
  #hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  #pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  #matches = re.findall(pattern, hdr, re.MULTILINE)
  # get this from python instead
  hdrpy = (pathlib.Path(__file__).parent / "autogen" / "kfd.py").read_text()
  pattern = r'# (AMDKFD_IOC_[A-Z0-9_]+)\s=\s_(IOW?R?).*\(( 0x[0-9a-fA-F]+) ,\s+struct\s([A-Za-z0-9_]+)\s+\)'
  matches = re.findall(pattern, hdrpy, re.MULTILINE)
  idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
  fxns = {name.replace("AMDKFD_IOC_", "").lower():
          functools.partial(kfd_ioctl, idirs[idir], int(nr, 0x10), getattr(kfd, "struct_"+sname))
          for name, idir, nr, sname in matches}
  return type("KIO", (object, ), fxns)
kio = ioctls_from_header()

def create_sdma_packets():
  # TODO: clean up this, if we want to keep it
  structs = {}
  for name,pkt in [(name,s) for name,s in amd_gpu.__dict__.items() if name.startswith("struct_SDMA_PKT_") and name.endswith("_TAG")]:
    names = set()
    fields = []
    for pkt_fields in pkt._fields_:
      if not pkt_fields[0].endswith("_UNION"): fields.append(pkt_fields)
      else:
        assert pkt_fields[1]._fields_[0][0] == '_0'
        for union_fields in pkt_fields[1]._fields_[0][1]._fields_:
          fname = union_fields[0]
          if fname in names: fname = pkt_fields[0]+fname
          names.add(fname)
          if fname.endswith("_63_32") and fields[-1][0].endswith("_31_0"):
            fields[-1] = tuple([fname[:-6], ctypes.c_ulong, 64])  # merge together 64-bit fields
          else:
            fields.append(tuple([fname, *union_fields[1:]]))
    new_name = name[16:-4].lower()
    structs[new_name] = init_c_struct_t(tuple(fields))
    assert ctypes.sizeof(structs[new_name]) == ctypes.sizeof(pkt), f"{ctypes.sizeof(structs[new_name])} != {ctypes.sizeof(pkt)}"
  return type("SDMA_PKTS", (object, ), structs)
sdma_pkts = create_sdma_packets()

class KFDCompiler(Compiler):
  compiler_opts = CompilerOptions("KFD", has_tensor_cores=True, shared_max=65536)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
SDMA_MAX_COPY_SIZE = 0x400000

VENDOR_HEADER = hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE

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

    _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]

    lib_gpu_size = round_up(max(sh[5]+sh[3] for sh in sections if sh[1] == SHT_PROGBITS), 0x1000)
    self.lib_gpu = self.device._gpu_alloc(lib_gpu_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    lib_gpu_view = to_mv(self.lib_gpu.va_addr, lib_gpu_size)

    for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, _, _ in sections:
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC: lib_gpu_view[sh_addr:sh_addr+sh_size] = self.lib[sh_offset:sh_offset+sh_size]

    self.device._submit_cache_inv(gli=2)

    entry_point = min(sh[3] for sh in sections if sh[1] == SHT_PROGBITS and sh[2] & SHF_ALLOC)
    self.handle = self.lib_gpu.va_addr + entry_point
    self.group_segment_size = lib_gpu_view.cast("I")[entry_point//4]
    self.private_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 1]
    self.kernargs_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 2]
    assert self.private_segment_size <= self.device.max_private_segment_size, \
      f"{self.private_segment_size=} > {self.device.max_private_segment_size=}"

  # NOTE: no programs are ever freed
  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device._gpu_free(self.lib_gpu)

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
                                                           (self.device.aql_doorbell_value*AQL_PACKET_SIZE) % self.device.aql_ring.size)
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
    self.device.amd_aql_queue.write_dispatch_id = self.device.aql_doorbell_value + 1
    self.device.aql_doorbell[0] = self.device.aql_doorbell_value
    self.device.aql_doorbell_value += 1

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
    try:
      if options.host: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True)
      else: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    except OSError as e:
      if e.errno == errno.ENOMEM: raise MemoryError("Cannot allocate memory") from e
      else: raise

  def _free(self, gpumem, options:BufferOptions):
    self.device._gpu_free(gpumem)

  def copyin(self, dest, src: memoryview):
    # TODO: need to make the address visible to gpu and pass it directly to sdma.
    self.device._map_userptr_to_gpu(ctypes.addressof(from_mv(src).contents), src.nbytes)
    self.device.completion_signal.value = 1
    self.device._submit_sdma(dest.va_addr, ctypes.addressof(from_mv(src).contents), src.nbytes, completion_signal=self.device.completion_signal)
    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = self.device.completion_signal.event_id
    kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

  def copyout(self, dest:memoryview, src):
    self.device._map_userptr_to_gpu(ctypes.addressof(from_mv(dest).contents), dest.nbytes)
    self.device.completion_signal.value = 1
    self.device._submit_sdma(ctypes.addressof(from_mv(dest).contents), src.va_addr, dest.nbytes, completion_signal=self.device.completion_signal)
    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = self.device.completion_signal.event_id
    kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class KFDDevice(Compiled):
  kfd:int = -1
  event_page:Any = None  # TODO: fix types in kfd, Optional[kfd.struct_kfd_ioctl_alloc_memory_of_gpu_args]

  def _map_userptr_to_gpu(self, addr, size):
    self.map_uptr2gpu_struct.start_addr = addr&~0xfff
    self.map_uptr2gpu_struct.size = round_up(size+addr-(addr&~0xfff), 0x1000)
    kio.svm(self.kfd, made_struct=self.map_uptr2gpu_struct)

  def _gpu_map(self, mem):
    mem.__setattr__("mapped_gpu_ids", (ctypes.c_int32 * 1)(self.gpu_id))
    stm = kio.map_memory_to_gpu(self.kfd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(gpus:=mem.mapped_gpu_ids), n_devices=len(gpus))
    assert stm.n_success == 1

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
    if map_to_gpu: self._gpu_map(mem)
    return mem

  def _gpu_free(self, mem):
    if (gpus:=getattr(mem, "mapped_gpu_ids", None)) is not None:
      stm = kio.unmap_memory_from_gpu(self.kfd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(gpus), n_devices=len(gpus))
      assert stm.n_success == len(gpus)
    libc.munmap(mem.va_addr, mem.size)
    kio.free_memory_of_gpu(self.kfd, handle=mem.handle)

  def __init__(self, device:str=""):
    if KFDDevice.kfd == -1: KFDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    with open(node_sysfs_path(self.device_id+1, "gpu_id"), "r") as f: self.gpu_id = int(f.read())
    with open(node_sysfs_path(self.device_id+1, "properties"), "r") as f: self.properties = {line.split()[0]: int(line.split()[1]) for line in f}
    self.drm_fd = os.open(f"/dev/dri/renderD{self.properties['drm_render_minor']}", os.O_RDWR)
    self.arch = f"gfx{self.properties['gfx_target_version']//100}"
    kio.acquire_vm(KFDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    if KFDDevice.event_page is None:
      KFDDevice.event_page = self._gpu_alloc(0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      self.sync_event = kio.create_event(KFDDevice.kfd, event_page_offset=KFDDevice.event_page.handle, auto_reset=1)
    else:
      self._gpu_map(KFDDevice.event_page)
      self.sync_event = kio.create_event(KFDDevice.kfd, auto_reset=1)

    self.gart = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.aql_ring = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.signals_page = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.pm4_indirect_buf = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)

    self.eop_buffer = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.kernargs = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.ctx_save_restore_address = self._gpu_alloc(0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    self.completion_signal = hsa.amd_signal_t.from_address(self.signals_page.va_addr)
    self.completion_signal.value = 1
    self.completion_signal.kind = hsa.AMD_SIGNAL_KIND_USER
    self.completion_signal.event_mailbox_ptr = KFDDevice.event_page.va_addr + self.sync_event.event_slot_index*8
    self.completion_signal.event_id = self.sync_event.event_id

    # AQL Queue
    self.amd_aql_queue = hsa.amd_queue_t.from_address(self.gart.va_addr)
    self.amd_aql_queue.write_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id_field_base_byte_offset = getattr(hsa.amd_queue_t, 'read_dispatch_id').offset
    self.amd_aql_queue.queue_properties = hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING

    self.amd_aql_queue.max_cu_id = self.properties['simd_count'] // self.properties['simd_per_cu'] - 1
    self.amd_aql_queue.max_wave_id = self.properties['max_waves_per_simd'] * self.properties['simd_per_cu'] - 1

    # scratch setup
    self.max_private_segment_size = 512
    wave_scratch_len = round_up(((self.amd_aql_queue.max_wave_id + 1) * self.max_private_segment_size), 256) # gfx11 requires alignment of 256
    self.scratch_len = (self.amd_aql_queue.max_cu_id + 1) * self.properties['max_slots_scratch_cu'] * wave_scratch_len
    self.scratch = self._gpu_alloc(self.scratch_len, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.amd_aql_queue.scratch_backing_memory_location = self.scratch.va_addr
    self.amd_aql_queue.scratch_backing_memory_byte_size = self.scratch_len
    self.amd_aql_queue.scratch_wave64_lane_byte_size = self.max_private_segment_size * (self.amd_aql_queue.max_wave_id + 1) // 64
    self.amd_aql_queue.scratch_resource_descriptor[0] = self.scratch.va_addr & 0xFFFFFFFF
    self.amd_aql_queue.scratch_resource_descriptor[1] = ((self.scratch.va_addr >> 32) & 0xFFFF) | (1 << 30) # va_hi | SWIZZLE_ENABLE
    self.amd_aql_queue.scratch_resource_descriptor[2] = self.scratch_len & 0xFFFFFFFF
    self.amd_aql_queue.scratch_resource_descriptor[3] = 0x20814fac # FORMAT=BUF_FORMAT_32_UINT,OOB_SELECT=2,ADD_TID_ENABLE=1,TYPE=SQ_RSRC_BUF,SQ_SELs
    engines = self.properties['array_count'] // self.properties['simd_arrays_per_engine']
    self.amd_aql_queue.compute_tmpring_size = (wave_scratch_len // 256) << 12 | (self.scratch_len // (wave_scratch_len * engines))

    self.aql_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.aql_ring.va_addr, ring_size=self.aql_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=self.eop_buffer.va_addr, eop_buffer_size=self.eop_buffer.size,
      ctx_save_restore_address=self.ctx_save_restore_address.va_addr, ctx_save_restore_size=self.ctx_save_restore_address.size,
      ctl_stack_size = 0xa000,
      write_pointer_address=self.gart.va_addr + getattr(hsa.amd_queue_t, 'write_dispatch_id').offset,
      read_pointer_address=self.gart.va_addr + getattr(hsa.amd_queue_t, 'read_dispatch_id').offset)

    self.doorbells_base = self.aql_queue.doorbell_offset & (~0xfff)
    self.doorbells = libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, KFDDevice.kfd, self.doorbells_base)
    self.aql_doorbell = to_mv(self.doorbells + self.aql_queue.doorbell_offset - self.doorbells_base, 4).cast("I")
    self.aql_doorbell_value = 0

    # SDMA Queue
    self.sdma_ring = self._gpu_alloc(1 << 20, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.sdma_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.sdma_ring.va_addr, ring_size=self.sdma_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_SDMA, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      write_pointer_address=self.gart.va_addr + 0x100, read_pointer_address=self.gart.va_addr + 0x108)

    self.sdma_read_pointer = to_mv(self.sdma_queue.read_pointer_address, 8).cast("Q")
    self.sdma_write_pointer = to_mv(self.sdma_queue.write_pointer_address, 8).cast("Q")
    self.sdma_doorbell = to_mv(self.doorbells + self.sdma_queue.doorbell_offset - self.doorbells_base, 4).cast("I")
    self.sdma_doorbell_value = 0

    # prebuilt packets
    self.sdma_flush_hdp_pkt = sdma_pkts.hdp_flush(0x8, 0x0, 0x80000000, 0x0, 0x0, 0x0)
    self.sdma_cache_inv = sdma_pkts.gcr(op=amd_gpu.SDMA_OP_GCR, sub_op=amd_gpu.SDMA_SUBOP_USER_GCR, GCR_CONTROL_GL2_WB=1, GCR_CONTROL_GLK_WB=1,
                                        GCR_CONTROL_GL2_INV=1, GCR_CONTROL_GL1_INV=1, GCR_CONTROL_GLV_INV=1, GCR_CONTROL_GLK_INV=1,
                                        GCR_CONTROL_GL2_RANGE=0)
    self.sdma_cache_wb = sdma_pkts.gcr(op=amd_gpu.SDMA_OP_GCR, sub_op=amd_gpu.SDMA_SUBOP_USER_GCR, GCR_CONTROL_GL2_WB=1, GCR_CONTROL_GLK_WB=1,
                                        GCR_CONTROL_GL2_RANGE=0)

    pm4_indirect_cmd = (ctypes.c_uint32*13)(amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), self.pm4_indirect_buf.va_addr & 0xffffffff,
                                            (self.pm4_indirect_buf.va_addr>>32) & 0xffffffff, 8 | amd_gpu.INDIRECT_BUFFER_VALID, 0xa)
    ctypes.memmove(ctypes.addressof(pm4_cmds:=(ctypes.c_uint16*27)(1))+2, ctypes.addressof(pm4_indirect_cmd), ctypes.sizeof(pm4_indirect_cmd))
    self.pm4_packet = hsa.hsa_ext_amd_aql_pm4_packet_t(header=VENDOR_HEADER, pm4_command=pm4_cmds,
                                                       completion_signal=hsa.hsa_signal_t(ctypes.addressof(self.completion_signal)))

    # Helpers
    map_uptr2gpu_struct_t = init_c_struct_t(tuple(kfd.struct_kfd_ioctl_svm_args._fields_[:-1]+[('attrs', kfd.struct_kfd_ioctl_svm_attribute*2)])) # type: ignore
    self.map_uptr2gpu_struct = map_uptr2gpu_struct_t(nattr=2, op=0x0)
    self.map_uptr2gpu_struct.attrs[0].type = kfd.KFD_IOCTL_SVM_ATTR_SET_FLAGS
    self.map_uptr2gpu_struct.attrs[0].value = kfd.KFD_IOCTL_SVM_FLAG_COHERENT
    self.map_uptr2gpu_struct.attrs[1].type = kfd.KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE
    self.map_uptr2gpu_struct.attrs[1].value = self.gpu_id

    super().__init__(device, KFDAllocator(self), KFDCompiler(self.arch), functools.partial(KFDProgram, self))

  def _submit_sdma(self, dest, src, copy_size, wait_signals=None, completion_signal=None):
    def blit_sdma_command(cmd):
      if (cmdsz:=ctypes.sizeof(cmd)) > (fill:=self.sdma_ring.size - self.sdma_doorbell_value % self.sdma_ring.size):
        ctypes.memset(self.sdma_ring.va_addr + (self.sdma_doorbell_value % self.sdma_ring.size), 0, fill)
        self.sdma_doorbell_value += fill
      ctypes.memmove(self.sdma_ring.va_addr + (self.sdma_doorbell_value % self.sdma_ring.size), ctypes.addressof(cmd), cmdsz)
      self.sdma_doorbell_value += cmdsz

    if wait_signals is not None:
      # NOTE: we check only low 32 bits to be zeroed, we don't use higher values for signals
      for sig in wait_signals:
        poll_addr = ctypes.addressof(sig) + getattr(hsa.amd_signal_t, 'value').offset
        blit_sdma_command(sdma_pkts.poll_regmem(op=amd_gpu.SDMA_OP_POLL_REGMEM, mem_poll=1, func=0x3, addr=poll_addr,
                          value=0, mask=0xffffffff, interval=0x04, retry_count=0xfff))

    if completion_signal is not None:
      blit_sdma_command(sdma_pkts.timestamp(op=amd_gpu.SDMA_OP_TIMESTAMP, sub_op=amd_gpu.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL,
                                            addr=ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'start_ts').offset))
    blit_sdma_command(self.sdma_flush_hdp_pkt)
    blit_sdma_command(self.sdma_cache_inv)

    copied = 0
    copies_commands = (copy_size + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE
    for _ in range(copies_commands):
      step_copy_size = min(copy_size - copied, SDMA_MAX_COPY_SIZE)
      blit_sdma_command(sdma_pkts.copy_linear(op=amd_gpu.SDMA_OP_COPY, sub_op=amd_gpu.SDMA_SUBOP_COPY_LINEAR,
                                              count=step_copy_size-1, src_addr=src+copied, dst_addr=dest+copied))
      copied += step_copy_size

    blit_sdma_command(self.sdma_cache_wb)
    if completion_signal is not None:
      blit_sdma_command(sdma_pkts.timestamp(op=amd_gpu.SDMA_OP_TIMESTAMP, sub_op=amd_gpu.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL,
                                            addr=ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'end_ts').offset))

    if completion_signal is not None:
      signal_addr = ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'value').offset
      blit_sdma_command(sdma_pkts.atomic(op=amd_gpu.SDMA_OP_ATOMIC, operation=amd_gpu.SDMA_ATOMIC_ADD64, addr=signal_addr, src_data=(1<<64)-1))
      if completion_signal.event_mailbox_ptr != 0:
        blit_sdma_command(sdma_pkts.fence(op=amd_gpu.SDMA_OP_FENCE, mtype=3, addr=completion_signal.event_mailbox_ptr,
                          data=completion_signal.event_id))
        blit_sdma_command(sdma_pkts.trap(op=amd_gpu.SDMA_OP_TRAP, int_ctx=completion_signal.event_id))

    self.sdma_write_pointer[0] = self.sdma_doorbell_value
    self.sdma_doorbell[0] = self.sdma_doorbell_value

  def _submit_cache_inv(self, addr=0x0, sz=(1 << 64)-1, gli=0, glv=0, glk=0, gl1=0, gl2=0):
    pm4_buffer_view = to_mv(self.pm4_indirect_buf.va_addr, 0x1000).cast("I")
    pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0,
               sz & 0xffffffff, (sz >> 32) & 0xff, addr & 0xffffffff, (addr >> 32) & 0xffffff, 0,
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2)]
    for i, value in enumerate(pm4_cmd): pm4_buffer_view[i] = value
    ctypes.memmove(self.aql_ring.va_addr + (self.aql_doorbell_value * AQL_PACKET_SIZE) % self.aql_ring.size,
                   ctypes.addressof(self.pm4_packet), AQL_PACKET_SIZE)

    self.amd_aql_queue.write_dispatch_id = self.aql_doorbell_value + 1
    self.aql_doorbell[0] = self.aql_doorbell_value
    self.aql_doorbell_value += 1

    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = self.completion_signal.event_id
    kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)

    assert (wp:=self.amd_aql_queue.write_dispatch_id) == (rp:=self.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"
