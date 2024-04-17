from __future__ import annotations
from typing import Tuple, List, Any
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct, errno
from tinygrad.device import Compiled, LRUAllocator, Compiler, CompilerOptions
from tinygrad.buffer import BufferOptions
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

def is_usable_gpu(gpu_id):
  try:
    with gpu_id.open() as f:
      return int(f.read()) != 0
  except OSError:
    return False

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
          # merge together 64-bit fields, otherwise just append them
          if fname.endswith("_63_32") and fields[-1][0].endswith("_31_0"): fields[-1] = tuple([fname[:-6], ctypes.c_ulong, 64])
          else: fields.append(tuple([fname, *union_fields[1:]]))
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
PAGE_SIZE = 0x1000

SIGNAL_SIZE, SIGNAL_COUNT = ctypes.sizeof(hsa.amd_signal_t), 16384

VENDOR_HEADER = hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

SHT_PROGBITS = 0x1
SHF_ALLOC = 0x2

EMPTY_SIGNAL = hsa.hsa_signal_t()
SIGNAL_VALUE_OFFSET = getattr(hsa.amd_signal_t, 'value').offset

class HWComputeQueue:
  def __init__(self): self.q = []

  def exec(self, prg:KFDProgram, kernargs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), completion_signal=None):
    if completion_signal is not None: completion_signal.value = 1
    self.q.append(hsa.hsa_kernel_dispatch_packet_t(
      header=DISPATCH_KERNEL_HEADER, setup=DISPATCH_KERNEL_SETUP,
      workgroup_size_x=local_size[0], workgroup_size_y=local_size[1], workgroup_size_z=local_size[2],
      grid_size_x=global_size[0]*local_size[0], grid_size_y=global_size[1]*local_size[1], grid_size_z=global_size[2]*local_size[2],
      kernel_object=prg.handle, group_segment_size=prg.group_segment_size, private_segment_size=prg.private_segment_size, kernarg_address=kernargs,
      completion_signal=hsa.hsa_signal_t(ctypes.addressof(completion_signal)) if completion_signal is not None else EMPTY_SIGNAL))
    return self

  def signal(self, signal:hsa.amd_signal_t):
    signal.value = 1
    self.q.append(hsa.hsa_barrier_and_packet_t(header=BARRIER_HEADER, completion_signal=hsa.hsa_signal_t(ctypes.addressof(signal))))
    return self

  def wait(self, signal:hsa.amd_signal_t):
    sig = hsa.hsa_barrier_and_packet_t(header=BARRIER_HEADER)
    sig.dep_signal[0] = hsa.hsa_signal_t(ctypes.addressof(signal))
    self.q.append(sig)
    return self

  def submit(self, device:KFDDevice):
    if not len(self.q): return
    write_ptr, read_ptr = device.amd_aql_queue.write_dispatch_id, device.amd_aql_queue.read_dispatch_id
    if (len(self.q)+write_ptr-read_ptr)*AQL_PACKET_SIZE > device.aql_ring.size: raise RuntimeError("AQL queue overrun")
    for cmd in self.q:
      ring_addr = device.aql_ring.va_addr + (write_ptr*AQL_PACKET_SIZE) % device.aql_ring.size
      ctypes.memmove(ring_addr, ctypes.addressof(cmd), AQL_PACKET_SIZE)
      write_ptr += 1
    # TODO: add CPU memory barrier here
    device.amd_aql_queue.write_dispatch_id = write_ptr
    device.aql_doorbell[0] = device.aql_doorbell_value + len(self.q) - 1
    device.aql_doorbell_value += len(self.q)
    return self

# prebuilt sdma packets
sdma_flush_hdp_pkt = sdma_pkts.hdp_flush(0x8, 0x0, 0x80000000, 0x0, 0x0, 0x0)
sdma_cache_inv = sdma_pkts.gcr(op=amd_gpu.SDMA_OP_GCR, sub_op=amd_gpu.SDMA_SUBOP_USER_GCR, GCR_CONTROL_GL2_WB=1, GCR_CONTROL_GLK_WB=1,
                              GCR_CONTROL_GL2_INV=1, GCR_CONTROL_GL1_INV=1, GCR_CONTROL_GLV_INV=1, GCR_CONTROL_GLK_INV=1,
                              GCR_CONTROL_GL2_RANGE=0)
sdma_cache_wb = sdma_pkts.gcr(op=amd_gpu.SDMA_OP_GCR, sub_op=amd_gpu.SDMA_SUBOP_USER_GCR, GCR_CONTROL_GL2_WB=1, GCR_CONTROL_GLK_WB=1,
                              GCR_CONTROL_GL2_RANGE=0)

class HWCopyQueue:
  def __init__(self): self.q = []

  def submit(self, device:KFDDevice):
    read_ptr = device.sdma_read_pointer[0]
    if (device.sdma_doorbell_value-read_ptr) > device.sdma_ring.size: raise RuntimeError("SDMA queue overrun")
    for cmd in self.q:
      if (cmdsz:=ctypes.sizeof(cmd)) > (fill:=device.sdma_ring.size - device.sdma_doorbell_value % device.sdma_ring.size):
        ctypes.memset(device.sdma_ring.va_addr + (device.sdma_doorbell_value % device.sdma_ring.size), 0, fill)
        device.sdma_doorbell_value += fill
      ctypes.memmove(device.sdma_ring.va_addr + (device.sdma_doorbell_value % device.sdma_ring.size), ctypes.addressof(cmd), cmdsz)
      device.sdma_doorbell_value += cmdsz
    device.sdma_write_pointer[0] = device.sdma_doorbell_value
    device.sdma_doorbell[0] = device.sdma_doorbell_value
    return self

  def timestamp(self, addr):
    self.q.append(sdma_pkts.timestamp(op=amd_gpu.SDMA_OP_TIMESTAMP, sub_op=amd_gpu.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL, addr=addr))
    return self

  def copy(self, dest, src, copy_size):
    self.q.append(sdma_flush_hdp_pkt)  # TODO: do I need this?
    self.q.append(sdma_cache_inv)
    copied = 0
    copies_commands = (copy_size + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE
    for _ in range(copies_commands):
      step_copy_size = min(copy_size - copied, SDMA_MAX_COPY_SIZE)
      self.q.append(sdma_pkts.copy_linear(op=amd_gpu.SDMA_OP_COPY, sub_op=amd_gpu.SDMA_SUBOP_COPY_LINEAR,
                                          count=step_copy_size-1, src_addr=src+copied, dst_addr=dest+copied))
      copied += step_copy_size
    self.q.append(sdma_cache_wb)
    return self

  def signal(self, signal:hsa.amd_signal_t):
    signal.value = 1
    self.q.append(sdma_pkts.atomic(op=amd_gpu.SDMA_OP_ATOMIC, operation=amd_gpu.SDMA_ATOMIC_ADD64,
                                   addr=ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET, src_data=(1<<64)-1))
    if signal.event_mailbox_ptr != 0:
      self.q.append(sdma_pkts.fence(op=amd_gpu.SDMA_OP_FENCE, mtype=3, addr=signal.event_mailbox_ptr, data=signal.event_id))
      self.q.append(sdma_pkts.trap(op=amd_gpu.SDMA_OP_TRAP, int_ctx=signal.event_id))
    return self

  def wait(self, signal:hsa.amd_signal_t):
    self.q.append(sdma_pkts.poll_regmem(op=amd_gpu.SDMA_OP_POLL_REGMEM, mem_poll=1, func=0x3,
                                        addr=ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET,
                                        value=0, mask=0xffffffff, interval=0x04, retry_count=0xfff))
    return self

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
    if self.device.kernargs_ptr + self.kernargs_segment_size > (self.device.kernargs.va_addr + self.device.kernargs.size):
      self.device.kernargs_ptr = self.device.kernargs.va_addr
    assert self.device.kernargs_ptr + self.kernargs_segment_size <= (self.device.kernargs.va_addr + self.device.kernargs.size), "kernargs overrun"
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
        raise RuntimeError(f"HSAProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")
    args_st = self.args_struct_t.from_address(self.device.kernargs_ptr)
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].va_addr)
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

    HWComputeQueue().exec(self, self.device.kernargs_ptr, global_size, local_size,
                          self.device.completion_signal if wait else None).submit(self.device)
    self.device.kernargs_ptr += self.kernargs_segment_size

    if wait:
      self.device._wait_signal(self.device.completion_signal)
      assert (wp:=self.device.amd_aql_queue.write_dispatch_id) == (rp:=self.device.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"
      return (self.device.completion_signal.end_ts-self.device.completion_signal.start_ts)/1e8

class KFDAllocator(LRUAllocator):
  def __init__(self, device:KFDDevice):
    self.device = device
    # NOTE: KFD_IOC_ALLOC_MEM_FLAGS_GTT doesn't work here for readinto
    self.b = [self.device._gpu_alloc(SDMA_MAX_COPY_SIZE*4, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True) for _ in range(2)]
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    try:
      if options.host: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True)
      else: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    except OSError as e:
      if e.errno == errno.ENOMEM: raise MemoryError("Cannot allocate memory") from e
      else: raise

  def _free(self, gpumem, options:BufferOptions): self.device._gpu_free(gpumem)
  #def as_buffer(self, src:Any) -> memoryview:
  #  self.device.synchronize()
  #  return to_mv(src.va_addr, src.size)

  #def copy_from_fd(self, dest, fd, offset, size):
  #  fo = io.FileIO(fd, "a+b", closefd=False)
  #  fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
  #  copied_in, total_copy_size = 0, round_up(size+minor_offset, PAGE_SIZE)
  #  for i in range(0, size+minor_offset, self.b[0].size):
  #    local_size = min(self.b[0].size, total_copy_size-i)
  #    copy_size = min(local_size-minor_offset, size-copied_in)
  #    if copy_size == 0: break

  #    fo.readinto(to_mv(self.b[1].va_addr, local_size))
  #    if i != 0: self.device._wait_signal(self.device.signal_sdma)
  #    self.b = self.b[::-1]
  #    self.device._submit_sdma(dest.va_addr+copied_in, self.b[0].va_addr+minor_offset, copy_size, completion_signal=self.device.signal_sdma)

  #    copied_in += copy_size
  #    minor_offset = 0 # only on the first
  #  self.device._wait_signal(self.device.signal_sdma)

  def copyin(self, dest, src: memoryview):
    for i in range(0, src.nbytes, self.b[0].size):
      ctypes.memmove(self.b[1].va_addr, from_mv(src[i:]), lsize:=min(self.b[0].size, src.nbytes-i))
      if i != 0: self.device._wait_signal(self.device.signal_sdma)
      self.b = self.b[::-1]
      self.device._submit_sdma(dest.va_addr+i, self.b[0].va_addr, lsize, completion_signal=self.device.signal_sdma)
    self.device._wait_signal(self.device.signal_sdma)

  def copyout(self, dest:memoryview, src):
    self.device.synchronize()
    for i in range(0, dest.nbytes, self.b[0].size):
      self.device._submit_sdma(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i), completion_signal=self.device.signal_sdma)
      self.device._wait_signal(self.device.signal_sdma)
      ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest, src, sz:int, src_dev:KFDDevice, dest_dev:KFDDevice):
    dest_dev._gpu_map(src)
    q = HWComputeQueue().signal(sig := KFDDevice._get_signal())
    HWCopyQueue().wait(sig).copy(dest.va_addr, src.va_addr, sz).signal(sigc := KFDDevice._get_signal()).submit(dest_dev)
    HWComputeQueue().wait(sigc).submit(dest_dev)
    q.wait(sigc).submit(src_dev)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class KFDDevice(Compiled):
  kfd:int = -1
  event_page:Any = None  # TODO: fix types in kfd, Optional[kfd.struct_kfd_ioctl_alloc_memory_of_gpu_args]
  signals_page:Any = None
  signal_number:int = 16
  gpus:List[pathlib.Path] = []

  def _gpu_map(self, mem):
    if self.gpu_id in getattr(mem, "mapped_gpu_ids", []): return
    mem.__setattr__("mapped_gpu_ids", getattr(mem, "mapped_gpu_ids", []) + [self.gpu_id])
    c_gpus = (ctypes.c_int32 * len(mem.mapped_gpu_ids))(*mem.mapped_gpu_ids)
    stm = kio.map_memory_to_gpu(self.kfd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=len(mem.mapped_gpu_ids))
    assert stm.n_success == len(mem.mapped_gpu_ids)

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
      assert addr == buf == mem.va_addr
    if map_to_gpu: self._gpu_map(mem)
    return mem

  def _gpu_free(self, mem):
    if len(gpus:=getattr(mem, "mapped_gpu_ids", [])):
      c_gpus = (ctypes.c_int32 * len(gpus))(*gpus)
      stm = kio.unmap_memory_from_gpu(self.kfd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=len(gpus))
      assert stm.n_success == len(gpus)
    libc.munmap(mem.va_addr, mem.size)
    kio.free_memory_of_gpu(self.kfd, handle=mem.handle)

  @classmethod
  def _get_signal(self, num=None, sync_event=None) -> hsa.amd_signal_t:
    if num is None:
      num = KFDDevice.signal_number
      KFDDevice.signal_number += 1
      if KFDDevice.signal_number == SIGNAL_COUNT: KFDDevice.signal_number = 16
    ret = hsa.amd_signal_t.from_address(KFDDevice.signals_page.va_addr + SIGNAL_SIZE*num)
    ret.kind = hsa.AMD_SIGNAL_KIND_USER
    if sync_event is not None:
      ret.event_mailbox_ptr = KFDDevice.event_page.va_addr + sync_event.event_slot_index*8
      ret.event_id = sync_event.event_id
    return ret

  @classmethod
  def _wait_signal(self, signal:hsa.amd_signal_t, timeout=60000):
    assert signal.event_id != 0, "can't wait on this signal"
    evt_arr = (kfd.struct_kfd_event_data * 1)()
    evt_arr[0].event_id = signal.event_id
    ret = kio.wait_events(KFDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=timeout)
    if ret.wait_result != 0: raise RuntimeError(f"wait_result: {ret.wait_result}, {timeout} ms TIMEOUT!")

  def __init__(self, device:str=""):
    if KFDDevice.kfd == -1:
      KFDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
      KFDDevice.gpus = [g.parent for g in pathlib.Path("/sys/devices/virtual/kfd/kfd/topology/nodes").glob("*/gpu_id") if is_usable_gpu(g)]
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    with open(f"{KFDDevice.gpus[self.device_id]}/gpu_id", "r") as f: self.gpu_id = int(f.read())
    with open(f"{KFDDevice.gpus[self.device_id]}/properties", "r") as f: self.properties = {line.split()[0]: int(line.split()[1]) for line in f}
    self.drm_fd = os.open(f"/dev/dri/renderD{self.properties['drm_render_minor']}", os.O_RDWR)
    target = int(self.properties['gfx_target_version'])
    self.arch = "gfx%d%x%x" % (target // 10000, (target // 100) % 100, target % 100)
    kio.acquire_vm(KFDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    if KFDDevice.event_page is None:
      KFDDevice.signals_page = self._gpu_alloc(SIGNAL_SIZE*SIGNAL_COUNT, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      KFDDevice.event_page = self._gpu_alloc(0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      sync_event = kio.create_event(KFDDevice.kfd, event_page_offset=KFDDevice.event_page.handle, auto_reset=1)
    else:
      self._gpu_map(KFDDevice.signals_page)
      self._gpu_map(KFDDevice.event_page)
      sync_event = kio.create_event(KFDDevice.kfd, auto_reset=1)

    self.completion_signal = KFDDevice._get_signal(self.device_id*2, sync_event=sync_event)
    self.signal_sdma = KFDDevice._get_signal(self.device_id*2+1, sync_event=kio.create_event(KFDDevice.kfd, auto_reset=1))

    self.gart_aql = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.gart_sdma = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.aql_ring = self._gpu_alloc(0x100000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.eop_buffer = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.kernargs = self._gpu_alloc(0x1000000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.kernargs_ptr = self.kernargs.va_addr
    self.ctx_save_restore_address = self._gpu_alloc(0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    # AQL Queue
    self.amd_aql_queue = hsa.amd_queue_t.from_address(self.gart_aql.va_addr)
    self.amd_aql_queue.write_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id = 0
    self.amd_aql_queue.read_dispatch_id_field_base_byte_offset = getattr(hsa.amd_queue_t, 'read_dispatch_id').offset
    self.amd_aql_queue.queue_properties = hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING

    self.amd_aql_queue.max_cu_id = self.properties['simd_count'] // self.properties['simd_per_cu'] - 1
    self.amd_aql_queue.max_wave_id = self.properties['max_waves_per_simd'] * self.properties['simd_per_cu'] - 1

    # scratch setup
    self.max_private_segment_size = 4096
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
      write_pointer_address=self.gart_aql.va_addr + getattr(hsa.amd_queue_t, 'write_dispatch_id').offset,
      read_pointer_address=self.gart_aql.va_addr + getattr(hsa.amd_queue_t, 'read_dispatch_id').offset)

    self.doorbells_base = self.aql_queue.doorbell_offset & (~0x1fff)  # doorbell is two pages
    self.doorbells = libc.mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, KFDDevice.kfd, self.doorbells_base)
    self.aql_doorbell = to_mv(self.doorbells + self.aql_queue.doorbell_offset - self.doorbells_base, 4).cast("I")
    self.aql_doorbell_value = 0

    # SDMA Queue
    self.sdma_ring = self._gpu_alloc(0x100000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.sdma_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.sdma_ring.va_addr, ring_size=self.sdma_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_SDMA, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      write_pointer_address=self.gart_sdma.va_addr, read_pointer_address=self.gart_sdma.va_addr+8)

    self.sdma_read_pointer = to_mv(self.sdma_queue.read_pointer_address, 8).cast("Q")
    self.sdma_write_pointer = to_mv(self.sdma_queue.write_pointer_address, 8).cast("Q")
    self.sdma_doorbell = to_mv(self.doorbells + self.sdma_queue.doorbell_offset - self.doorbells_base, 4).cast("I")
    self.sdma_doorbell_value = 0

    # PM4 stuff
    self.pm4_indirect_buf = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    pm4_indirect_cmd = (ctypes.c_uint32*13)(amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), self.pm4_indirect_buf.va_addr & 0xffffffff,
                                            (self.pm4_indirect_buf.va_addr>>32) & 0xffffffff, 8 | amd_gpu.INDIRECT_BUFFER_VALID, 0xa)
    ctypes.memmove(ctypes.addressof(pm4_cmds:=(ctypes.c_uint16*27)(1))+2, ctypes.addressof(pm4_indirect_cmd), ctypes.sizeof(pm4_indirect_cmd))
    self.pm4_packet = hsa.hsa_ext_amd_aql_pm4_packet_t(header=VENDOR_HEADER, pm4_command=pm4_cmds,
                                                       completion_signal=hsa.hsa_signal_t(ctypes.addressof(self.completion_signal)))

    super().__init__(device, KFDAllocator(self), KFDCompiler(self.arch), functools.partial(KFDProgram, self))

  def _submit_sdma(self, dest, src, copy_size, wait_signals=None, completion_signal=None):
    q = HWCopyQueue()
    if wait_signals is not None:
      # NOTE: we check only low 32 bits to be zeroed, we don't use higher values for signals
      for sig in wait_signals: q.wait(ctypes.addressof(sig) + getattr(hsa.amd_signal_t, 'value').offset)
    if completion_signal is not None: q.timestamp(ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'start_ts').offset)
    q.copy(dest, src, copy_size)
    if completion_signal is not None: q.timestamp(ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'end_ts').offset)
    if completion_signal is not None: q.signal(completion_signal)
    q.submit(self)

  def _submit_cache_inv(self, addr=0x0, sz=(1 << 64)-1, gli=0, glv=0, glk=0, gl1=0, gl2=0):
    pm4_buffer_view = to_mv(self.pm4_indirect_buf.va_addr, 0x1000).cast("I")
    pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0,
               sz & 0xffffffff, (sz >> 32) & 0xff, addr & 0xffffffff, (addr >> 32) & 0xffffff, 0,
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2)]
    for i, value in enumerate(pm4_cmd): pm4_buffer_view[i] = value
    q = HWComputeQueue()
    q.q.append(self.pm4_packet)
    q.submit(self)
    self._wait_signal(self.completion_signal)
    assert (wp:=self.amd_aql_queue.write_dispatch_id) == (rp:=self.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"

  def synchronize(self):
    HWComputeQueue().signal(self.completion_signal).submit(self)
    self._wait_signal(self.completion_signal)
    assert (wp:=self.amd_aql_queue.write_dispatch_id) == (rp:=self.amd_aql_queue.read_dispatch_id), f"didn't run {wp} != {rp}"

    # reset kernargs
    self.kernargs_ptr = self.kernargs.va_addr
