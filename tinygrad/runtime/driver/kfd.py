from __future__ import annotations
from typing import Tuple
import os, fcntl, ctypes, functools, re, pathlib, mmap
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
import tinygrad.runtime.autogen.amd_sdma as amd_sdma

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

class SDMAQueue:
  max_copy_size = 0x400000

  def __init__(self, device, size=(1<<20)):
    from tinygrad.runtime.ops_kfd import KFDDevice # this is not the best solution, need to better arrange files
    self.device = device
    self.ring_size = size
    self.sdma_ring = device._gpu_alloc(self.ring_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
    self.ptrs = device._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    # self.write_pointer = ctypes.cast(address_as_uint64, ctypes.POINTER(ctypes.c_int))
    self.sdma_queue = kio.create_queue(KFDDevice.kfd, ring_base_address=self.sdma_ring.va_addr, ring_size=self.sdma_ring.size, gpu_id=device.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_SDMA, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      write_pointer_address=self.ptrs.va_addr + 0x0, read_pointer_address=self.ptrs.va_addr + 0x8)

    self.read_pointer = ctypes.cast(self.sdma_queue.read_pointer_address, ctypes.POINTER(ctypes.c_uint64))
    self.write_pointer = ctypes.cast(self.sdma_queue.write_pointer_address, ctypes.POINTER(ctypes.c_uint64))
    print("dma", hex(self.sdma_queue.doorbell_offset))
    # dmmp = libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, KFDDevice.kfd, self.sdma_queue.doorbell_offset)
    # print(hex(dmmp))
    # self.doorbell = to_mv(dmmp, 8192).cast("I")
    self.doorbell_value = 0

  def _build_poll_cmd(self, addr, value):
    cmd = amd_sdma.SDMA_PKT_POLL_REGMEM.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))
    cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_POLL_REGMEM
    cmd.HEADER_UNION.mem_poll = 1
    cmd.HEADER_UNION.func = 0x3 # is equal
    cmd.ADDR_LO_UNION.addr_31_0 = addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (addr >> 32) & 0xffffffff

    cmd.VALUE_UNION.value = value

    cmd.MASK_UNION.mask = 0xffffffff; # the whole content.

    cmd.DW5_UNION.interval = 0x04
    cmd.DW5_UNION.retry_count = 0xfff # retry forever.

    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_POLL_REGMEM)

  def _build_atomic_dec_cmd(self, addr):
    cmd = amd_sdma.SDMA_PKT_ATOMIC.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))

    cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_ATOMIC
    cmd.HEADER_UNION.operation = amd_sdma.SDMA_ATOMIC_ADD64
    cmd.ADDR_LO_UNION.addr_31_0 = addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (addr >> 32) & 0xffffffff

    cmd.SRC_DATA_LO_UNION.src_data_31_0 = 0xffffffff
    cmd.SRC_DATA_HI_UNION.src_data_63_32 = 0xffffffff

    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_ATOMIC)
    return cmd

  def _build_cache_cmd(self, invalidate=False):
    cmd = amd_sdma.SDMA_PKT_GCR.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))

    cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_GCR
    cmd.HEADER_UNION.sub_op = amd_sdma.SDMA_SUBOP_USER_GCR
    cmd.WORD2_UNION.GCR_CONTROL_GL2_WB = 1
    cmd.WORD2_UNION.GCR_CONTROL_GLK_WB = 1

    if invalidate:
      cmd.WORD2_UNION.GCR_CONTROL_GL2_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GL1_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GLV_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GLK_INV = 1

    # TODO: They inv the whole cache, try the required part only?
    cmd.WORD2_UNION.GCR_CONTROL_GL2_RANGE = 0

    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_GCR)
    return cmd

  def _build_hdp_cmd(self):
    cmd = amd_sdma.SDMA_PKT_HDP_FLUSH.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))
    cmd.DW_0_DATA = 0x8
    cmd.DW_1_DATA = 0x0
    cmd.DW_2_DATA = 0x80000000
    cmd.DW_3_DATA = 0x0
    cmd.DW_4_DATA = 0x0
    cmd.DW_5_DATA = 0x0

    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_HDP_FLUSH)

  def _build_fence_cmd(self, fence_addr, value):
    cmd = amd_sdma.SDMA_PKT_FENCE.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))
    cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_FENCE
    cmd.ADDR_LO_UNION.addr_31_0 = fence_addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (fence_addr >> 32) & 0xffffffff
    cmd.DATA_UNION.data = value
    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_FENCE)
    return cmd

  def _build_trap_cmd(self, event_id):
    cmd = amd_sdma.SDMA_PKT_TRAP.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))
    cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_TRAP
    cmd.INT_CONTEXT_UNION.int_ctx = event_id
    self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_TRAP)
    return cmd

  def _build_cp_cmd(self, dest, src, sz):
    copies_commands = (sz + SDMAQueue.max_copy_size - 1) // SDMAQueue.max_copy_size
    copied = 0

    for _ in range(copies_commands):
      # print(self.doorbell_value)
      copy_size = min(sz - copied, SDMAQueue.max_copy_size)
      src_off = src + copied
      dest_off = dest + copied

      cmd = amd_sdma.SDMA_PKT_COPY_LINEAR.from_address(self.sdma_ring.va_addr + (self.doorbell_value % self.ring_size))
      cmd.HEADER_UNION.op = amd_sdma.SDMA_OP_COPY
      cmd.HEADER_UNION.sub_op = amd_sdma.SDMA_SUBOP_COPY_LINEAR
      cmd.COUNT_UNION.count = copy_size - 1

      cmd.SRC_ADDR_LO_UNION.src_addr_31_0 = src_off & 0xffffffff
      cmd.SRC_ADDR_HI_UNION.src_addr_63_32 = (src_off >> 32) & 0xffffffff

      cmd.DST_ADDR_LO_UNION.dst_addr_31_0 = dest_off & 0xffffffff
      cmd.DST_ADDR_HI_UNION.dst_addr_63_32 = (dest_off >> 32) & 0xffffffff

      copied += copy_size
      self.doorbell_value += ctypes.sizeof(amd_sdma.SDMA_PKT_COPY_LINEAR)

  def _ring_doorbell(self):
    # print(self.read_pointer[0], self.doorbell_value, self.device.doorbell[0x101])
    self.write_pointer[0] = self.doorbell_value
    self.device.doorbell[0x202] = self.doorbell_value

  def submit_copy(self, dest, src, nbytes, wait_signals:Optional[List[hsa.amd_signal_t]]=None, completion_signal:Optional[hsa.amd_signal_t]=None):
    # print(dest, src)
    if wait_signals is not None:
      for sig in wait_signals:
        self._build_poll_cmd(ctypes.addressof(sig) + getattr(hsa.amd_signal_t, 'value').offset, 0)
        self._build_poll_cmd(ctypes.addressof(sig) + getattr(hsa.amd_signal_t, 'value').offset + 4, 0)

    self._build_hdp_cmd()
    self._build_cache_cmd(invalidate=True)
    self._build_cp_cmd(dest, src, nbytes)
    self._build_cache_cmd()

    # Signal we have finished.
    if completion_signal is not None:
      # print("here")
      # check(hsa.hsa_amd_signal_value_pointer(completion_signal, ctypes.byref(val_ptr := ctypes.POINTER(ctypes.c_int64)())))
      # print("completion_signal", hex(ctypes.addressof(completion_signal)))
      # print("cmp", completion_signal.value, hex(ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'value').offset))
      self._build_atomic_dec_cmd(ctypes.addressof(completion_signal) + getattr(hsa.amd_signal_t, 'value').offset)
      self._build_fence_cmd(completion_signal.event_mailbox_ptr, completion_signal.event_id)
      self._build_trap_cmd(completion_signal.event_id)

    self._ring_doorbell()
