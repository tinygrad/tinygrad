from __future__ import annotations
import os, ctypes, contextlib, fcntl, mmap
from dataclasses import dataclass
from tinygrad.runtime.autogen import nv_570 as nv_gpu, tegra_36 as tegra
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface, FileIOInterface
from tinygrad.runtime.support.system import MAP_FIXED # Force mmap to fixed address
from tinygrad.helpers import round_up, to_mv

_NVMAP_FREE = (ord('N') << 8) | 4  # _IO('N', 4): handle passed as integer, not struct
_NVMAP_TAG = 0x0900

def _nvmap_buf(nvmap_fd, size, cache_flags, align=4096):
  c = tegra.NVMAP_IOC_CREATE(nvmap_fd, size=size)
  tegra.NVMAP_IOC_ALLOC(nvmap_fd, handle=c.handle, heap_mask=tegra.NVMAP_HEAP_IOVMM, flags=(_NVMAP_TAG << 16) | cache_flags, align=align)
  return c.handle, tegra.NVMAP_IOC_GET_FD(nvmap_fd, handle=c.handle).size

@dataclass(eq=False)
class TegraMem:
  handle: int = 0        # nvmap handle (kernel tracking)
  dmabuf_fd: int = -1    # dmabuf file descriptor (for mmap + GPU map)
  gpu_va: int = 0        # GPU virtual address (from MAP_BUFFER_EX)
  size: int = 0
  cpu_addr: int = 0      # CPU virtual address (from mmap)
  hMemory: int = 0       # synthetic RM handle (for gpfifo_area lookup)

class TegraIface:
  _nvmap_fd, _ctrl_fd, _gpu_info = -1, -1, None

  def __init__(self, dev, device_id):
    if device_id != 0: raise RuntimeError("TegraIface only supports device 0 (single iGPU)")
    if TegraIface._gpu_info is None:
      if not os.path.exists("/dev/nvgpu/igpu0/ctrl"): raise FileNotFoundError("/dev/nvgpu/igpu0/ctrl")
      TegraIface._nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
      TegraIface._ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
      tegra_gpu_info = tegra.nvgpu_gpu_characteristics()
      tegra.NVGPU_GPU_IOCTL_GET_CHARACTERISTICS(TegraIface._ctrl_fd,
        buf_size=ctypes.sizeof(tegra_gpu_info), buf_addr=ctypes.addressof(tegra_gpu_info))
      TegraIface._gpu_info = tegra_gpu_info

    self.dev, self.device_id = dev, device_id
    tegra_gpu_info = TegraIface._gpu_info
    self.compute_class, self.gpfifo_class, self.dma_class = tegra_gpu_info.compute_class, tegra_gpu_info.gpfifo_class, tegra_gpu_info.dma_copy_class
    self.viddec_class = None # no video decoder on Tegra
    self._handle_cnt = 0x10000 # (NVKIface starts at 0x1000) TinyGrad internal bookkeeping (RM's rm_alloc)
    self._as_fd, self._tsg_fd, self._subctx_veid = -1, -1, 0
    self._ch_fds: dict[int, int] = {} # handle -> channel fd
    self._ws_tokens: dict[int, int] = {} # handle -> work_submit_token
    self._ch_bufs: dict[int, tuple[int, int, int, int]] = {} # handle -> (ring_nvmap_h, ring_fd, userd_nvmap_h, userd_fd)
    self.root, self.gpu_instance = self._next_handle(), 0
    self._allocs: list[TegraMem] = []

  # RM would get integer handle from kernel, we mimic this with int handles (start at 0x10000 to avoid collisions with NVKIface that starts at 0x1000)
  # Used to lookup gpfifo_area
  def _next_handle(self) -> int:
    self._handle_cnt += 1
    return self._handle_cnt

  # map RM's alloc to TegraIface alloc
  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    handle = self._next_handle()

    # stub/no-op (They exist for RM but no nvgpu driver equivalent)
    if clss in (nv_gpu.NV01_DEVICE_0, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.FERMI_VASPACE_A, nv_gpu.NV01_ROOT_CLIENT, nv_gpu.NV01_ROOT,
                nv_gpu.NV1_MEMORY_SYSTEM, nv_gpu.NV1_MEMORY_USER, nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR):
      return handle

    # open GPU address space and pre-reserve two 1GB VA windows tinygrad uses for fixed-address buffers
    if clss == nv_gpu.NV01_MEMORY_VIRTUAL:
      self._as_fd = tegra.NVGPU_GPU_IOCTL_ALLOC_AS(self._ctrl_fd, flags=2, va_range_start=(1 << 21), va_range_end=(1 << 40) - (1 << 21)).as_fd
      for reserved_va in [0xFD00000000, 0xFE00000000]:
        tegra.NVGPU_AS_IOCTL_ALLOC_SPACE(self._as_fd, pages=0x40000000 // mmap.PAGESIZE, page_size=mmap.PAGESIZE, flags=1, offset=reserved_va)
      return handle

    # open TSG (time-sliced GPU group); channels bind to it for scheduling
    if clss == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      self._tsg_fd = tegra.NVGPU_GPU_IOCTL_OPEN_TSG(self._ctrl_fd).tsg_fd
      return handle

    # create subcontext; VEID lets the GR engine distinguish address spaces within the TSG
    if clss == nv_gpu.FERMI_CONTEXT_SHARE_A:
      self._subctx_veid = tegra.NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT(self._tsg_fd, type=1, as_fd=self._as_fd).veid
      return handle

    # open channel fd, bind to AS + TSG, overlay ring and USERD dmabufs into gpfifo_area via MAP_FIXED
    if clss in (self.gpfifo_class, nv_gpu.AMPERE_CHANNEL_GPFIFO_A):
      ch_fd = tegra.NVGPU_GPU_IOCTL_OPEN_CHANNEL(self._ctrl_fd, channel_fd=-1).channel_fd
      tegra.NVGPU_AS_IOCTL_BIND_CHANNEL(self._as_fd, channel_fd=ch_fd) # attach to address space
      tegra.NVGPU_TSG_IOCTL_BIND_CHANNEL_EX(self._tsg_fd, channel_fd=ch_fd, subcontext_id=self._subctx_veid) # attach to scheduler
      tegra.NVGPU_IOCTL_CHANNEL_WDT(ch_fd, wdt_status=1) # disable watchdog

      gpfifo_entries, gpfifo_buf_handle, gpfifo_va, userd_off = 0x10000, 0, 0, 0
      if params is not None:
        gpfifo_va, gpfifo_entries, gpfifo_buf_handle = params.gpFifoOffset, params.gpFifoEntries, params.hObjectBuffer
        userd_off = params.userdOffset[0]

      # find the gpfifo_area TegraMem by the synthetic RM handle tinygrad assigned it
      gpfifo_area_mem = next((m for m in self._allocs if m.hMemory == gpfifo_buf_handle), None)
      if gpfifo_area_mem is None: raise RuntimeError(f"TegraIface: gpfifo_area alloc not found for handle {gpfifo_buf_handle}")
      ring_sz = gpfifo_entries * 8

      # nvgpu requires separate owned dmabufs for ring and USERD - can't use offsets into gpfifo_area
      ring_nvmap_h, ring_fd = _nvmap_buf(self._nvmap_fd, ring_sz, tegra.NVMAP_HANDLE_WRITE_COMBINE)
      userd_nvmap_h, userd_fd = _nvmap_buf(self._nvmap_fd, 4096, tegra.NVMAP_HANDLE_WRITE_COMBINE)

      # pin ring+USERD into hw channel; work_submit_token is the doorbell value written to wake the GPU
      setup = tegra.NVGPU_IOCTL_CHANNEL_SETUP_BIND(ch_fd, num_gpfifo_entries=gpfifo_entries, gpfifo_dmabuf_fd=ring_fd, userd_dmabuf_fd=userd_fd,
                                                    flags=tegra.NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT|tegra.NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC)

      # punch dmabuf pages into gpfifo_area CPU mapping so tinygrad writes hit the same physical pages the GPU reads
      cpu_base = gpfifo_area_mem.cpu_addr
      FileIOInterface._mmap(cpu_base + (gpfifo_va - gpfifo_area_mem.gpu_va), ring_sz,
                            mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, ring_fd, 0)
      FileIOInterface._mmap(cpu_base + userd_off, 4096,
                            mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, userd_fd, 0)

      self._ch_fds[handle], self._ws_tokens[handle] = ch_fd, setup.work_submit_token
      self._ch_bufs[handle] = (ring_nvmap_h, ring_fd, userd_nvmap_h, userd_fd)
      return handle

    # bind GR (compute) or CE (copy) engine to the channel
    if clss in (self.compute_class, nv_gpu.AMPERE_COMPUTE_B, self.dma_class, nv_gpu.AMPERE_DMA_COPY_B):
      ch_fd = self._ch_fds.get(parent, -1)
      if ch_fd == -1: raise RuntimeError(f"TegraIface: no channel fd for handle {parent}")
      obj_cls = self.compute_class if clss in (self.compute_class, nv_gpu.AMPERE_COMPUTE_B) else self.dma_class
      tegra.NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX(ch_fd, class_num=obj_cls)
      return handle

    return handle

  def rm_control(self, obj, cmd, params=None, **kwargs):
    if cmd == nv_gpu.NV2080_CTRL_CMD_PERF_BOOST:
      try:
        with open("/sys/class/devfreq/17000000.gpu/max_freq") as f: mx = f.read().strip()
        with open("/sys/class/devfreq/17000000.gpu/min_freq", "w") as f: f.write(mx)
      except OSError: pass
      return params

    if cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      if params is not None: params.workSubmitToken = self._ws_tokens.get(obj, 0)
      return params

    if cmd == nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO:
      tegra_gpu_info = TegraIface._gpu_info
      assert tegra_gpu_info is not None
      info_map = {
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS: tegra_gpu_info.num_gpc,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC: tegra_gpu_info.num_tpc_per_gpc,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC: 2,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM: tegra_gpu_info.sm_arch_warp_count,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_SM_VERSION: tegra_gpu_info.sm_arch_sm_version,
      }
      if params is not None:
        for i in range(params.grInfoListSize):
          info = nv_gpu.NV2080_CTRL_GR_INFO.from_address(params.grInfoList + i * ctypes.sizeof(nv_gpu.NV2080_CTRL_GR_INFO))
          info.data = info_map.get(info.index, 0)
      return params

    if cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST:
      if params is not None:
        known = [self.compute_class, self.gpfifo_class, self.dma_class, nv_gpu.TURING_USERMODE_A]
        if params.numClasses == 0: params.numClasses = len(known)
        else:
          cl = to_mv(params.classList, params.numClasses * 4).cast('I')
          for i, c in enumerate(known[:params.numClasses]): cl[i] = c
      return params

    if cmd == nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO:
      if params is not None:
        params.length = 16
        for i in range(16): params.data[i] = (0x4A + i) & 0xFF
      return params

    return params

  def setup_usermode(self):
    addr = FileIOInterface._mmap(0, 0x10000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, self._ctrl_fd, 0)
    return 0, MMIOInterface(addr, 0x10000, fmt='I')

  def setup_vm(self, vaspace): pass # nvgpu owns the AS, no separate UVM step needed
  def setup_gpfifo_vm(self, gpfifo): pass # handled internally by nvgpu on channel bind

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, **kwargs) -> HCQBuffer:
    # 2MB huge pages for large (>=8MB) buffers to reduce TLB pressure; 4KB otherwise
    alloc_align = mmap.PAGESIZE if (uncached or host) else ((2 << 20) if size >= (8 << 20) else mmap.PAGESIZE)
    size = round_up(size, alloc_align)
    # WRITE_COMBINE for CPU-polled bufs (notifiers/signals) - no cache, GPU writes visible immediately
    # INNER_CACHEABLE for compute data - Orin's interconnect provides hardware IO coherence, no manual flushes
    cache = tegra.NVMAP_HANDLE_WRITE_COMBINE if (uncached or host) else tegra.NVMAP_HANDLE_INNER_CACHEABLE

    handle, dmabuf_fd = _nvmap_buf(self._nvmap_fd, size, cache, alloc_align)

    # GPU mapping: map dmabuf into GPU's address space
    gpu_va = 0
    if self._as_fd >= 0:
      gpu_va = tegra.NVGPU_AS_IOCTL_MAP_BUFFER_EX(self._as_fd, compr_kind=-1, dmabuf_fd=dmabuf_fd, page_size=mmap.PAGESIZE).offset

    # CPU mapping: try MAP_FIXED at the GPU VA for CPU VA = GPU VA, fallback is kernel chosen address (PCIface pattern)
    addr = FileIOInterface._mmap(gpu_va or 0, size, mmap.PROT_READ | mmap.PROT_WRITE,
                                 mmap.MAP_SHARED | (MAP_FIXED if gpu_va else 0), dmabuf_fd, 0)
    if gpu_va and addr != gpu_va:
      addr = FileIOInterface._mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, dmabuf_fd, 0)

    # track alloc metadata
    meta = TegraMem(handle=handle, dmabuf_fd=dmabuf_fd, gpu_va=gpu_va, size=size, cpu_addr=addr, hMemory=self._next_handle())
    self._allocs.append(meta) # track allocs for free() and gpfifo_area lookup
    return HCQBuffer(va_addr=gpu_va, size=size, meta=meta, view=MMIOInterface(addr, size, fmt='B'), owner=self.dev)

  def free(self, mem:HCQBuffer):
    meta = mem.meta
    if meta.gpu_va and self._as_fd >= 0:
      with contextlib.suppress(OSError): tegra.NVGPU_AS_IOCTL_UNMAP_BUFFER(self._as_fd, offset=meta.gpu_va)
    if meta.cpu_addr:
      with contextlib.suppress(Exception): FileIOInterface.munmap(meta.cpu_addr, meta.size)
    if meta.dmabuf_fd >= 0:
      with contextlib.suppress(OSError): os.close(meta.dmabuf_fd)
    if meta.handle and self._nvmap_fd >= 0:
      with contextlib.suppress(OSError): fcntl.ioctl(self._nvmap_fd, _NVMAP_FREE, ctypes.c_int32(meta.handle).value)
    if meta in self._allocs: self._allocs.remove(meta)

  def map(self, mem:HCQBuffer): pass # already mapped to GPU as in alloc()
  def sleep(self, tm:int): pass # no GSP firmware on Tegra iGPU so no status queue to service during long waits

  def device_fini(self):
    for ch_fd in self._ch_fds.values():
      with contextlib.suppress(OSError): os.close(ch_fd)
    for ring_nvmap_h, ring_fd, userd_nvmap_h, userd_fd in self._ch_bufs.values():
      for fd in (ring_fd, userd_fd):
        with contextlib.suppress(OSError): os.close(fd)
      for h in (ring_nvmap_h, userd_nvmap_h):
        with contextlib.suppress(OSError): fcntl.ioctl(TegraIface._nvmap_fd, _NVMAP_FREE, ctypes.c_int32(h).value)
    if self._tsg_fd >= 0: os.close(self._tsg_fd)
    if self._as_fd >= 0: os.close(self._as_fd)
    self._tsg_fd, self._as_fd = -1, -1
    if TegraIface._ctrl_fd >= 0: os.close(TegraIface._ctrl_fd)
    if TegraIface._nvmap_fd >= 0: os.close(TegraIface._nvmap_fd)
    TegraIface._ctrl_fd, TegraIface._nvmap_fd, TegraIface._gpu_info = -1, -1, None
