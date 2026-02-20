from __future__ import annotations
import os, ctypes, contextlib, fcntl, mmap
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface, FileIOInterface
from tinygrad.runtime.support.system import MAP_FIXED
from tinygrad.helpers import round_up, to_mv

def _ct(fields):
  return type("_s", (ctypes.Structure,), {"_fields_": fields})

# struct nvgpu_gpu_characteristics from include/uapi/linux/nvgpu-ctrl.h
_nvgpu_gpu_characteristics = _ct([
  ("arch", ctypes.c_uint32), ("impl", ctypes.c_uint32), ("rev", ctypes.c_uint32), ("num_gpc", ctypes.c_uint32),
  ("numa_domain_id", ctypes.c_int32),
  ("L2_cache_size", ctypes.c_uint64), ("on_board_video_memory_size", ctypes.c_uint64),
  ("num_tpc_per_gpc", ctypes.c_uint32), ("bus_type", ctypes.c_uint32), ("big_page_size", ctypes.c_uint32),
  ("compression_page_size", ctypes.c_uint32), ("pde_coverage_bit_count", ctypes.c_uint32), ("available_big_page_sizes", ctypes.c_uint32),
  ("flags", ctypes.c_uint64),
  ("twod_class", ctypes.c_uint32), ("threed_class", ctypes.c_uint32), ("compute_class", ctypes.c_uint32),
  ("gpfifo_class", ctypes.c_uint32), ("inline_to_memory_class", ctypes.c_uint32), ("dma_copy_class", ctypes.c_uint32),
  ("gpc_mask", ctypes.c_uint32), ("sm_arch_sm_version", ctypes.c_uint32), ("sm_arch_spa_version", ctypes.c_uint32),
  ("sm_arch_warp_count", ctypes.c_uint32),
  ("gpu_ioctl_nr_last", ctypes.c_int16), ("tsg_ioctl_nr_last", ctypes.c_int16), ("dbg_gpu_ioctl_nr_last", ctypes.c_int16),
  ("ioctl_channel_nr_last", ctypes.c_int16), ("as_ioctl_nr_last", ctypes.c_int16), ("gpu_va_bit_count", ctypes.c_uint8), ("reserved", ctypes.c_uint8),
  ("max_fbps_count", ctypes.c_uint32), ("fbp_en_mask", ctypes.c_uint32), ("emc_en_mask", ctypes.c_uint32),
  ("max_ltc_per_fbp", ctypes.c_uint32), ("max_lts_per_ltc", ctypes.c_uint32), ("max_tex_per_tpc", ctypes.c_uint32),
  ("max_gpc_count", ctypes.c_uint32), ("rop_l2_en_mask_DEPRECATED", ctypes.c_uint32 * 2), ("chipname", ctypes.c_uint8 * 8),
  ("gr_compbit_store_base_hw", ctypes.c_uint64),
  ("gr_gobs_per_comptagline_per_slice", ctypes.c_uint32), ("num_ltc", ctypes.c_uint32), ("lts_per_ltc", ctypes.c_uint32),
  ("cbc_cache_line_size", ctypes.c_uint32), ("cbc_comptags_per_line", ctypes.c_uint32), ("map_buffer_batch_limit", ctypes.c_uint32),
  ("max_freq", ctypes.c_uint64),
  ("graphics_preemption_mode_flags", ctypes.c_uint32), ("compute_preemption_mode_flags", ctypes.c_uint32),
  ("default_graphics_preempt_mode", ctypes.c_uint32), ("default_compute_preempt_mode", ctypes.c_uint32),
  ("local_video_memory_size", ctypes.c_uint64),
  ("pci_vendor_id", ctypes.c_uint16), ("pci_device_id", ctypes.c_uint16), ("pci_subsystem_vendor_id", ctypes.c_uint16),
  ("pci_subsystem_device_id", ctypes.c_uint16), ("pci_class", ctypes.c_uint16), ("pci_revision", ctypes.c_uint8),
  ("vbios_oem_version", ctypes.c_uint8),
  ("vbios_version", ctypes.c_uint32), ("reg_ops_limit", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
  ("event_ioctl_nr_last", ctypes.c_int16), ("pad", ctypes.c_uint16), ("max_css_buffer_size", ctypes.c_uint32),
  ("ctxsw_ioctl_nr_last", ctypes.c_int16), ("prof_ioctl_nr_last", ctypes.c_int16), ("nvs_ioctl_nr_last", ctypes.c_int16),
  ("reserved2", ctypes.c_uint8 * 2),
  ("max_ctxsw_ring_buffer_size", ctypes.c_uint32), ("reserved3", ctypes.c_uint32), ("per_device_identifier", ctypes.c_uint64),
  ("num_ppc_per_gpc", ctypes.c_uint32), ("max_veid_count_per_tsg", ctypes.c_uint32),
  ("num_sub_partition_per_fbpa", ctypes.c_uint32), ("gpu_instance_id", ctypes.c_uint32),
  ("gr_instance_id", ctypes.c_uint32), ("max_gpfifo_entries", ctypes.c_uint32),
  ("max_dbg_tsg_timeslice", ctypes.c_uint32), ("reserved5", ctypes.c_uint32), ("device_instance_id", ctypes.c_uint64)])
_nvgpu_gpu_get_characteristics = _ct([("buf_size", ctypes.c_uint64), ("buf_addr", ctypes.c_uint64)])
_nvmap_handle = _ct([("size", ctypes.c_uint32), ("handle", ctypes.c_uint32)])
_nvmap_alloc = _ct([("handle", ctypes.c_uint32), ("heap_mask", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                     ("align", ctypes.c_uint32), ("numa_nid", ctypes.c_int32)])
_nvgpu_alloc_as = _ct([("big_page_size", ctypes.c_uint32), ("as_fd", ctypes.c_int32), ("flags", ctypes.c_uint32),
                        ("reserved", ctypes.c_uint32), ("va_range_start", ctypes.c_uint64), ("va_range_end", ctypes.c_uint64),
                        ("va_range_split", ctypes.c_uint64), ("padding", ctypes.c_uint32 * 6)])
_nvgpu_as_bind_channel = _ct([("channel_fd", ctypes.c_uint32)])
_nvgpu_as_alloc_space = _ct([("pages", ctypes.c_uint64), ("page_size", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                              ("offset", ctypes.c_uint64), ("padding", ctypes.c_uint32 * 2)])
_nvgpu_as_map_buffer_ex = _ct([("flags", ctypes.c_uint32), ("compr_kind", ctypes.c_int16), ("incompr_kind", ctypes.c_int16),
                                ("dmabuf_fd", ctypes.c_uint32), ("page_size", ctypes.c_uint32), ("buffer_offset", ctypes.c_uint64),
                                ("mapping_size", ctypes.c_uint64), ("offset", ctypes.c_uint64)])
_nvgpu_as_unmap_buffer = _ct([("offset", ctypes.c_uint64)])
_nvgpu_open_tsg = _ct([("tsg_fd", ctypes.c_int32), ("flags", ctypes.c_uint32), ("token", ctypes.c_uint32),
                        ("reserved", ctypes.c_uint32), ("subctx_id", ctypes.c_uint32), ("_pad", ctypes.c_uint32)])
_nvgpu_tsg_bind_channel_ex = _ct([("channel_fd", ctypes.c_int32), ("subcontext_id", ctypes.c_uint32),
                                   ("reserved", ctypes.c_uint8 * 16)])
_nvgpu_tsg_create_subctx = _ct([("type", ctypes.c_uint32), ("as_fd", ctypes.c_int32),
                                 ("veid", ctypes.c_uint32), ("reserved", ctypes.c_uint32)])
_nvgpu_open_channel = _ct([("channel_fd", ctypes.c_int32)])
_nvgpu_alloc_obj_ctx = _ct([("class_num", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("obj_id", ctypes.c_uint64)])
_nvgpu_setup_bind = _ct([("num_gpfifo_entries", ctypes.c_uint32), ("num_inflight_jobs", ctypes.c_uint32),
                          ("flags", ctypes.c_uint32), ("userd_dmabuf_fd", ctypes.c_int32), ("gpfifo_dmabuf_fd", ctypes.c_int32),
                          ("work_submit_token", ctypes.c_uint32), ("userd_dmabuf_offset", ctypes.c_uint64),
                          ("gpfifo_dmabuf_offset", ctypes.c_uint64), ("gpfifo_gpu_va", ctypes.c_uint64),
                          ("userd_gpu_va", ctypes.c_uint64), ("usermode_mmio_gpu_va", ctypes.c_uint64),
                          ("reserved", ctypes.c_uint32 * 9)])
_nvgpu_channel_wdt = _ct([("wdt_status", ctypes.c_uint32), ("timeout_ms", ctypes.c_uint32)])

def _ioc(d, t, nr, size): return (d << 30) | (size << 16) | (ord(t) << 8) | nr
def _io(t, nr): return _ioc(0, t, nr, 0)
def _iow(t, nr, sz): return _ioc(1, t, nr, sz)
def _iowr(t, nr, sz): return _ioc(3, t, nr, sz)

_NVGPU_GET_CHARS = _iowr('G', 5, ctypes.sizeof(_nvgpu_gpu_get_characteristics))
_NVGPU_ALLOC_AS = _iowr('G', 8, ctypes.sizeof(_nvgpu_alloc_as))
_NVGPU_OPEN_TSG = _iowr('G', 9, ctypes.sizeof(_nvgpu_open_tsg))
_NVGPU_OPEN_CH = _iowr('G', 11, ctypes.sizeof(_nvgpu_open_channel))
_NVMAP_CREATE = _iowr('N', 0, ctypes.sizeof(_nvmap_handle))
_NVMAP_ALLOC = _iow('N', 3, ctypes.sizeof(_nvmap_alloc))
_NVMAP_GET_FD = _iowr('N', 15, ctypes.sizeof(_nvmap_handle))
_NVMAP_FREE = _io('N', 4)
_NVGPU_AS_BIND_CH = _iowr('A', 1, ctypes.sizeof(_nvgpu_as_bind_channel))
_NVGPU_AS_ALLOC_SPACE = _iowr('A', 6, ctypes.sizeof(_nvgpu_as_alloc_space))
_NVGPU_AS_MAP_BUF = _iowr('A', 7, ctypes.sizeof(_nvgpu_as_map_buffer_ex))
_NVGPU_AS_UNMAP_BUF = _iowr('A', 5, ctypes.sizeof(_nvgpu_as_unmap_buffer))
_NVGPU_TSG_BIND_CH = _iowr('T', 11, ctypes.sizeof(_nvgpu_tsg_bind_channel_ex))
_NVGPU_TSG_SUBCTX = _iowr('T', 18, ctypes.sizeof(_nvgpu_tsg_create_subctx))
_NVGPU_CH_ALLOC_OBJ = _iowr('H', 108, ctypes.sizeof(_nvgpu_alloc_obj_ctx))
_NVGPU_CH_SETUP_BIND = _iowr('H', 128, ctypes.sizeof(_nvgpu_setup_bind))
_NVGPU_CH_WDT = _iow('H', 119, ctypes.sizeof(_nvgpu_channel_wdt))

_NVMAP_HEAP_IOVMM, _NVMAP_WC, _NVMAP_CACHED, _NVMAP_TAG = (1 << 30), 1, 2, 0x0900

def _tioctl(fd, nr, buf): fcntl.ioctl(fd, nr, buf)

def _nvmap_buf(nvmap_fd, size, cache_flags, align=4096):
  c = _nvmap_handle(size=size)
  _tioctl(nvmap_fd, _NVMAP_CREATE, c)
  a = _nvmap_alloc(handle=c.handle, heap_mask=_NVMAP_HEAP_IOVMM, flags=(_NVMAP_TAG << 16) | cache_flags, align=align)
  _tioctl(nvmap_fd, _NVMAP_ALLOC, a)
  g = _nvmap_handle(handle=c.handle)
  _tioctl(nvmap_fd, _NVMAP_GET_FD, g)
  return c.handle, g.size  # g.size is the dmabuf fd

class TegraMem:
  def __init__(self, handle, dmabuf_fd, gpu_va, size, cpu_addr=0, hMemory=0):
    self.handle, self.dmabuf_fd, self.gpu_va, self.size, self.cpu_addr, self.hMemory = handle, dmabuf_fd, gpu_va, size, cpu_addr, hMemory

class TegraIface:
  _nvmap_fd, _ctrl_fd, _chars = -1, -1, None

  def __init__(self, dev, device_id):
    if device_id != 0: raise RuntimeError("TegraIface only supports device 0 (single iGPU)")
    if TegraIface._chars is None:
      if not os.path.exists("/dev/nvgpu/igpu0/ctrl"): raise FileNotFoundError("/dev/nvgpu/igpu0/ctrl")
      TegraIface._nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
      TegraIface._ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
      chars = _nvgpu_gpu_characteristics()
      req = _nvgpu_gpu_get_characteristics(buf_size=ctypes.sizeof(chars), buf_addr=ctypes.addressof(chars))
      _tioctl(TegraIface._ctrl_fd, _NVGPU_GET_CHARS, req)
      TegraIface._chars = chars

    self.dev, self.device_id = dev, device_id
    chars = TegraIface._chars
    self.compute_class, self.gpfifo_class, self.dma_class = chars.compute_class, chars.gpfifo_class, chars.dma_copy_class
    self.viddec_class = None
    self._hcnt = 0x10000
    self._as_fd, self._tsg_fd, self._subctx_veid = -1, -1, 0
    self._ch_fds: dict[int, int] = {}
    self._ws_tokens: dict[int, int] = {}
    self.root, self.gpu_instance = self._nh(), 0
    self._allocs: list[TegraMem] = []

  def _nh(self) -> int:
    self._hcnt += 1
    return self._hcnt

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    handle = self._nh()

    if clss in (nv_gpu.NV01_DEVICE_0, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.FERMI_VASPACE_A, nv_gpu.NV01_ROOT_CLIENT, nv_gpu.NV01_ROOT,
                nv_gpu.NV1_MEMORY_SYSTEM, nv_gpu.NV1_MEMORY_USER, nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR):
      return handle

    if clss == nv_gpu.NV01_MEMORY_VIRTUAL:
      args = _nvgpu_alloc_as(flags=2, va_range_start=(1 << 21), va_range_end=(1 << 40) - (1 << 21))
      _tioctl(self._ctrl_fd, _NVGPU_ALLOC_AS, args)
      self._as_fd = args.as_fd
      for wva in [0xFD00000000, 0xFE00000000]:
        rsv = _nvgpu_as_alloc_space(pages=0x40000000 // mmap.PAGESIZE, page_size=mmap.PAGESIZE, flags=1, offset=wva)
        _tioctl(self._as_fd, _NVGPU_AS_ALLOC_SPACE, rsv)
      return handle

    if clss == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      tsg = _nvgpu_open_tsg()
      _tioctl(self._ctrl_fd, _NVGPU_OPEN_TSG, tsg)
      self._tsg_fd = tsg.tsg_fd
      return handle

    if clss == nv_gpu.FERMI_CONTEXT_SHARE_A:
      subctx = _nvgpu_tsg_create_subctx(type=1, as_fd=self._as_fd)
      _tioctl(self._tsg_fd, _NVGPU_TSG_SUBCTX, subctx)
      self._subctx_veid = subctx.veid
      return handle

    if clss in (self.gpfifo_class, nv_gpu.AMPERE_CHANNEL_GPFIFO_A):
      ch = _nvgpu_open_channel(channel_fd=-1)
      _tioctl(self._ctrl_fd, _NVGPU_OPEN_CH, ch)
      ch_fd = ch.channel_fd
      _tioctl(self._as_fd, _NVGPU_AS_BIND_CH, _nvgpu_as_bind_channel(channel_fd=ch_fd))
      _tioctl(self._tsg_fd, _NVGPU_TSG_BIND_CH, _nvgpu_tsg_bind_channel_ex(channel_fd=ch_fd, subcontext_id=self._subctx_veid))
      _tioctl(ch_fd, _NVGPU_CH_WDT, _nvgpu_channel_wdt(wdt_status=1))

      gpfifo_entries, gpfifo_buf_handle, gpfifo_va, userd_off = 0x10000, 0, 0, 0
      if params is not None:
        gpfifo_va, gpfifo_entries, gpfifo_buf_handle = params.gpFifoOffset, params.gpFifoEntries, params.hObjectBuffer
        userd_off = params.userdOffset[0]

      gpfifo_area_mem = next((m for m in self._allocs if m.hMemory == gpfifo_buf_handle), None)
      if gpfifo_area_mem is None: raise RuntimeError(f"TegraIface: gpfifo_area alloc not found for handle {gpfifo_buf_handle}")
      ring_sz = gpfifo_entries * 8

      _, ring_fd = _nvmap_buf(self._nvmap_fd, ring_sz, _NVMAP_WC)
      _, userd_fd = _nvmap_buf(self._nvmap_fd, 4096, _NVMAP_WC)

      setup = _nvgpu_setup_bind(num_gpfifo_entries=gpfifo_entries, gpfifo_dmabuf_fd=ring_fd, userd_dmabuf_fd=userd_fd,
                                flags=(1 << 3) | (1 << 1))
      _tioctl(ch_fd, _NVGPU_CH_SETUP_BIND, setup)

      cpu_base = gpfifo_area_mem.cpu_addr
      FileIOInterface._mmap(cpu_base + (gpfifo_va - gpfifo_area_mem.gpu_va), ring_sz,
                            mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, ring_fd, 0)
      FileIOInterface._mmap(cpu_base + userd_off, 4096,
                            mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, userd_fd, 0)

      self._ch_fds[handle], self._ws_tokens[handle] = ch_fd, setup.work_submit_token
      return handle

    if clss in (self.compute_class, nv_gpu.AMPERE_COMPUTE_B, self.dma_class, nv_gpu.AMPERE_DMA_COPY_B):
      ch_fd = self._ch_fds.get(parent, -1)
      if ch_fd == -1: raise RuntimeError(f"TegraIface: no channel fd for handle {parent}")
      obj = _nvgpu_alloc_obj_ctx(class_num=self.compute_class if clss in (self.compute_class, nv_gpu.AMPERE_COMPUTE_B) else self.dma_class)
      _tioctl(ch_fd, _NVGPU_CH_ALLOC_OBJ, obj)
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
      chars = TegraIface._chars
      info_map = {
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS: chars.num_gpc,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC: chars.num_tpc_per_gpc,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC: 2,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM: chars.sm_arch_warp_count,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_SM_VERSION: chars.sm_arch_sm_version,
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

  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def alloc(self, size: int, host=False, uncached=False, cpu_access=False, **kwargs) -> HCQBuffer:
    alloc_align = mmap.PAGESIZE if (uncached or host) else ((2 << 20) if size >= (8 << 20) else mmap.PAGESIZE)
    size = round_up(size, alloc_align)
    cache = _NVMAP_WC if (uncached or host) else _NVMAP_CACHED

    handle, dmabuf_fd = _nvmap_buf(self._nvmap_fd, size, cache, alloc_align)

    gpu_va = 0
    if self._as_fd >= 0:
      m = _nvgpu_as_map_buffer_ex(compr_kind=-1, dmabuf_fd=dmabuf_fd, page_size=mmap.PAGESIZE)
      _tioctl(self._as_fd, _NVGPU_AS_MAP_BUF, m)
      gpu_va = m.offset

    addr = FileIOInterface._mmap(gpu_va or 0, size, mmap.PROT_READ | mmap.PROT_WRITE,
                                 mmap.MAP_SHARED | (MAP_FIXED if gpu_va else 0), dmabuf_fd, 0)
    if gpu_va and addr != gpu_va:
      addr = FileIOInterface._mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, dmabuf_fd, 0)

    meta = TegraMem(handle=handle, dmabuf_fd=dmabuf_fd, gpu_va=gpu_va, size=size, cpu_addr=addr, hMemory=self._nh())
    self._allocs.append(meta)
    return HCQBuffer(va_addr=gpu_va, size=size, meta=meta, view=MMIOInterface(addr, size, fmt='B'), owner=self.dev)

  def free(self, mem: HCQBuffer):
    meta = mem.meta
    if meta.gpu_va and self._as_fd >= 0:
      with contextlib.suppress(OSError): _tioctl(self._as_fd, _NVGPU_AS_UNMAP_BUF, _nvgpu_as_unmap_buffer(offset=meta.gpu_va))
    if meta.cpu_addr:
      with contextlib.suppress(Exception): FileIOInterface.munmap(meta.cpu_addr, meta.size)
    if meta.dmabuf_fd >= 0:
      with contextlib.suppress(OSError): os.close(meta.dmabuf_fd)
    if meta.handle:
      h = meta.handle if meta.handle < 0x80000000 else meta.handle - 0x100000000
      with contextlib.suppress(OSError): fcntl.ioctl(self._nvmap_fd, _NVMAP_FREE, h)
    if meta in self._allocs: self._allocs.remove(meta)

  def map(self, mem: HCQBuffer): pass
  def sleep(self, tm: int): pass
