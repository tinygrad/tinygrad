from __future__ import annotations
import os, ctypes, mmap, struct, array, functools, sys
assert sys.platform != 'win32'
from dataclasses import dataclass
from tinygrad.device import BufferSpec, Compiled, LRUAllocator
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.autogen import msm_drm, mesa
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import to_mv, round_up, data64_le, flatten, prod
from tinygrad.dtype import ImageDType
from tinygrad.runtime.support.adreno import pkt7_hdr, pkt4_hdr, parse_ir3_shader, compute_program_sizes
from tinygrad.runtime.support.adreno import build_a6xx_compute_pm4, build_a6xx_tex_descriptor

@dataclass(frozen=True)
class MSMBuffer:
  va_addr: int
  size: int
  handle: int
  offset: int  # mmap offset

class MSMAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> MSMBuffer:
    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.dev.fd, size=round_up(size, 0x1000), flags=msm_drm.MSM_BO_WC)
    iova_info = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.dev.fd, handle=gem.handle, info=msm_drm.MSM_INFO_GET_IOVA)
    off_info = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.dev.fd, handle=gem.handle, info=msm_drm.MSM_INFO_GET_OFFSET)
    va = self.dev.fd.mmap(0, round_up(size, 0x1000), mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, off_info.value)
    return MSMBuffer(va_addr=iova_info.value, size=size, handle=gem.handle, offset=va)

  def _free(self, opaque:MSMBuffer, options:BufferSpec):
    self.dev.synchronize()
    FileIOInterface.munmap(opaque.offset, round_up(opaque.size, 0x1000))
    msm_drm.DRM_IOCTL_GEM_CLOSE(self.dev.fd, handle=opaque.handle)

  def _copyin(self, dest:MSMBuffer, src:memoryview): ctypes.memmove(dest.offset, (ctypes.c_char * len(src)).from_buffer(src), len(src))
  def _copyout(self, dest:memoryview, src:MSMBuffer):
    self.dev.synchronize()
    ctypes.memmove((ctypes.c_char * len(dest)).from_buffer(dest), src.offset, len(dest))
  def _as_buffer(self, src:MSMBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.offset, src.size)

def _build_pm4(prg:MSMProgram, args_va:int, global_size, local_size) -> list[int]:
  q: list[int] = []
  def cmd(opcode, *vals): q.extend([pkt7_hdr(opcode, len(vals)), *vals])
  def reg(register, *vals): q.extend([pkt4_hdr(register, len(vals)), *vals])
  # pre-dispatch: invalidate cache
  cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE)
  cmd(mesa.CP_WAIT_FOR_IDLE)
  # dispatch
  build_a6xx_compute_pm4(cmd, reg, prg, args_va, prg.lib_buf.va_addr, prg.dev._stack.va_addr,
                         prg.dev.border_color_buf.va_addr, global_size, local_size)
  # post-dispatch: flush cache
  cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(prg.dev.dummy_buf.va_addr), 0)
  cmd(mesa.CP_WAIT_FOR_IDLE)
  return q

class MSMProgram:
  def __init__(self, dev:MSMDevice, name:str, lib:bytes, buf_dtypes=[], **kwargs):
    self.dev, self.name, self.buf_dtypes, self.NIR = dev, name, buf_dtypes, True
    parse_ir3_shader(self, lib)
    compute_program_sizes(self)

    self.lib_buf: MSMBuffer = dev.allocator.alloc(self.image_size, BufferSpec())
    ctypes.memmove(self.lib_buf.offset, self.image, self.image_size)
    dev._ensure_stack_size(self.hw_stack_offset * 4)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait=False, **kw):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")

    # fill args buffer
    args_buf: MSMBuffer = self.dev.allocator.alloc(self.kernargs_alloc_size, BufferSpec())
    ctypes.memset(args_buf.offset, 0, self.kernargs_alloc_size)
    args_mv = to_mv(args_buf.offset, self.kernargs_alloc_size)

    ubos = [b for i,b in enumerate(bufs) for _,dt in self.buf_dtypes[i] if not isinstance(dt, ImageDType)]
    uavs = [(dt,b) for i,b in enumerate(bufs) for _,dt in self.buf_dtypes[i] if isinstance(dt, ImageDType)]
    ibos, texs = uavs[:self.ibo_cnt], [uavs[self.ibo_cnt + self.tex_to_image[i]] for i in range(self.tex_cnt)]

    for cnst_val, cnst_off, cnst_sz in self.consts_info:
      args_mv[cnst_off:cnst_off+cnst_sz] = cnst_val.to_bytes(cnst_sz, byteorder='little')
    if self.samp_cnt > 0:
      to_mv(args_buf.offset + self.samp_off, len(self.samplers) * 4).cast('I')[:] = array.array('I', self.samplers)

    # write UBO addresses and vals
    struct.pack_into(f'{len(ubos)}Q', args_mv, self.buf_off, *[b.va_addr for b in ubos])
    struct.pack_into(f'{len(vals)}I', args_mv, self.buf_off + len(ubos) * 8, *vals)

    # write texture/IBO descriptors
    tex_data = flatten(build_a6xx_tex_descriptor(dt, b.va_addr) for dt,b in texs)
    ibo_data = flatten(build_a6xx_tex_descriptor(dt, b.va_addr, ibo=True) for dt,b in ibos)
    if tex_data: struct.pack_into(f'{len(tex_data)}I', args_mv, self.tex_off, *tex_data)
    if ibo_data: struct.pack_into(f'{len(ibo_data)}I', args_mv, self.ibo_off, *ibo_data)

    # build PM4 command stream
    pm4 = _build_pm4(self, args_buf.va_addr, global_size, local_size)

    # copy PM4 to GPU-visible buffer
    cmd_buf: MSMBuffer = self.dev.allocator.alloc(len(pm4) * 4, BufferSpec())
    to_mv(cmd_buf.offset, len(pm4) * 4).cast('I')[:] = array.array('I', pm4)

    # collect all BO handles for submit
    bo_handles = {cmd_buf.handle, args_buf.handle, self.lib_buf.handle, self.dev._stack.handle,
                  self.dev.border_color_buf.handle, self.dev.dummy_buf.handle}
    for b in bufs:
      if hasattr(b, 'handle'): bo_handles.add(b.handle)

    bo_list = list(bo_handles)
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * len(bo_list))(*[
      msm_drm.struct_drm_msm_gem_submit_bo(flags=msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE, handle=h) for h in bo_list])

    cmd_idx = bo_list.index(cmd_buf.handle)
    cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)(
      msm_drm.struct_drm_msm_gem_submit_cmd(type=msm_drm.MSM_SUBMIT_CMD_BUF, submit_idx=cmd_idx, size=len(pm4) * 4))

    submit = msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT(self.dev.fd, flags=msm_drm.MSM_PIPE_3D0, nr_bos=len(bo_list), nr_cmds=1,
                                                bos=ctypes.addressof(bos), cmds=ctypes.addressof(cmds), queueid=self.dev.queue_id)
    self.dev.last_fence = submit.fence

    if wait: self.dev.synchronize()

class MSMDevice(Compiled):
  def __init__(self, device:str=""):
    self.fd = FileIOInterface(_find_msm_device(), os.O_RDWR)

    # query GPU info
    info = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_CHIP_ID)
    chip_id = info.value
    self.gpu_id = (chip_id >> 24, (chip_id >> 16) & 0xFF, (chip_id >> 8) & 0xFF)
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={chip_id:#x}")

    # create submit queue (priority 1 = medium)
    sq = msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW(self.fd, flags=0, prio=1)
    self.queue_id = sq.id
    self.last_fence: int = 0

    self.allocator = MSMAllocator(self)

    # allocate internal buffers
    self.border_color_buf: MSMBuffer = self.allocator.alloc(0x1000, BufferSpec())
    ctypes.memset(self.border_color_buf.offset, 0, 0x1000)
    self.dummy_buf: MSMBuffer = self.allocator.alloc(0x1000, BufferSpec())

    super().__init__(device, self.allocator, [IR3Renderer], functools.partial(MSMProgram, self), arch="a%d%d%d" % self.gpu_id)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self.allocator.alloc(sz, BufferSpec())
    elif self._stack.size < sz:
      self.synchronize()
      self._stack = self.allocator.alloc(sz, BufferSpec())

  def synchronize(self):
    if self.last_fence == 0: return
    import time
    timeout = msm_drm.struct_drm_msm_timespec(tv_sec=int(time.time()) + 10, tv_nsec=0)
    msm_drm.DRM_IOCTL_MSM_WAIT_FENCE(self.fd, fence=self.last_fence, flags=0, timeout=timeout, queueid=self.queue_id)

def _find_msm_device() -> str:
  for i in range(128, 144):
    path = f"/dev/dri/renderD{i}"
    if not os.path.exists(path): continue
    try:
      fd = os.open(path, os.O_RDWR)
      try:
        version = bytearray(256)
        # check driver name via DRM version ioctl
        ver = msm_drm.DRM_IOCTL_VERSION(fd, name=ctypes.addressof((ctypes.c_char * 256)(*version)),
                                          name_len=256, date_len=0, desc_len=0)
        name = ctypes.string_at(ver.name, ver.name_len).decode()
        if name == "msm": return path
      finally: os.close(fd)
    except OSError: continue
  raise RuntimeError("No MSM DRM device found")
