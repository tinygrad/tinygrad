import os, time, math, ctypes, fcntl, functools, mmap, struct, array, decimal
from types import SimpleNamespace
from typing import Tuple, List, Any, cast
from tinygrad.device import BufferOptions, HCQBuffer, HWComputeQueue, HCQProgram, HCQCompiled, HCQSignal, HCQAllocator, HCQArgsState, hcq_command
import tinygrad.runtime.autogen.kgsl as kgsl
import tinygrad.runtime.autogen.adreno as adreno
import tinygrad.runtime.autogen.libc as libc
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice
from tinygrad.renderer.cstyle import QCOMRenderer
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, from_mv, mv_address, to_mv, round_up, data64_le
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

def next_power2(x): return 1 if x == 0 else 1 << (x - 1).bit_length()

def prt(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return adreno.CP_TYPE7_PKT | cnt & 0x3FFF | prt(cnt) << 15 | (opcode & 0x7F) << 16 | prt(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return adreno.CP_TYPE4_PKT | cnt & 0x7F | prt(cnt) << 7 | (reg & 0x3FFFF) << 8 | prt(reg) << 27

def parse_cl_lib(lib: bytes, name:str):
  """
  Extract information from OpenCL binary used to run a shader: image offset, image size, prg_offset, offsets to argument buffers,
  constants offsets and values, HALFREGFOOTPRINT and FULLREGFOOTPRINT
  """
  image_offset, image_size, buffs_info, consts_info = struct.unpack("I", lib[0xC0:0xC4])[0], struct.unpack("I", lib[0x100:0x104])[0], [], []

  ptr = struct.unpack("I", lib[0x110:0x114])[0] # read img desc offset
  prg_offset = struct.unpack("I", lib[ptr+196:ptr+200])[0]
  branch_stack = struct.unpack("I", lib[ptr+264:ptr+268])[0]
  pvtmem = struct.unpack("I", lib[ptr+0xc8:ptr+0xcc])[0]
  shmem = struct.unpack("I", lib[ptr+0xd8:ptr+0xdc])[0]
  ptr = round_up(ptr + 344 + len(name), 4) + 8 * struct.unpack("I", lib[ptr+220:ptr+224])[0] # skip to bufs descr, align name, skip samplers

  while (ptr + 16 <= len(lib)):
    length, _, _, offset_words = struct.unpack("I" * 4, lib[ptr:ptr+16])
    if length == 0: break
    buffs_info.append((offset_words * 4, struct.unpack("I", lib[ptr+0x3c:ptr+0x40])[0] == 0x0))
    ptr += length

  ptr = struct.unpack("I", lib[0xb0:0xb4])[0]
  if ptr != 0:
    ptr = struct.unpack("I", lib[0xac:0xb0])[0] # read consts desc offset
    # constant vals are placed just before a shader
    while (ptr + 40 <= image_offset):
      cnst, offset_words, _, is32 = struct.unpack("I", lib[ptr:ptr+4])[0], *struct.unpack("III", lib[ptr+16:ptr+28])
      ptr += 40
      consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))

  ptr = struct.unpack("I", lib[0x34:0x38])[0] # read main offset
  fullreg, hlfreg = struct.unpack("II", lib[ptr+20:ptr+28])

  return image_offset, image_size, prg_offset, buffs_info, consts_info, hlfreg, fullreg, branch_stack, shmem, pvtmem

class QcomCompiler(CLCompiler):
  def __init__(self, device:str=""): super().__init__(CLDevice(device), 'compile_qcom')

class QcomSignal(HCQSignal):
  def __init__(self, value=0, **kwargs):
    self._signal = QcomDevice.signals_pool.pop()
    super().__init__(value)
  def __del__(self): QcomDevice.signals_pool.append(self._signal)
  def _get_value(self) -> int: return self._signal[0]
  def _get_timestamp(self) -> decimal.Decimal: return decimal.Decimal(self._signal[1])
  def _set_value(self, new_value:int): self._signal[0] = new_value
  def wait(self, value:int, timeout:int=60000):
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if self._signal[0] >= value: return
    raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

MAP_FIXED = 0x10
class QcomDevice(HCQCompiled):
  signals_page: Any = None
  signals_pool: List[Any] = []
  gpu_id: int = 0

  def __init__(self, device:str=""):
    self.fd = os.open('/dev/kgsl-3d0', os.O_RDWR)
    QcomDevice.signals_page = self._gpu_alloc(16 * 65536, map_to_cpu=True, uncached=True)
    QcomDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, self.signals_page.size, 16)]
    info, self.ctx, self.cmd_buf, self.cmd_buf_ptr = self._info(), self._ctx_create(), self._gpu_alloc(0x1000000, map_to_cpu=True), 0
    QcomDevice.gpu_id = ((info.chip_id >> 24) & 0xFF) * 100 + ((info.chip_id >> 16) & 0xFF) * 10 + ((info.chip_id >>  8) & 0xFF)
    assert QcomDevice.gpu_id < 700

    super().__init__(device, QcomAllocator(self), QCOMRenderer(), QcomCompiler(device), functools.partial(QcomProgram, self),
                     QcomSignal, QcomComputeQueue, None, timeline_signals=(QcomSignal(), QcomSignal()))

    QcomComputeQueue().setup().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

  def _ctx_create(self):
    cr = kgsl.struct_kgsl_drawctxt_create(flags=(kgsl.KGSL_CONTEXT_TYPE(kgsl.KGSL_CONTEXT_TYPE_CL) | kgsl.KGSL_CONTEXT_PREAMBLE
                                                 | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE
                                                 | kgsl.KGSL_CONTEXT_PREEMPT_STYLE(kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
                                                 | kgsl.KGSL_CONTEXT_PRIORITY(8)))
    self._ioctl(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, cr)
    self.context_id = cr.drawctxt_id
    return self.context_id

  def _ctx_destroy(self, ctx_id):
    dstr = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=ctx_id)
    self._ioctl(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, dstr)

  def _info(self):
    info = kgsl.struct_kgsl_devinfo()
    get_property = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self._ioctl(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY, get_property)
    return info

  def _ioctl(self, nr, arg):
    ret = fcntl.ioctl(self.fd, (3 << 30) | (ctypes.sizeof(arg) & 0x1FFF) << 16 | 0x9 << 8 | (nr & 0xFF), arg)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return ret

  def _gpu_alloc(self, size:int, flags:int=0, map_to_cpu=False, uncached=False, fill_zeroes=False):
    size = round_up(size, 1 << (alignment_hint:=12))
    flags |= (kgsl.KGSL_MEMALIGN(alignment_hint))
    if uncached: flags |= (kgsl.KGSL_CACHEMODE(kgsl.KGSL_CACHEMODE_UNCACHED))

    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=size, flags=flags)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    va_addr, va_len = None, 0
    if not (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP):
      info = kgsl.struct_kgsl_gpuobj_info(id=alloc.id)
      self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_INFO, info)
      va_addr, va_len = info.gpuaddr, info.va_len

    if map_to_cpu or (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP):
      va_addr = libc.mmap(va_addr, va_len := (va_len or alloc.mmapsize), mmap.PROT_READ|mmap.PROT_WRITE,
                          mmap.MAP_SHARED | (MAP_FIXED if va_addr is not None else 0), self.fd, alloc.id * 0x1000)
      if fill_zeroes: ctypes.memset(va_addr, 0, va_len)

    return SimpleNamespace(va_addr=va_addr, size=va_len, mapped=map_to_cpu or (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP), info=alloc)

  def _gpu_free(self, mem):
    free = kgsl.struct_kgsl_gpuobj_free(id=mem.info.id)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free)
    if mem.mapped: libc.munmap(mem.va_addr, mem.size)

  def _alloc_cmd_buf(self, sz: int):
    self.cmd_buf_ptr = (cur_ptr:=self.cmd_buf_ptr if self.cmd_buf_ptr + sz < self.cmd_buf.size else 0) + sz
    return self.cmd_buf.va_addr + cur_ptr

  def _border_color_base(self):
    if not hasattr(self, '_border_color_gpu'): self._border_color_gpu = self._gpu_alloc(0x1000000, map_to_cpu=True, fill_zeroes=True)
    return self._border_color_gpu.va_addr

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)

class QcomAllocator(HCQAllocator):
  def __init__(self, device:QcomDevice): super().__init__(device)

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer:
    if options.image is not None:
      pitchalign = 6
      pitch = round_up(round_up(options.image.shape[1], 16) * (4 * options.image.base.itemsize), 1 << pitchalign)
      texture = self.device._gpu_alloc(pitch * round_up(options.image.shape[0], 16), kgsl.KGSL_MEMTYPE_TEXTURE, map_to_cpu=True)

      # save it here to load in one command (the same approach as OpenCL and mesa)
      texture.samplers, texture.descriptor, texture.ibo = [0] * 4, [0] * 16, [0] * 16
      texture.samplers[0:2] = [0x1b60, 0x30] # compiled sampler. always the same in tinygrad.

      texture.descriptor[0] = adreno.A6XX_TEX_CONST_0_SWIZ_X(0) | adreno.A6XX_TEX_CONST_0_SWIZ_Y(1) | adreno.A6XX_TEX_CONST_0_SWIZ_Z(2) \
        | adreno.A6XX_TEX_CONST_0_SWIZ_W(3) | adreno.A6XX_TEX_CONST_0_FMT(0x82 if options.image.base == dtypes.float32 else 0x62)
      texture.descriptor[1] = adreno.A6XX_TEX_CONST_1_WIDTH(options.image.shape[1]) | adreno.A6XX_TEX_CONST_1_HEIGHT(options.image.shape[0])
      texture.descriptor[2] = adreno.A6XX_TEX_CONST_2_TYPE(adreno.A6XX_TEX_2D) | adreno.A6XX_TEX_CONST_2_PITCH(pitch) \
        | adreno.A6XX_TEX_CONST_2_PITCHALIGN(pitchalign - 6)

      texture.descriptor[4:6] = data64_le(texture.va_addr)
      texture.descriptor[6] = 0x40000000
      texture.descriptor[7] = 0xe

      texture.ibo = [texture.descriptor[0] & (~0xffff), *texture.descriptor[1:len(texture.descriptor)]]

      return texture

    return self.device._gpu_alloc(size, map_to_cpu=True)

  def copyin(self, dest:HCQBuffer, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)

  def copyout(self, dest:memoryview, src:HCQBuffer):
    self.device.synchronize()
    ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)

  def _free(self, opaque, options:BufferOptions):
    self.device.synchronize()
    self.device._gpu_free(opaque)

class QcomComputeQueue(HWComputeQueue):
  def __init__(self):
    self.cmd_idx_to_dims = {}
    super().__init__()

  def cmd(self, opcode: int, *vals: int): self.q += [pkt7_hdr(opcode, len(vals)), *vals]

  def reg(self, reg: int, *vals: int): self.q += [pkt4_hdr(reg, len(vals)), *vals]

  def _signal(self, signal, value=0, ts=False):
    if QcomDevice.gpu_id < 700:
      self.cmd(adreno.CP_EVENT_WRITE, adreno.CACHE_FLUSH_TS | (0 if not ts else adreno.CP_EVENT_WRITE_0_TIMESTAMP),
               *data64_le(mv_address(signal._signal) + (0 if not ts else 8)), value & 0xFFFFFFFF)
      self.cmd(adreno.CP_EVENT_WRITE, adreno.CACHE_INVALIDATE)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')

  def _timestamp(self, signal): return self._signal(signal, 0, ts=True)

  def _wait(self, signal, value=0):
    self.cmd(adreno.CP_WAIT_REG_MEM, adreno.WRITE_GE | adreno.CP_WAIT_REG_MEM_0_POLL(adreno.POLL_MEMORY),
             *data64_le(mv_address(signal._signal)), value & 0xFFFFFFFF, 0xFFFFFFFF, 32) # busy wait for 32 cycles

  def _update_signal(self, cmd_idx, signal, value):
    if signal is not None: self._patch(cmd_idx, offset=2, data=data64_le(mv_address(signal._signal)))
    if value is not None: self._patch(cmd_idx, offset=4, data=[value & 0xFFFFFFFF])

  def _update_wait(self, cmd_idx, signal, value):
    if signal is not None: self._patch(cmd_idx, offset=2, data=data64_le(mv_address(signal._signal)))
    if value is not None: self._patch(cmd_idx, offset=4, data=[value & 0xFFFFFFFF])

  def bind(self, device):
    self.binded_device = device

    cmdbytes, hw_page_addr = array.array('I', self.q), device._alloc_cmd_buf(len(self.q) * 4)
    ctypes.memmove(hw_page_addr, mv_address(memoryview(cmdbytes)), len(cmdbytes) * 4)

    self.obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(cmdbytes) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    self.submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(self.obj), numcmds=1, context_id=device.ctx,
                                                   cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    # From now on, the queue is on the device for faster submission.
    self.q = to_mv(hw_page_addr, len(self.q) * 4).cast("I") # type: ignore

  def _submit(self, device):
    if self.binded_device == device:
      device._ioctl(kgsl.IOCTL_KGSL_GPU_COMMAND, self.submit_req)
      return

    cmdbytes, hw_page_addr = array.array('I', self.q), device._alloc_cmd_buf(len(self.q) * 4)
    ctypes.memmove(hw_page_addr, mv_address(memoryview(cmdbytes)), len(cmdbytes) * 4)

    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(cmdbytes) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=device.ctx,
                                              cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    device._ioctl(kgsl.IOCTL_KGSL_GPU_COMMAND, submit_req)

  @hcq_command
  def setup(self):
    self.cmd(adreno.CP_WAIT_FOR_IDLE)
    self.cmd(adreno.CP_SET_MARKER, adreno.RM6_COMPUTE)
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, adreno.A6XX_HLSQ_INVALIDATE_CMD_CS_STATE | adreno.A6XX_HLSQ_INVALIDATE_CMD_CS_IBO)
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, 0x0)
    self.reg(adreno.REG_A6XX_SP_CS_TEX_COUNT, 0xff) # set to max
    self.reg(adreno.REG_A6XX_SP_CS_IBO_COUNT, 0xff) # set to max
    self.reg(adreno.REG_A6XX_SP_MODE_CONTROL, adreno.A6XX_SP_MODE_CONTROL_ISAMMODE(adreno.ISAMMODE_CL))
    self.reg(adreno.REG_A6XX_SP_PERFCTR_ENABLE, adreno.A6XX_SP_PERFCTR_ENABLE_CS)
    self.reg(adreno.REG_A6XX_SP_TP_MODE_CNTL, adreno.ISAMMODE_CL | (1 << 3)) # ISAMMODE|UNK3
    self.reg(adreno.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)

  def _exec(self, prg, args_state, global_size, local_size):
    global_size_mp = cast(Tuple[int,int,int], tuple(int(g*l) for g,l in zip(global_size, local_size))) if local_size else global_size
    self.cmd_idx_to_dims[len(self) - 1] = [global_size, local_size]

    self.cmd(adreno.CP_WAIT_FOR_IDLE)
    self.reg(adreno.REG_A6XX_HLSQ_CS_NDRANGE_0,
             adreno.A6XX_HLSQ_CS_NDRANGE_0_KERNELDIM(3) | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(local_size[0] - 1)
             | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(local_size[1] - 1) | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(local_size[2] - 1),
             global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf,
             0xfc | adreno.A6XX_HLSQ_CS_CNTL_1_THREADSIZE(adreno.THREAD128), global_size[0], global_size[1], global_size[2])
    self.reg(adreno.REG_A6XX_SP_CS_CTRL_REG0,
             adreno.A6XX_SP_CS_CTRL_REG0_THREADSIZE(adreno.THREAD128) | adreno.A6XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT(prg.hlfreg)
             | adreno.A6XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT(prg.fullreg) | adreno.A6XX_SP_CS_CTRL_REG0_BRANCHSTACK(prg.branch_stack),
             adreno.A6XX_SP_CS_UNKNOWN_A9B1_UNK5 | adreno.A6XX_SP_CS_UNKNOWN_A9B1_UNK6
             | adreno.A6XX_SP_CS_UNKNOWN_A9B1_SHARED_SIZE(max(1, (prg.shmem - 1) // 1024)), 0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr),
             adreno.A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM(prg.pvtmem_size_per_item), *data64_le(prg.device._stack.va_addr),
             adreno.A6XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(prg.pvtmem_size_total))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG,
             adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_CONSTANTS) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
             | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_SHADER) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(prg.kernargs_alloc_size // 4),
             *data64_le(args_state.ptr))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG, adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_SHADER) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
             | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_SHADER) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(math.ceil(prg.image_size / 128)),
             *data64_le(prg.lib_gpu.va_addr))    
    self.reg(adreno.REG_A6XX_HLSQ_CONTROL_2_REG, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
             adreno.A6XX_HLSQ_CS_CNTL_CONSTLEN(prg.kernargs_alloc_size // 4) | adreno.A6XX_HLSQ_CS_CNTL_ENABLED)

    self.reg(adreno.REG_A6XX_SP_CS_PVT_MEM_HW_STACK_OFFSET, prg.hw_stack_offset)
    self.reg(adreno.REG_A6XX_SP_CS_INSTRLEN, prg.image_size // 4)

    if hasattr(args_state, 'samplers_ptr'):      
      self.cmd(adreno.CP_LOAD_STATE6_FRAG,
               adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_SHADER) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
               | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_TEX) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(args_state.samplers_cnt),
               *data64_le(args_state.samplers_ptr.va_addr))

      self.reg(adreno.REG_A6XX_SP_CS_TEX_SAMP, *data64_le(args_state.samplers_ptr.va_addr))
      self.reg(adreno.REG_A6XX_SP_PS_TP_BORDER_COLOR_BASE_ADDR, *data64_le(prg.device._border_color_base()))

    if hasattr(args_state, 'descriptors_ptr'):
      self.cmd(adreno.CP_LOAD_STATE6_FRAG,
               adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_CONSTANTS) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
               | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_TEX) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(args_state.descriptors_cnt),
               *data64_le(args_state.descriptors_ptr.va_addr))
      self.reg(adreno.REG_A6XX_SP_CS_TEX_CONST, *data64_le(args_state.descriptors_ptr.va_addr))

    if hasattr(args_state, 'ibos_ptr'):
      self.cmd(adreno.CP_LOAD_STATE6_FRAG,
               adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST6_IBO) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
               | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_SHADER) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(args_state.ibos_cnt),
               *data64_le(args_state.ibos_ptr.va_addr))
      self.reg(adreno.REG_A6XX_SP_CS_IBO, *data64_le(args_state.ibos_ptr.va_addr))

    self.reg(adreno.REG_A6XX_SP_CS_CONFIG,
             adreno.A6XX_SP_CS_CONFIG_ENABLED | adreno.A6XX_SP_CS_CONFIG_NSAMP(args_state.samplers_cnt)
             | adreno.A6XX_SP_CS_CONFIG_NTEX(args_state.descriptors_cnt) | adreno.A6XX_SP_CS_CONFIG_NIBO(args_state.ibos_cnt))

    self.cmd(adreno.CP_RUN_OPENCL, 0)

  def _update_exec(self, cmd_idx, global_size, local_size):
    if global_size is not None:
      self._patch(cmd_idx, offset=11, data=global_size)
      self.cmd_idx_to_dims[cmd_idx][0] = global_size

    if local_size is not None:      
      payload = (3 | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(local_size[0] - 1) | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(local_size[1] - 1)
                 | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(local_size[2] - 1))

      self._patch(cmd_idx, offset=2, data=[payload])
      self.cmd_idx_to_dims[cmd_idx][1] = local_size

    global_size_mp = self.cmd_idx_to_dims[cmd_idx][0]
    if self.cmd_idx_to_dims[cmd_idx][1]:
      global_size_mp = cast(Tuple[int,int,int], tuple(int(g*l)for g,l in zip(
        self.cmd_idx_to_dims[cmd_idx][0], self.cmd_idx_to_dims[cmd_idx][1])))
    self._patch(cmd_idx, offset=3, data=[global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0])

class QcomProgram(HCQProgram):
  def __init__(self, device: QcomDevice, name: str, lib: bytes):
    self.device, self.name, self.lib = device, name, lib

    image_offset, self.image_size, self.prg_offset, self.buffs_info, self.consts_info, self.hlfreg, self.fullreg, \
      self.branch_stack, self.shmem, self.pvtmem = parse_cl_lib(self.lib, self.name)

    image = bytearray(lib[image_offset:image_offset+self.image_size])

    # reserve some space after for gpu to use
    self.lib_gpu = self.device.allocator.alloc(len(image) + 0x20000, options=BufferOptions(cpu_access=True, nolru=True))
    ctypes.memmove(self.lib_gpu.va_addr, mv_address(memoryview(image)), self.image_size)

    self.pvtmem_size_per_item = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    device._ensure_stack_size(self.hw_stack_offset * 4)

    super().__init__(QcomArgsState, self.device, self.name, kernargs_alloc_size=1024)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_gpu.size, options=BufferOptions(cpu_access=True, nolru=True))

class QcomArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:QcomProgram, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)
    self.ibos_cnt, self.descriptors_cnt, self.samplers_cnt = 0, 0, 0
    ctypes.memset(ptr, 0, 1024)

    if len(bufs) + len(vals) != len(prg.buffs_info): raise RuntimeError(f'incorrect args size given={len(bufs)} != want={len(prg.buffs_info)}')
    self.boffs, self.aoffs = self.prg.buffs_info[:len(bufs)], self.prg.buffs_info[len(bufs):]
    for i, v in enumerate(vals): self.update_var(i, v)
    for cnst_val, cnst_off, cnst_sz in self.prg.consts_info:
      ctypes.memmove(self.ptr + cnst_off, (ctypes.c_int8 * cnst_sz).from_buffer_copy(cnst_val.to_bytes(cnst_sz, byteorder='little')), cnst_sz)

    samplers, descriptors, ibos, self.i2descr, self.i2ibo = [], [], [], {}, {}
    for i, b in enumerate(bufs):
      if not hasattr(b, 'samplers') and not hasattr(b, 'descriptor') and not hasattr(b, 'ibo'): self.update_buffer(i, b)
      else:
        if self.boffs[i][1]: ibos, self.i2ibo = [*ibos, *b.ibo], {**self.i2ibo, i: len(ibos)}
        else: samplers, descriptors, self.i2descr = [*samplers, *b.samplers], [*descriptors, *b.descriptor], {**self.i2descr, i: len(descriptors)}

    def alloc_tex_gpu(data, chunk_size):
      to_mv((ptr:=prg.device._gpu_alloc(len(data) * 4, map_to_cpu=True)).va_addr, len(data) * 4).cast('I')[:] = array.array('I', data)
      return ptr, len(data) // chunk_size

    if len(samplers): self.samplers_ptr, self.samplers_cnt = alloc_tex_gpu(samplers, 4)
    if len(descriptors): self.descriptors_ptr, self.descriptors_cnt = alloc_tex_gpu(descriptors, 16)
    if len(ibos): self.ibos_ptr, self.ibos_cnt = alloc_tex_gpu(ibos, 16)

  def __del__(self):
    self.prg.device.synchronize()
    if hasattr(self, 'samplers_ptr'): self.prg.device._gpu_free(self.samplers_ptr)
    if hasattr(self, 'descriptors_ptr'): self.prg.device._gpu_free(self.descriptors_ptr)
    if hasattr(self, 'ibos_ptr'): self.prg.device._gpu_free(self.ibos_ptr)

  def update_buffer(self, index:int, buf:HCQBuffer):
    if (descr:=self.i2descr.get(index, None)) is not None: to_mv(self.descriptors_ptr + 16 * descr + 4 * 4, 8).cast('I')[:] = data64_le(buf.va_addr)
    elif (ibo:=self.i2ibo.get(index, None)) is not None: to_mv(self.ibos_ptr + 16 * ibo + 4 * 4, 8).cast('I')[:] = data64_le(buf.va_addr)
    else: ctypes.cast(self.ptr + self.boffs[index][0], ctypes.POINTER(ctypes.c_int64))[0] = buf.va_addr

  def update_var(self, index:int, val:int): ctypes.cast(self.ptr + self.aoffs[index][0], ctypes.POINTER(ctypes.c_int32))[0] = val
