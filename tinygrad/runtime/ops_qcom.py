import os, time, math, ctypes, fcntl, functools, mmap, struct, array, decimal
from types import SimpleNamespace
from typing import Tuple, List, Any, cast

from tinygrad.device import BufferOptions, HCQBuffer, HWComputeQueue, HCQProgram, HCQCompiled, HCQSignal, HCQAllocator, hcq_command
import tinygrad.runtime.autogen.kgsl as kgsl
import tinygrad.runtime.autogen.adreno as adreno
import tinygrad.runtime.autogen.libc as libc
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice
from tinygrad.renderer.cstyle import QCOMRenderer
from tinygrad.helpers import getenv, from_mv, mv_address, to_mv, round_up, data64_le
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401

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
  image_offset, image_size = struct.unpack("I", lib[0xC0:0xC4])[0], struct.unpack("I", lib[0x100:0x104])[0]

  # parse argument buffers layout
  buffs_info = []
  ptr = struct.unpack("I", lib[0x110:0x114])[0] # read img desc offset
  prg_offset = struct.unpack("I", lib[ptr+196:ptr+200])[0]

  ptr = round_up(ptr + 344 + len(name), 4) # skip to bufs descr, align name to 4 bytes
  while (ptr + 16 <= len(lib)):
    length, _, _, offset_words = struct.unpack("I" * 4, lib[ptr:ptr+16])
    if length == 0: break
    ptr += length
    buffs_info.append(offset_words * 4)

  # parse constants layout
  consts_info = []
  ptr = struct.unpack("I", lib[0xb0:0xb4])[0]
  # check for consts
  if ptr != 0:
    ptr = struct.unpack("I", lib[0xac:0xb0])[0] # read consts desc offset
    # constant vals are placed just before a shader
    while (ptr + 40 <= image_offset):
      cnst, offset_words, _, is32 = struct.unpack("I", lib[ptr:ptr+4])[0], *struct.unpack("III", lib[ptr+16:ptr+28])
      ptr += 40
      consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))

  ptr = struct.unpack("I", lib[0x34:0x38])[0] # read main offset
  fullreg, halfreg = struct.unpack("II", lib[ptr+20:ptr+28])

  return image_offset, image_size, prg_offset, buffs_info, consts_info, halfreg, fullreg

class QcomCompiler(CLCompiler):
  def __init__(self, device:str=""): super().__init__(CLDevice(device), 'compile_qcom')

class QcomSignal(HCQSignal):
  def __init__(self, value=0, **kwargs):
    self._signal = QcomDevice.signals_pool.pop()
    super().__init__(value)
  def __del__(self): QcomDevice.signals_pool.append(self._signal)
  def _get_value(self) -> int: return self._signal[0]
  def _get_timestamp(self) -> decimal.Decimal: return decimal.Decimal(self._signal[1]) / decimal.Decimal(1000)
  def _set_value(self, new_value:int): self._signal[0] = new_value
  def wait(self, value:int, timeout:int=10000):
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if self._signal[0] >= value: return
    raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

MAP_FIXED = 0x10
class QcomDevice(HCQCompiled):
  signals_page:Any = None
  signals_pool: List[Any] = []

  def __init__(self, device:str=""):
    self.fd = os.open('/dev/kgsl-3d0', os.O_RDWR)
    QcomDevice.signals_page = self._gpu_alloc(16 * 65536, flags=0xC0A00, map_to_cpu=True, uncached=True)
    QcomDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, self.signals_page.size, 16)]
    self.ctx, self.cmd_buf, self.cmd_buf_ptr = self._ctx(), self._gpu_alloc(0x100000, 0xC0A00, map_to_cpu=True), 0

    super().__init__(device, QcomAllocator(self), QCOMRenderer(), QcomCompiler(device), functools.partial(QcomProgram, self),
                     QcomSignal, QcomComputeQueue, None, timeline_signals=(QcomSignal(), QcomSignal()))

    QcomComputeQueue().setup().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

  def __del__(self): self._gpu_free(self.cmd_buf)

  def _ctx(self):
    cr = kgsl.struct_kgsl_drawctxt_create(flags=(kgsl.KGSL_CONTEXT_TYPE_CL << kgsl.KGSL_CONTEXT_TYPE_SHIFT) | kgsl.KGSL_CONTEXT_PREAMBLE
                                                 | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE)
    self._ioctl(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, cr)
    self.context_id = cr.drawctxt_id
    return self.context_id

  def _ioctl(self, nr, arg):
    ret = fcntl.ioctl(self.fd, (3 << 30) | (ctypes.sizeof(arg) & 0x1FFF) << 16 | 0x9 << 8 | (nr & 0xFF), arg)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return ret

  def _gpu_alloc(self, size:int, flags:int=0, map_to_cpu=False, uncached=False):
    size = round_up(size, 1 << (alignment_hint:=12))
    flags |= ((alignment_hint << kgsl.KGSL_MEMALIGN_SHIFT) & kgsl.KGSL_MEMALIGN_MASK)
    if uncached: flags |= ((kgsl.KGSL_CACHEMODE_UNCACHED << kgsl.KGSL_CACHEMODE_SHIFT) & kgsl.KGSL_CACHEMODE_MASK)

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

    return SimpleNamespace(va_addr=va_addr, size=va_len, info=alloc)

  def _gpu_free(self, opaque):
    free = kgsl.struct_kgsl_gpuobj_free(id=opaque.info.id)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free)

  def _alloc_cmd_buf(self, sz: int):
    self.cmd_buf_ptr = (cur_ptr:=self.cmd_buf_ptr if self.cmd_buf_ptr + sz < self.cmd_buf.size else 0) + sz
    return self.cmd_buf.va_addr + cur_ptr

class QcomAllocator(HCQAllocator):
  def __init__(self, device:QcomDevice): super().__init__(device)

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer: return self.device._gpu_alloc(size, map_to_cpu=True)

  def copyin(self, dest:HCQBuffer, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)

  def copyout(self, dest:memoryview, src:HCQBuffer):
    self.device.synchronize()
    ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)

  def _free(self, opaque, options:BufferOptions): return self.device._gpu_free(opaque)

class QcomComputeQueue(HWComputeQueue):
  def __init__(self):
    self.q, self.cmd_idx_to_dims = [], {}
    super().__init__()

  def cmd(self, opcode: int, *vals: int): self.q += [pkt7_hdr(opcode, len(vals)), *vals]

  def reg(self, reg: int, *vals: int): self.q += [pkt4_hdr(reg, len(vals)), *vals]

  def _signal(self, signal, value=0): self.cmd(adreno.CP_MEM_WRITE, *data64_le(mv_address(signal._signal)), *data64_le(value))

  def _wait(self, signal, value=0):
    self.cmd(adreno.CP_WAIT_REG_MEM, adreno.WRITE_GE | (adreno.POLL_MEMORY << adreno.CP_WAIT_REG_MEM_0_POLL__SHIFT),
             *data64_le(mv_address(signal._signal)), value & 0xffffffff, 0xffffffff, 32) # busy wait for 32 cycles

  def _update_signal(self, cmd_idx, signal, value):
    if signal is not None: self._patch(cmd_idx, offset=1, data=data64_le(mv_address(signal._signal)))
    if value is not None: self._patch(cmd_idx, offset=3, data=data64_le(value))

  def _update_wait(self, cmd_idx, signal, value):
    if signal is not None: self._patch(cmd_idx, offset=2, data=data64_le(mv_address(signal._signal)))
    if value is not None: self._patch(cmd_idx, offset=4, data=[value & 0xffffffff])

  def _timestamp(self, signal):
    # doesnt really work right now
    self.cmd(adreno.CP_REG_TO_MEM, 0x40000980,  *data64_le(mv_address(signal._signal)))
    self.cmd(adreno.CP_EVENT_WRITE7, adreno.CACHE_INVALIDATE)

  def bind(self, device):
    self.binded_device = device

    cmdbytes, hw_page_addr = array.array('I', self.q), device._alloc_cmd_buf(len(self.q) * 4)
    ctypes.memmove(hw_page_addr, mv_address(memoryview(cmdbytes)), len(cmdbytes) * 4)

    self.obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(cmdbytes) * 4, flags=0x1)
    self.submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(self.obj), cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object),
                                                   numcmds=1, context_id=device.ctx)
    self.q = to_mv(hw_page_addr, len(self.q) * 4).cast("I")

  def _submit(self, device: QcomDevice):
    if self.binded_device == device:
      device._ioctl(kgsl.IOCTL_KGSL_GPU_COMMAND, self.submit_req)
      return

    cmdbytes, hw_page_addr = array.array('I', self.q), device._alloc_cmd_buf(len(self.q) * 4)
    ctypes.memmove(hw_page_addr, mv_address(memoryview(cmdbytes)), len(cmdbytes) * 4)

    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(cmdbytes) * 4, flags=0x1)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object),
                                              numcmds=1, context_id=device.ctx)
    device._ioctl(kgsl.IOCTL_KGSL_GPU_COMMAND, submit_req)

  @hcq_command
  def setup(self):
    self.cmd(adreno.CP_WAIT_FOR_IDLE)
    self.cmd(adreno.CP_SET_MARKER, adreno.RM6_COMPUTE)
    self.reg(adreno.REG_A6XX_HLSQ_CONTROL_2_REG, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, 0x140)
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, adreno.A6XX_HLSQ_INVALIDATE_CMD_CS_STATE | adreno.A6XX_HLSQ_INVALIDATE_CMD_CS_IBO)
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, 0x0)
    self.reg(adreno.REG_A6XX_SP_CS_TEX_COUNT, 0x80)
    self.reg(adreno.REG_A6XX_SP_CS_IBO_COUNT, 0x40)
    self.reg(adreno.REG_A6XX_SP_MODE_CONTROL, adreno.ISAMMODE_CL << adreno.A6XX_SP_MODE_CONTROL_ISAMMODE__SHIFT)
    self.reg(adreno.REG_A6XX_SP_PERFCTR_ENABLE, adreno.A6XX_SP_PERFCTR_ENABLE_CS)
    self.reg(adreno.REG_A6XX_SP_TP_MODE_CNTL, adreno.ISAMMODE_CL | (1 << 3)) # ISAMMODE|UNK3
    self.reg(adreno.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    self.reg(adreno.REG_A6XX_SP_CHICKEN_BITS, 0x20)
    self.reg(adreno.REG_A6XX_SP_CS_CONFIG, adreno.A6XX_SP_CS_CONFIG_ENABLED)
    self.reg(adreno.REG_A6XX_SP_CS_PVT_MEM_HW_STACK_OFFSET, 0)

  def _exec(self, prg, kernargs, global_size, local_size):
    global_size_mp = cast(Tuple[int,int,int], tuple(int(g*l) for g,l in zip(global_size, local_size))) if local_size else global_size

    self.cmd(adreno.CP_WAIT_FOR_IDLE)
    self.cmd_idx_to_dims[len(self) - 1] = [global_size, local_size]
    self.reg(
      adreno.REG_A6XX_HLSQ_CS_NDRANGE_0,
        # kernel dimenstion = 3
        3 | ((local_size[0] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX__SHIFT)
        | ((local_size[1] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY__SHIFT)
        | ((local_size[2] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ__SHIFT),
        global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, # global size x,y,z followed by offsets
        0xccc0cf, 0x2fc,
        global_size[0], global_size[1], global_size[2], # original global sizes
    )
    self.reg(
      adreno.REG_A6XX_SP_CS_CTRL_REG0,
      (adreno.THREAD128 << adreno.A6XX_SP_CS_CTRL_REG0_THREADSIZE__SHIFT)
      | (prg.halfreg << adreno.A6XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT__SHIFT)
      | (prg.fullreg << adreno.A6XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT__SHIFT)
      | (4 << adreno.A6XX_SP_CS_CTRL_REG0_BRANCHSTACK__SHIFT),
      0x41, 0, prg.prg_offset, # offsets
      *data64_le(prg.lib_gpu.va_addr), 0, *data64_le(prg.private_gpu.va_addr), prg.private_gpu.size,
    )
    self.cmd(adreno.CP_LOAD_STATE6_FRAG,
             (adreno.ST_CONSTANTS << adreno.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT)
             | (adreno.SS6_INDIRECT << adreno.CP_LOAD_STATE6_0_STATE_SRC__SHIFT)
             | (adreno.SB6_CS_SHADER << adreno.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT)
             | (256 << adreno.CP_LOAD_STATE6_0_NUM_UNIT__SHIFT),
             *data64_le(kernargs))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG,
            (adreno.ST_SHADER << adreno.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT)
            | (adreno.SS6_INDIRECT << adreno.CP_LOAD_STATE6_0_STATE_SRC__SHIFT)
            | (adreno.SB6_CS_SHADER << adreno.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT)
            | (math.ceil(prg.image_size / 128) << adreno.CP_LOAD_STATE6_0_NUM_UNIT__SHIFT),
            *data64_le(prg.lib_gpu.va_addr))
    self.reg(adreno.REG_A6XX_SP_CS_INSTRLEN, prg.image_size // 4)
    self.cmd(adreno.CP_RUN_OPENCL, 0)
    self.cmd(adreno.CP_WAIT_FOR_IDLE)

  def _update_exec(self, cmd_idx, global_size, local_size):
    if global_size is not None:
      self._patch(cmd_idx, offset=11, data=global_size)
      self.cmd_idx_to_dims[cmd_idx][0] = global_size

    if local_size is not None:
      payload = (3 | ((local_size[0] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX__SHIFT)
        | ((local_size[1] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY__SHIFT)
        | ((local_size[2] - 1) << adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ__SHIFT))

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

    self.private_gpu = self.device._gpu_alloc(0x101, 0xC0F00)

    image_offset, self.image_size, self.prg_offset, self.buffs_info, self.consts_info, self.halfreg, self.fullreg = parse_cl_lib(self.lib, self.name)
    image = bytearray(lib[image_offset:image_offset+self.image_size])

    # check if buffers are contigious for hcq
    for i in range(1, len(self.buffs_info)): assert(self.buffs_info[i] - self.buffs_info[i-1] == 0x10)

    self.lib_gpu = self.device._gpu_alloc(len(image), 0x10C0A00, map_to_cpu=True)
    ctypes.memmove(self.lib_gpu.va_addr, mv_address(image), self.image_size)

    super().__init__(self.device, self.name, kernargs_alloc_size=0x1000, kernargs_args_offset=self.buffs_info[0] if len(self.buffs_info) else 0)

  def _fill_kernargs(self, kernargs_ptr:int, bufs:Tuple[Any, ...], vals:Tuple[int, ...]=()):
    if len(bufs) + len(vals) != len(self.buffs_info): raise RuntimeError(f'incorrect args size given={len(bufs)} != want={len(self.buffs_info)}')
    for i, b in enumerate(bufs): ctypes.cast(kernargs_ptr + self.buffs_info[i], ctypes.POINTER(ctypes.c_int64))[0] = b.va_addr
    for i, v in enumerate(vals): ctypes.cast(kernargs_ptr + self.buffs_info[i+len(bufs)], ctypes.POINTER(ctypes.c_int32))[0] = v
    for cnst_val, cnst_off, cnst_sz in self.consts_info:
      ctypes.memmove(kernargs_ptr + cnst_off, (ctypes.c_int8 * cnst_sz).from_buffer_copy(cnst_val.to_bytes(cnst_sz, byteorder='little')), cnst_sz)
