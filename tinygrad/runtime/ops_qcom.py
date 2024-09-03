from __future__ import annotations
import os, time, ctypes, functools, mmap, struct, array, decimal, math
from types import SimpleNamespace
from typing import Tuple, List, Dict, Any, cast
from tinygrad.device import BufferOptions, HCQBuffer, HWComputeQueue, HCQProgram, HCQCompiled, HCQSignal, HCQAllocator, HCQArgsState, hcq_command
from tinygrad.runtime.autogen import kgsl, adreno, libc
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice
from tinygrad.renderer.cstyle import QCOMRenderer
from tinygrad.helpers import getenv, from_mv, mv_address, to_mv, round_up, data64_le, prod
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

def next_power2(x): return 1 if x == 0 else 1 << (x - 1).bit_length()

def prt(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return adreno.CP_TYPE7_PKT | cnt & 0x3FFF | prt(cnt) << 15 | (opcode & 0x7F) << 16 | prt(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return adreno.CP_TYPE4_PKT | cnt & 0x7F | prt(cnt) << 7 | (reg & 0x3FFFF) << 8 | prt(reg) << 27

class QCOMCompiler(CLCompiler):
  def __init__(self, device:str=""): super().__init__(CLDevice(device), 'compile_qcom')

class QCOMSignal(HCQSignal):
  def __init__(self, value=0, **kwargs):
    self._signal = QCOMDevice.signals_pool.pop()
    super().__init__(value)
  def __del__(self): QCOMDevice.signals_pool.append(self._signal)
  def _get_value(self) -> int: return self._signal[0]
  def _get_timestamp(self) -> decimal.Decimal: return decimal.Decimal(self._signal[1]) / decimal.Decimal(19.2) # based on the 19.2MHz always-on timer
  def _set_value(self, new_value:int): self._signal[0] = new_value
  def wait(self, value:int, timeout:int=60000):
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if self._signal[0] >= value: return
    raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

class QCOMComputeQueue(HWComputeQueue):
  def __init__(self):
    self.cmd_idx_to_dims = {}
    super().__init__()

  def cmd(self, opcode: int, *vals: int): self.q += [pkt7_hdr(opcode, len(vals)), *vals]

  def reg(self, reg: int, *vals: int): self.q += [pkt4_hdr(reg, len(vals)), *vals]

  def _signal(self, signal, value=0, ts=False):
    if QCOMDevice.gpu_id < 700:
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

  def _build_gpu_command(self, device):
    to_mv((hw_page_addr:=device._alloc_cmd_buf(len(self.q) * 4)), len(self.q) * 4).cast('I')[:] = array.array('I', self.q)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(self.q) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=device.ctx,
                                              cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    return submit_req, obj

  def bind(self, device):
    self.binded_device = device
    self.submit_req, self.obj = self._build_gpu_command(self.binded_device)
    # From now on, the queue is on the device for faster submission.
    self.q = to_mv(self.obj.gpuaddr, len(self.q) * 4).cast("I") # type: ignore

  def _submit(self, device):
    if self.binded_device == device: submit_req = self.submit_req
    else: submit_req, _ = self._build_gpu_command(device)
    kgsl.IOCTL_KGSL_GPU_COMMAND(device.fd, __payload=submit_req)

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
             0xfc | adreno.A6XX_HLSQ_CS_CNTL_1_THREADSIZE(adreno.THREAD64),
             int(math.ceil(global_size[0])), int(math.ceil(global_size[1])), int(math.ceil(global_size[2])))
    self.reg(adreno.REG_A6XX_SP_CS_CTRL_REG0,
             adreno.A6XX_SP_CS_CTRL_REG0_THREADSIZE(adreno.THREAD64) | adreno.A6XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT(prg.hregs_count)
             | adreno.A6XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT(prg.fregs_count) | adreno.A6XX_SP_CS_CTRL_REG0_BRANCHSTACK(prg.branch_stack // 2),
             adreno.A6XX_SP_CS_UNKNOWN_A9B1_UNK5 | adreno.A6XX_SP_CS_UNKNOWN_A9B1_UNK6 | adreno.A6XX_SP_CS_UNKNOWN_A9B1_SHARED_SIZE(prg.shared_size),
              0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr), adreno.A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM(prg.pvtmem_size_per_item),
              *data64_le(prg.device._stack.va_addr), adreno.A6XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(prg.pvtmem_size_total))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG,
             adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_CONSTANTS) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
             | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_SHADER) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(prg.kernargs_alloc_size // 4),
             *data64_le(args_state.ptr))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG, adreno.CP_LOAD_STATE6_0_STATE_TYPE(adreno.ST_SHADER) | adreno.CP_LOAD_STATE6_0_STATE_SRC(adreno.SS6_INDIRECT)
             | adreno.CP_LOAD_STATE6_0_STATE_BLOCK(adreno.SB6_CS_SHADER) | adreno.CP_LOAD_STATE6_0_NUM_UNIT(round_up(prg.image_size, 128) // 128),
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
      self._patch(cmd_idx, offset=11, data=[int(math.ceil(global_size[0])), int(math.ceil(global_size[1])), int(math.ceil(global_size[2]))])
      self.cmd_idx_to_dims[cmd_idx][0] = global_size

    if local_size is not None:
      payload = (adreno.A6XX_HLSQ_CS_NDRANGE_0_KERNELDIM(3) | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(local_size[0] - 1)
                 | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(local_size[1] - 1) | adreno.A6XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(local_size[2] - 1))

      self._patch(cmd_idx, offset=2, data=[payload])
      self.cmd_idx_to_dims[cmd_idx][1] = local_size

    global_size_mp = self.cmd_idx_to_dims[cmd_idx][0]
    if self.cmd_idx_to_dims[cmd_idx][1]:
      global_size_mp = cast(Tuple[int,int,int], tuple(int(g*l) for g,l in zip(self.cmd_idx_to_dims[cmd_idx][0], self.cmd_idx_to_dims[cmd_idx][1])))
    self._patch(cmd_idx, offset=3, data=[global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0])

class QCOMArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:QCOMProgram, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)
    self.ibos_cnt, self.descriptors_cnt, self.samplers_cnt = 0, 0, 0
    ctypes.memset(ptr, 0, 1024)

    if len(bufs) + len(vals) != len(prg.buffs_info): raise RuntimeError(f'incorrect args size given={len(bufs)} != want={len(prg.buffs_info)}')
    self.boffs, self.aoffs = prg.buffs_info[:len(bufs)], prg.buffs_info[len(bufs):]
    for i, v in enumerate(vals): self.update_var(i, v)
    for cnst_val, cnst_off, cnst_sz in prg.consts_info:
      ctypes.memmove(self.ptr + cnst_off, (ctypes.c_int8 * cnst_sz).from_buffer_copy(cnst_val.to_bytes(cnst_sz, byteorder='little')), cnst_sz)

    samplers: List[Any] = []
    descriptors: List[Any] = []
    ibos: List[Any] = []
    self.i2descr: Dict[int, int] = {}
    self.i2ibo: Dict[int, int] = {}
    for i, b in enumerate(bufs):
      if not hasattr(b, 'samplers') and not hasattr(b, 'descriptor') and not hasattr(b, 'ibo'): self.update_buffer(i, b)
      elif self.boffs[i][1]: ibos, self.i2ibo = [*ibos, *getattr(b, 'ibo')], {**self.i2ibo, i: len(ibos)}
      else:
        samplers, descriptors = [*samplers, *getattr(b, 'samplers')], [*descriptors, *getattr(b, 'descriptor')]
        self.i2descr[i] = len(descriptors) - 1

    def alloc_tex_gpu(data, chunk_size) -> Tuple[HCQBuffer, int]:
      tex_gpu = self.prg.device.allocator.alloc(len(data) * 4, BufferOptions(nolru=True, cpu_access=True))
      to_mv(tex_gpu.va_addr, len(data) * 4).cast('I')[:] = array.array('I', data)
      return tex_gpu, len(data) // chunk_size

    if len(samplers): self.samplers_ptr, self.samplers_cnt = alloc_tex_gpu(samplers, 4)
    if len(descriptors): self.descriptors_ptr, self.descriptors_cnt = alloc_tex_gpu(descriptors, 16)
    if len(ibos): self.ibos_ptr, self.ibos_cnt = alloc_tex_gpu(ibos, 16)

  def __del__(self):
    for ptr in ('samplers_ptr', 'descriptors_ptr', 'ibos_ptr'):
      if hasattr(self, ptr): self.prg.device.allocator.free((x:=getattr(self, ptr)), x.size, BufferOptions(nolru=True, cpu_access=True))

  def update_buffer(self, index:int, buf:HCQBuffer):
    if (descr:=self.i2descr.get(index, None)) is not None: to_mv(self.descriptors_ptr.va_addr + 16 * descr + 4 * 4, 8).cast('Q')[0] = buf.va_addr
    elif (ibo:=self.i2ibo.get(index, None)) is not None: to_mv(self.ibos_ptr.va_addr + 16 * ibo + 4 * 4, 8).cast('Q')[0] = buf.va_addr
    else: to_mv(self.ptr + self.boffs[index][0], 8).cast('Q')[0] = buf.va_addr

  def update_var(self, index:int, val:int): to_mv(self.ptr + self.aoffs[index][0], 8).cast('Q')[0] = val

class QCOMProgram(HCQProgram):
  def __init__(self, device: QCOMDevice, name: str, lib: bytes):
    self.device, self.name, self.lib = device, name, lib
    self._parse_lib()

    # reserve some space after for gpu to use
    self.lib_gpu = self.device.allocator.alloc(len(self.image) + 0x20000, options=BufferOptions(cpu_access=True, nolru=True))
    to_mv(self.lib_gpu.va_addr, self.image_size)[:] = self.image

    self.pvtmem_size_per_item = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs_count + round_up(self.hregs_count, 2) // 2)) * 128)) * 128)
    device._ensure_stack_size(self.hw_stack_offset * 4)

    super().__init__(QCOMArgsState, self.device, self.name, kernargs_alloc_size=1024)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requsted for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

  def _parse_lib(self):
    def _read_lib(off): return struct.unpack("I", self.lib[off:off+4])[0]

    # Extract image binary
    self.image_size = _read_lib(0x100)
    self.image = bytearray(self.lib[(image_offset:=_read_lib(0xc0)):image_offset+self.image_size])

    # Parse image descriptors
    image_desc_off = _read_lib(0x110)
    self.prg_offset, self.branch_stack = _read_lib(image_desc_off+0xc4), _read_lib(image_desc_off+0x108)
    self.pvtmem, self.shmem = _read_lib(image_desc_off+0xc8), _read_lib(image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.buffs_info, self.consts_info = [], []

    samplers_count = _read_lib(image_desc_off + 0xdc)
    bdoff = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samplers_count
    while (bdoff + 16 <= len(self.lib)):
      length, _, _, offset_words = struct.unpack("I" * 4, self.lib[bdoff:bdoff+16])
      if length == 0: break
      self.buffs_info.append((offset_words * 4, struct.unpack("I", self.lib[bdoff+0x3c:bdoff+0x40])[0] == 0x0))
      bdoff += length

    if _read_lib(0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(0xac)
      while (cdoff + 40 <= image_offset):
        cnst, offset_words, _, is32 = struct.unpack("I", self.lib[cdoff:cdoff+4])[0], *struct.unpack("III", self.lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(0x34)
    self.fregs_count, self.hregs_count = _read_lib(reg_desc_off + 0x14), _read_lib(reg_desc_off + 0x18)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_gpu.size, options=BufferOptions(cpu_access=True, nolru=True))

class QCOMAllocator(HCQAllocator):
  def __init__(self, device:QCOMDevice): super().__init__(device)

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer:
    if options.image is not None:
      pitch = round_up(round_up(options.image.shape[1], 16) * (4 * options.image.base.itemsize), 1 << (pitchalign:=6))
      texture = self.device._gpu_alloc(pitch * round_up(options.image.shape[0], 16), kgsl.KGSL_MEMTYPE_TEXTURE, map_to_cpu=True)

      # save it here to load in one command (the same approach as OpenCL and mesa)
      texture.samplers, texture.descriptor, texture.ibo = [0] * 4, [0] * 16, [0] * 16
      texture.samplers[0:2] = [0x1b60, 0x30] # compiled sampler. always the same in tinygrad.

      texture.descriptor[0] = adreno.A6XX_TEX_CONST_0_SWIZ_X(0) | adreno.A6XX_TEX_CONST_0_SWIZ_Y(1) | adreno.A6XX_TEX_CONST_0_SWIZ_Z(2) \
        | adreno.A6XX_TEX_CONST_0_SWIZ_W(3) \
        | adreno.A6XX_TEX_CONST_0_FMT(adreno.FMT6_32_32_32_32_FLOAT if options.image.itemsize == 4 else adreno.FMT6_16_16_16_16_FLOAT)
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

MAP_FIXED = 0x10
class QCOMDevice(HCQCompiled):
  signals_page: Any = None
  signals_pool: List[Any] = []
  gpu_id: int = 0

  def __init__(self, device:str=""):
    self.fd = os.open('/dev/kgsl-3d0', os.O_RDWR)
    QCOMDevice.signals_page = self._gpu_alloc(16 * 65536, map_to_cpu=True, uncached=True)
    QCOMDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, self.signals_page.size, 16)]
    info, self.ctx, self.cmd_buf, self.cmd_buf_ptr = self._info(), self._ctx_create(), self._gpu_alloc(0x1000000, map_to_cpu=True), 0
    QCOMDevice.gpu_id = ((info.chip_id >> 24) & 0xFF) * 100 + ((info.chip_id >> 16) & 0xFF) * 10 + ((info.chip_id >>  8) & 0xFF)
    if QCOMDevice.gpu_id >= 700: raise RuntimeError(f"Unsupported GPU: {QCOMDevice.gpu_id}")

    super().__init__(device, QCOMAllocator(self), QCOMRenderer(), QCOMCompiler(device), functools.partial(QCOMProgram, self),
                     QCOMSignal, QCOMComputeQueue, None, timeline_signals=(QCOMSignal(), QCOMSignal()))

    QCOMComputeQueue().setup().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

  def _ctx_create(self):
    cr = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=(kgsl.KGSL_CONTEXT_TYPE(kgsl.KGSL_CONTEXT_TYPE_CL) | kgsl.KGSL_CONTEXT_PREAMBLE |
                                                         kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE |
                                                         kgsl.KGSL_CONTEXT_PREEMPT_STYLE(kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)))
    return cr.drawctxt_id

  def _info(self):
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    return info

  def _gpu_alloc(self, size:int, flags:int=0, map_to_cpu=False, uncached=False, fill_zeroes=False):
    size = round_up(size, 1 << (alignment_hint:=12))
    flags |= (kgsl.KGSL_MEMALIGN(alignment_hint))
    if uncached: flags |= (kgsl.KGSL_CACHEMODE(kgsl.KGSL_CACHEMODE_UNCACHED))

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=size, flags=flags)
    va_addr, va_len = None, 0
    if not (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP):
      info = kgsl.IOCTL_KGSL_GPUOBJ_INFO(self.fd, id=alloc.id)
      va_addr, va_len = info.gpuaddr, info.va_len

    if map_to_cpu or (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP):
      va_addr = libc.mmap(va_addr, va_len := (va_len or alloc.mmapsize), mmap.PROT_READ|mmap.PROT_WRITE,
                          mmap.MAP_SHARED | (MAP_FIXED if va_addr is not None else 0), self.fd, alloc.id * 0x1000)
      if fill_zeroes: ctypes.memset(va_addr, 0, va_len)

    return SimpleNamespace(va_addr=va_addr, size=va_len, mapped=map_to_cpu or (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP), info=alloc)

  def _gpu_free(self, mem):
    kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.info.id)
    if mem.mapped: libc.munmap(mem.va_addr, mem.size)

  def _alloc_cmd_buf(self, sz: int):
    self.cmd_buf_ptr = (cur_ptr:=self.cmd_buf_ptr if self.cmd_buf_ptr + sz < self.cmd_buf.size else 0) + sz
    return self.cmd_buf.va_addr + cur_ptr

  def _border_color_base(self):
    if not hasattr(self, '_border_color_gpu'): self._border_color_gpu = self._gpu_alloc(0x1000, map_to_cpu=True, fill_zeroes=True)
    return self._border_color_gpu.va_addr

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)
