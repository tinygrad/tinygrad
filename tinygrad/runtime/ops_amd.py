from __future__ import annotations
from typing import Tuple, List, Any, cast
import os, fcntl, ctypes, ctypes.util, functools, pathlib, mmap, errno, time, array, contextlib, decimal
from dataclasses import dataclass
from tinygrad.device import HCQCompiled, HCQAllocator, HCQBuffer, HWComputeQueue, HWCopyQueue, HCQArgsState, \
                            HCQSignal, HCQProgram, BufferOptions
from tinygrad.helpers import getenv, to_mv, round_up, data64_le, DEBUG, mv_address
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.runtime.support.compiler_hip import AMDCompiler, disasm
from tinygrad.runtime.support.elf import elf_loader
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import
if getenv("MOCKGPU"): import extra.mockgpu.mockgpu # noqa: F401 # pylint: disable=unused-import

def is_usable_gpu(gpu_id):
  with contextlib.suppress(OSError): return int(pathlib.Path(gpu_id).read_text()) != 0
  return False

def kfd_ioctl(idir, nr, user_struct, fd, **kwargs):
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(made := user_struct(**kwargs))<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

kio:Any = type("KIO", (object,), {name[11:].lower(): functools.partial(kfd_ioctl, {"IOW": 1, "IOR": 2, "IOWR": 3}[p[0]], p[1], p[2])
                              for name,p in kfd.__dict__.items() if name.startswith("AMDKFD_IOC_")})

regBIF_BX_PF1_GPU_HDP_FLUSH_REQ, regBIF_BX_PF1_GPU_HDP_FLUSH_DONE = 0x0106, 0x0107

# VGT_EVENT_TYPE in navi10_enum.h
CACHE_FLUSH_AND_INV_TS_EVENT = 0x14

WAIT_REG_MEM_FUNCTION_EQ = 3 # ==
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

COMPUTE_SHADER_EN, FORCE_START_AT_000, CS_W32_EN = (1 << 0), (1 << 2), (1 << 15)

def gfxreg(reg): return reg + 0x00001260 - amd_gpu.PACKET3_SET_SH_REG_START
def nbioreg(reg): return reg + 0x00000d20 # NBIO_BASE__INST0_SEG2

class AMDSignal(HCQSignal):
  def __init__(self, value=0, alloc_event=False):
    self._signal = AMDDevice.signals_pool.pop()
    self._value_addr, self._timestamp_addr = mv_address(self._signal), mv_address(self._signal) + 8
    if alloc_event:
      sync_event = kio.create_event(AMDDevice.kfd, auto_reset=1)
      self._event_mailbox_ptr = AMDDevice.event_page.va_addr + sync_event.event_slot_index*8
      self._event_id = sync_event.event_id
      self._evt_array = (kfd.struct_kfd_event_data)(event_id=self._event_id)
    else: self._event_mailbox_ptr = self._event_id = 0
    super().__init__(value)
  def __del__(self): AMDDevice.signals_pool.append(self._signal)
  def _get_value(self) -> int: return self._signal[0]
  def _get_timestamp(self) -> decimal.Decimal: return decimal.Decimal(self._signal[1]) / decimal.Decimal(100)
  def _set_value(self, new_value:int): self._signal[0] = new_value
  def wait(self, value:int, timeout:int=10000):
    start_time = time.time() * 1000
    while (time_spent:=time.time() * 1000 - start_time) < timeout:
      if self._signal[0] >= value: return

      # Wait active for 5s, then going to sleep.
      if time_spent > 5000 and self._event_id != 0:
        kio.wait_events(AMDDevice.kfd, events_ptr=ctypes.addressof(self._evt_array), num_events=1, wait_for_all=1, timeout=1000)
    raise RuntimeError(f"wait_signal: not set to {value}, but {self._signal[0]}, {timeout} ms TIMEOUT!")

class AMDComputeQueue(HWComputeQueue):
  def __init__(self):
    self.cmd_idx_to_local_offset, self.cmd_idx_to_global_offset, self.cmd_idx_to_dispatch_packet = {}, {}, {}
    super().__init__()

  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.synchronize()
      self.binded_device._gpu_free(self.hw_page)

  def _acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0, *data64_le(sz), *data64_le(addr), 0,
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)]

  def _release_mem(self, mem_event_type, mem_data_sel, mem_int_sel, address, value=0, cst=0, cache_flush=False):
    cache_flush_flags = 0

    if cache_flush:
      cache_flush_flags = amd_gpu.PACKET3_RELEASE_MEM_GCR_GLV_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GL1_INV | \
        amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_WB | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_INV | \
        amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_WB | amd_gpu.PACKET3_RELEASE_MEM_GCR_SEQ

    # event_index__mec_release_mem__end_of_pipe = 5
    # event_index__mec_release_mem__shader_done = 6
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_RELEASE_MEM, 6),
      amd_gpu.PACKET3_RELEASE_MEM_EVENT_TYPE(mem_event_type) | amd_gpu.PACKET3_RELEASE_MEM_EVENT_INDEX(5) | cache_flush_flags,
      amd_gpu.PACKET3_RELEASE_MEM_DATA_SEL(mem_data_sel) | amd_gpu.PACKET3_RELEASE_MEM_INT_SEL(mem_int_sel) | amd_gpu.PACKET3_RELEASE_MEM_DST_SEL(0),
      *data64_le(address), *data64_le(value), cst]

  def _memory_barrier(self):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5), amd_gpu.WAIT_REG_MEM_MEM_SPACE(0) | amd_gpu.WAIT_REG_MEM_OPERATION(1) | \
      amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ) | amd_gpu.WAIT_REG_MEM_ENGINE(0), nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_REQ),
      nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_DONE), 0xffffffff, 0xffffffff, 0x20]
    self._acquire_mem()

  def _exec(self, prg, args_state, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1)):
    self._acquire_mem(gli=0, gl2=0)

    user_regs, cmd_idx = [], len(self) - 1
    if prg.enable_dispatch_ptr:
      dp = hsa.hsa_kernel_dispatch_packet_t.from_address(dp_addr:=args_state.ptr + prg.kernargs_segment_size)
      dp.workgroup_size_x, dp.workgroup_size_y, dp.workgroup_size_z = local_size[0], local_size[1], local_size[2]
      dp.grid_size_x, dp.grid_size_y, dp.grid_size_z = global_size[0]*local_size[0], global_size[1]*local_size[1], global_size[2]*local_size[2]
      dp.group_segment_size, dp.private_segment_size, dp.kernarg_address = prg.group_segment_size, prg.private_segment_size, args_state.ptr
      user_regs += [*data64_le(dp_addr)]
      self.cmd_idx_to_dispatch_packet[cmd_idx] = dp
    user_regs += [*data64_le(args_state.ptr)]

    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 6), gfxreg(amd_gpu.regCOMPUTE_PGM_LO), *data64_le(prg.prog_addr >> 8),
               *data64_le(0), *data64_le(prg.device.scratch.va_addr >> 8)]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC1), prg.rsrc1, prg.rsrc2]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_TMPRING_SIZE), prg.device.tmpring_size]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_RESTART_X), 0, 0, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE0)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE2)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE4)] + [0xFFFFFFFF] * 4
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, len(user_regs)), gfxreg(amd_gpu.regCOMPUTE_USER_DATA_0)] + user_regs

    self.cmd_idx_to_local_offset[cmd_idx] = len(self.q) - self.cmds_offset[cmd_idx] + 5 # +1 to skip PACKET3_SET_SH_REG + reg + 3 zeros.
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 8), gfxreg(amd_gpu.regCOMPUTE_START_X), 0, 0, 0, *local_size, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_RESOURCE_LIMITS), 0]

    self.cmd_idx_to_global_offset[cmd_idx] = len(self.q) - self.cmds_offset[cmd_idx] + 1 # +1 to skip PACKET3_DISPATCH_DIRECT.
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, 3), *global_size, CS_W32_EN | FORCE_START_AT_000 | COMPUTE_SHADER_EN]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_EVENT_WRITE, 0), amd_gpu.EVENT_TYPE(7) | amd_gpu.EVENT_INDEX(4)]

  def _update_exec(self, cmd_idx, global_size, local_size):
    if local_size is not None: self._patch(cmd_idx, offset=self.cmd_idx_to_local_offset[cmd_idx], data=local_size)
    if global_size is not None: self._patch(cmd_idx, offset=self.cmd_idx_to_global_offset[cmd_idx], data=global_size)

    if (dp:=self.cmd_idx_to_dispatch_packet.get(cmd_idx)) is not None:
      if local_size is not None: dp.workgroup_size_x, dp.workgroup_size_y, dp.workgroup_size_z = local_size[0], local_size[1], local_size[2]
      if global_size is not None:
        dp.grid_size_x,dp.grid_size_y,dp.grid_size_z = [g*l for g,l in zip(global_size,[dp.workgroup_size_x,dp.workgroup_size_y,dp.workgroup_size_z])]

  def _wait(self, signal, value=0):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5),
      amd_gpu.WAIT_REG_MEM_MEM_SPACE(1) | amd_gpu.WAIT_REG_MEM_OPERATION(0) | amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | \
      amd_gpu.WAIT_REG_MEM_ENGINE(0), *data64_le(signal._value_addr), value, 0xffffffff, 4]

  def _timestamp(self, signal): self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=3, mem_int_sel=0, address=signal._timestamp_addr)

  def _signal(self, signal, value=0):
    # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
    self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=1, mem_int_sel=2, address=signal._value_addr, value=value, cache_flush=True)
    if signal._event_mailbox_ptr != 0:
      self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=1, mem_int_sel=2, address=signal._event_mailbox_ptr,
                        value=signal._event_id, cst=signal._event_id, cache_flush=False)

  def _update_wait(self, cmd_idx, signal=None, value=None):
    if signal is not None: self._patch(cmd_idx, offset=2, data=data64_le(signal._value_addr))
    if value is not None: self._patch(cmd_idx, offset=4, data=[value])

  def _update_signal(self, cmd_idx, signal=None, value=None):
    if signal is not None: self._patch(cmd_idx, offset=3, data=data64_le(signal._value_addr))
    if value is not None: self._patch(cmd_idx, offset=5, data=data64_le(value))

    # Check if the signal command has mailptr part
    if signal is not None and self.cmds_len[cmd_idx] > 8:
      self._patch(cmd_idx, offset=11, data=[*data64_le(signal._event_mailbox_ptr), *data64_le(signal._event_id), signal._event_id])

  def bind(self, device):
    self.binded_device = device
    self.hw_page = cast(AMDDevice, device)._gpu_alloc(len(self.q) * 4, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    hw_view = to_mv(self.hw_page.va_addr, self.hw_page.size).cast("I")
    for i, value in enumerate(self.q): hw_view[i] = value

    self.indirect_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), *data64_le(self.hw_page.va_addr),
                         len(self.q) | amd_gpu.INDIRECT_BUFFER_VALID]
    self.q = hw_view # type: ignore

  def _submit(self, device):
    cmds = self.indirect_cmd if device == self.binded_device else self.q

    for i, value in enumerate(cmds): device.compute_queue.ring[(device.compute_queue.put_value + i) % len(device.compute_queue.ring)] = value

    device.compute_queue.put_value += len(cmds)
    device.compute_queue.write_ptr[0] = device.compute_queue.put_value
    device.compute_queue.doorbell[0] = device.compute_queue.put_value

SDMA_MAX_COPY_SIZE = 0x400000
class AMDCopyQueue(HWCopyQueue):
  def __init__(self):
    self.internal_cmd_sizes, self.copy_cmds_per_copy = [], {}
    super().__init__()

  def _q(self, arr):
    self.q += arr
    self.internal_cmd_sizes.append(len(arr))

  def _copy(self, dest, src, copy_size):
    copied, copy_commands = 0, (copy_size + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE
    self.copy_cmds_per_copy[len(self) - 1] = copy_commands
    for _ in range(copy_commands):
      step_copy_size = min(copy_size - copied, SDMA_MAX_COPY_SIZE)

      self._q([amd_gpu.SDMA_OP_COPY | amd_gpu.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(amd_gpu.SDMA_SUBOP_COPY_LINEAR),
        amd_gpu.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size - 1), 0, *data64_le(src + copied), *data64_le(dest + copied)])

      copied += step_copy_size

  def _update_copy(self, cmd_idx, dest=None, src=None):
    for i in range(self.copy_cmds_per_copy[cmd_idx]):
      if src is not None: self._patch(cmd_idx, offset=3+i*7, data=[*data64_le(src + SDMA_MAX_COPY_SIZE*i)])
      if dest is not None: self._patch(cmd_idx, offset=5+i*7, data=[*data64_le(dest + SDMA_MAX_COPY_SIZE*i)])

  def _signal(self, signal, value=0):
    self._q([amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(signal._value_addr), value])

    if signal._event_mailbox_ptr != 0:
      self._q([amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(signal._event_mailbox_ptr), signal._event_id])
      self._q([amd_gpu.SDMA_OP_TRAP, amd_gpu.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(signal._event_id)])

  def _wait(self, signal, value=0):
    self._q([amd_gpu.SDMA_OP_POLL_REGMEM | amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) | \
      amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1), *data64_le(signal._value_addr), value, 0xffffffff,
      amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff)])

  def _update_signal(self, cmd_idx, signal=None, value=None): return self._update_wait(cmd_idx, signal, value) # the same offsets and commands
  def _update_wait(self, cmd_idx, signal=None, value=None):
    if signal is not None: self._patch(cmd_idx, offset=1, data=data64_le(signal._value_addr))
    if value is not None: self._patch(cmd_idx, offset=3, data=[value])

  def _timestamp(self, signal):
    self._q([amd_gpu.SDMA_OP_TIMESTAMP | amd_gpu.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(amd_gpu.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL),
             *data64_le(signal._timestamp_addr)])

  def _submit(self, device):
    if device.sdma_queue.put_value - device.sdma_queue.read_ptr[0] > device.sdma_queue.ring.nbytes: raise RuntimeError("SDMA queue overrun")

    tail_blit_dword = 0
    for cmdsz in self.internal_cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= device.sdma_queue.ring.nbytes - device.sdma_queue.put_value % device.sdma_queue.ring.nbytes: break
      tail_blit_dword += cmdsz

    start_idx = (device.sdma_queue.put_value % device.sdma_queue.ring.nbytes) // 4
    device.sdma_queue.ring[start_idx : start_idx + tail_blit_dword] = array.array('I', self.q[:tail_blit_dword])
    device.sdma_queue.put_value += tail_blit_dword * 4

    if (rem_packet_cnt := len(self.q) - tail_blit_dword) > 0:
      zero_fill = device.sdma_queue.ring.nbytes - device.sdma_queue.put_value % device.sdma_queue.ring.nbytes
      ctypes.memset(mv_address(device.sdma_queue.ring) + (device.sdma_queue.put_value % device.sdma_queue.ring.nbytes), 0, zero_fill)
      device.sdma_queue.put_value += zero_fill

      device.sdma_queue.ring[0:rem_packet_cnt] = array.array('I', self.q[tail_blit_dword:])
      device.sdma_queue.put_value += rem_packet_cnt * 4

    device.sdma_queue.write_ptr[0] = device.sdma_queue.put_value
    device.sdma_queue.doorbell[0] = device.sdma_queue.put_value

class AMDArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:AMDProgram, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)

    self.bufs = to_mv(self.ptr, len(bufs) * 8).cast('Q')
    self.vals = to_mv(self.ptr + len(bufs) * 8, len(vals) * 4).cast('I')

    self.bufs[:] = array.array('Q', [b.va_addr for b in bufs])
    self.vals[:] = array.array('I', vals)

  def update_buffer(self, index:int, buf:HCQBuffer): self.bufs[index] = buf.va_addr
  def update_var(self, index:int, val:int): self.vals[index] = val

class AMDProgram(HCQProgram):
  def __init__(self, device:AMDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6: print(disasm(lib))

    image, sections, _ = elf_loader(self.lib)
    self.lib_gpu = self.device._gpu_alloc(round_up(image.nbytes, 0x1000), kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    ctypes.memmove(self.lib_gpu.va_addr, mv_address(image), image.nbytes)

    entry_point = min(sh.header.sh_addr for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS and sh.header.sh_flags & libc.SHF_ALLOC)
    self.group_segment_size = image[entry_point:entry_point+4].cast("I")[0]
    self.private_segment_size = image[entry_point+4:entry_point+8].cast("I")[0]
    self.kernargs_segment_size = image[entry_point+8:entry_point+12].cast("I")[0]

    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if lds_size > (self.device.properties['lds_size_in_kb'] * 1024) // 512: raise RuntimeError("Too many resources requsted: group_segment_size")
    if self.private_segment_size > self.device.max_private_segment_size: raise RuntimeError("Too many resources requsted: private_segment_size")

    code = hsa.amd_kernel_code_t.from_address(self.lib_gpu.va_addr + entry_point) # NOTE: this is wrong, it's not this object
    assert code.kernel_code_properties & 0x400 == 0x400 # ENABLE_WAVEFRONT_SIZE32

    self.rsrc1 = code.compute_pgm_rsrc1
    self.rsrc2 = code.compute_pgm_rsrc2 | (lds_size << 15)
    self.prog_addr = self.lib_gpu.va_addr + entry_point + code.kernel_code_entry_byte_offset

    # Some programs use hsa_kernel_dispatch_packet_t to read workgroup sizes during execution.
    # The packet is represented as a pointer and set up in SGPRs. Space for the packet is allocated as part of the kernel arguments.
    self.enable_dispatch_ptr = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

    super().__init__(AMDArgsState, self.device, self.name, kernargs_alloc_size=self.kernargs_segment_size+additional_alloc_sz)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): cast(AMDDevice, self.device)._gpu_free(self.lib_gpu)

class AMDAllocator(HCQAllocator):
  def __init__(self, device:AMDDevice): super().__init__(device, batch_size=SDMA_MAX_COPY_SIZE)

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer:
    if options.host: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True)
    return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=options.cpu_access)

  def _free(self, opaque, options:BufferOptions): self.device._gpu_free(opaque)

  def map(self, buf:HCQBuffer): self.device._gpu_map(buf._base if hasattr(buf, '_base') else buf)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400

@dataclass
class AMDQueueDesc:
  ring: memoryview
  read_ptr: memoryview
  write_ptr: memoryview
  doorbell: memoryview
  put_value: int = 0

class AMDDevice(HCQCompiled):
  kfd:int = -1
  event_page:Any = None  # TODO: fix types in kfd, Optional[kfd.struct_kfd_ioctl_alloc_memory_of_gpu_args]
  signals_page:Any = None
  signals_pool:List[memoryview] = []
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

    try: mem = kio.alloc_memory_of_gpu(self.kfd, va_addr=addr, size=size, base=addr, length=size, gpu_id=self.gpu_id, flags=flags, mmap_offset=buf)
    except OSError as e:
      if e.errno == errno.EINVAL and (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) and public:
        raise MemoryError("Cannot allocate host-visible VRAM. Ensure the resizable BAR option is enabled on your system.") from e
      if e.errno == errno.ENOMEM: raise MemoryError("Cannot allocate memory: no memory is available.") from e
      raise

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

  def __init__(self, device:str=""):
    if AMDDevice.kfd == -1:
      AMDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
      gpus = [g.parent for g in pathlib.Path("/sys/devices/virtual/kfd/kfd/topology/nodes").glob("*/gpu_id") if is_usable_gpu(g)]
      gpus = sorted(gpus, key=lambda x: int(x.name.split('/')[-1]))
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('HIP_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
      AMDDevice.gpus = [gpus[x] for x in visible_devices] if visible_devices else gpus

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    if self.device_id >= len(AMDDevice.gpus): raise RuntimeError(f"No device found for {device}. Requesting more devices than the system has?")

    with open(f"{AMDDevice.gpus[self.device_id]}/gpu_id", "r") as f: self.gpu_id = int(f.read())
    with open(f"{AMDDevice.gpus[self.device_id]}/properties", "r") as f: self.properties = {line.split()[0]: int(line.split()[1]) for line in f}
    self.drm_fd = os.open(f"/dev/dri/renderD{self.properties['drm_render_minor']}", os.O_RDWR)
    target = int(self.properties['gfx_target_version'])
    self.arch = "gfx%d%x%x" % (target // 10000, (target // 100) % 100, target % 100)
    if target < 110000 or target >= 120000: raise RuntimeError(f"Unsupported arch: {self.arch}")

    kio.acquire_vm(AMDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    if AMDDevice.event_page is None:
      AMDDevice.signals_page = self._gpu_alloc(16 * 65536, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      AMDDevice.event_page = self._gpu_alloc(0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      AMDDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, AMDDevice.signals_page.size, 16)]
      kio.create_event(AMDDevice.kfd, event_page_offset=AMDDevice.event_page.handle)
    else:
      self._gpu_map(AMDDevice.signals_page)
      self._gpu_map(AMDDevice.event_page)

    # Scratch setup
    max_cu_id = self.properties['simd_count'] // self.properties['simd_per_cu'] - 1
    max_wave_id = self.properties['max_waves_per_simd'] * self.properties['simd_per_cu'] - 1
    self.max_private_segment_size = 4096
    wave_scratch_len = round_up(((max_wave_id + 1) * self.max_private_segment_size), 256) # gfx11 requires alignment of 256
    self.scratch_len = (max_cu_id + 1) * self.properties['max_slots_scratch_cu'] * wave_scratch_len
    self.scratch = self._gpu_alloc(self.scratch_len, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    engines = self.properties['array_count'] // self.properties['simd_arrays_per_engine']
    self.tmpring_size = (wave_scratch_len // 256) << 12 | (self.scratch_len // (wave_scratch_len * engines))

    self.compute_queue = self._alloc_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, 0x100000, ctx_save_restore_size=0x2C02000, eop_buffer_size=0x1000)
    self.sdma_queue = self._alloc_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x100000)

    super().__init__(device, AMDAllocator(self), AMDRenderer(), AMDCompiler(self.arch), functools.partial(AMDProgram, self),
                     AMDSignal, AMDComputeQueue, AMDCopyQueue, (AMDSignal(alloc_event=True), AMDSignal(alloc_event=True)))

  def _alloc_queue(self, queue_type, ring_size, ctx_save_restore_size=None, eop_buffer_size=None) -> AMDQueueDesc:
    gart = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    ring = self._gpu_alloc(ring_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    cwsr_ctx = self._gpu_alloc(ctx_save_restore_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) if ctx_save_restore_size else None
    eop_buffer = self._gpu_alloc(eop_buffer_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) if eop_buffer_size else None
    queue = kio.create_queue(AMDDevice.kfd, ring_base_address=ring.va_addr, ring_size=ring.size, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=eop_buffer.va_addr if eop_buffer else 0, eop_buffer_size=eop_buffer.size if eop_buffer else 0,
      ctx_save_restore_address=cwsr_ctx.va_addr if cwsr_ctx else 0, ctx_save_restore_size=cwsr_ctx.size if cwsr_ctx else 0,
      write_pointer_address=gart.va_addr, read_pointer_address=gart.va_addr + 8)

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = libc.mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, AMDDevice.kfd, self.doorbells_base)

    return AMDQueueDesc(ring=to_mv(ring.va_addr, ring_size).cast("I"),
                        read_ptr=to_mv(queue.read_pointer_address, 8).cast("Q"), write_ptr=to_mv(queue.write_pointer_address, 8).cast("Q"),
                        doorbell=to_mv(self.doorbells + queue.doorbell_offset - self.doorbells_base, 8).cast("Q"))

  def invalidate_caches(self):
    AMDComputeQueue().memory_barrier().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
    self.synchronize()
