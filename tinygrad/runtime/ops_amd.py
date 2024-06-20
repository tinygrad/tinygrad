from __future__ import annotations
from typing import Tuple, List, Any, cast
import os, fcntl, ctypes, ctypes.util, functools, re, pathlib, mmap, struct, errno, subprocess, time, array
from tinygrad.device import Compiled, Compiler, CompileError, BufferOptions, LRUAllocator
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up, DEBUG
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.runtime.driver.hip_comgr import compile_hip
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
import tinygrad.runtime.autogen.amd_gpu as amd_gpu
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

if getenv("MOCKGPU"):
  import extra.mockgpu.mockgpu  # noqa: F401
  libc.mmap = extra.mockgpu.mockgpu._mmap # type: ignore
  libc.munmap = extra.mockgpu.mockgpu._munmap # type: ignore

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

SIGNAL_SIZE, SIGNAL_COUNT = ctypes.sizeof(hsa.amd_signal_t), 16384
SIGNAL_VALUE_OFFSET = getattr(hsa.amd_signal_t, 'value').offset

regBIF_BX_PF1_GPU_HDP_FLUSH_REQ = 0x0106
regBIF_BX_PF1_GPU_HDP_FLUSH_DONE = 0x0107

# VGT_EVENT_TYPE in navi10_enum.h
CACHE_FLUSH_AND_INV_TS_EVENT = 0x14

WAIT_REG_MEM_FUNCTION_EQ = 3 # ==
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

COMPUTE_SHADER_EN, FORCE_START_AT_000, CS_W32_EN = (1 << 0), (1 << 2), (1 << 15)

def gfxreg(reg): return reg + 0x00001260 - amd_gpu.PACKET3_SET_SH_REG_START
def nbioreg(reg): return reg + 0x00000d20 # NBIO_BASE__INST0_SEG2
def data64_le(data): return (data & 0xFFFFFFFF, data >> 32)

class AMDCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    try: return compile_hip(src, self.arch)
    except RuntimeError as e: raise CompileError(e) from e

class HWQueue:
  def __init__(self): self.q, self.cmd_offsets = [], [0]
  def _mark_command_end(self):
    self.cmd_offsets.append(len(self.q))
    return self
  def _patch(self, off, data): self.q[off:off+len(data)] = array.array('I', data)
  def __len__(self): return len(self.cmd_offsets) - 1

class HWPM4Queue(HWQueue):
  def __init__(self):
    self.binded_device, self.ptr_to_dispatch_packet = None, {}
    super().__init__()

  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.synchronize()
      self.binded_device._gpu_free(self.hw_page)

  def _invalidate_cache(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0, #0x80000000,
               sz & 0xffffffff, (sz >> 32) & 0xff, addr & 0xffffffff, (addr >> 32) & 0xffffff, 0,
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)]

  def memory_barrier(self):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5), amd_gpu.WAIT_REG_MEM_MEM_SPACE(0) | amd_gpu.WAIT_REG_MEM_OPERATION(1) | \
      amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ) | amd_gpu.WAIT_REG_MEM_ENGINE(0), nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_REQ),
      nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_DONE), 0xffffffff, 0xffffffff, 0x20]
    self._invalidate_cache()
    return self._mark_command_end()

  def exec(self, prg, kernargs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), signal=None, signal_value=0):
    self._invalidate_cache()

    user_data = [*data64_le(kernargs)]
    if hasattr(prg, 'dispatch_packet_offset'):
      dp = hsa.hsa_kernel_dispatch_packet_t.from_address(dp_addr:=kernargs + prg.dispatch_packet_offset)
      dp.workgroup_size_x, dp.workgroup_size_y, dp.workgroup_size_z = local_size[0], local_size[1], local_size[2]
      dp.grid_size_x, dp.grid_size_y, dp.grid_size_z = global_size[0]*local_size[0], global_size[1]*local_size[1], global_size[2]*local_size[2]
      dp.group_segment_size, dp.private_segment_size, dp.kernarg_address = prg.group_segment_size, prg.private_segment_size, kernargs
      user_data = [*data64_le(dp_addr)] + user_data
      self.ptr_to_dispatch_packet[len(self)] = dp

    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 6), gfxreg(amd_gpu.regCOMPUTE_PGM_LO), *data64_le(prg.prog_addr >> 8),
               *data64_le(0), *data64_le(prg.device.scratch.va_addr >> 8)]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC1), prg.rsrc1, prg.rsrc2]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_TMPRING_SIZE), prg.device.tmpring_size]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_RESTART_X), 0, 0, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE0)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE2)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE4)] + [0xFFFFFFFF] * 4
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, len(user_data)), gfxreg(amd_gpu.regCOMPUTE_USER_DATA_0)] + user_data
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 8), gfxreg(amd_gpu.regCOMPUTE_START_X), 0, 0, 0, *local_size, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_RESOURCE_LIMITS), 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, 3), *global_size, CS_W32_EN | FORCE_START_AT_000 | COMPUTE_SHADER_EN]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_EVENT_WRITE, 0), amd_gpu.EVENT_TYPE(7) | amd_gpu.EVENT_INDEX(4)]

    if signal is not None: self.signal(signal, signal_value)
    return self._mark_command_end()

  def update_exec(self, cmd_idx, global_size, local_size):
    # Patch the exec cmd with new launch dims
    assert self.q[self.cmd_offsets[cmd_idx] + 60] == amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, 3), f"Command at index {cmd_idx} is not exec"
    self.q[self.cmd_offsets[cmd_idx] + 52 : self.cmd_offsets[cmd_idx] + 55] = array.array('I', local_size)
    self.q[self.cmd_offsets[cmd_idx] + 61 : self.cmd_offsets[cmd_idx] + 64] = array.array('I', global_size)

    if (dp:=self.ptr_to_dispatch_packet.get(cmd_idx)) is not None:
      dp.workgroup_size_x, dp.workgroup_size_y, dp.workgroup_size_z = local_size[0], local_size[1], local_size[2]
      dp.grid_size_x, dp.grid_size_y, dp.grid_size_z = global_size[0]*local_size[0], global_size[1]*local_size[1], global_size[2]*local_size[2]

  def wait(self, signal:hsa.amd_signal_t, value=0):
    addr = ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5),
      amd_gpu.WAIT_REG_MEM_MEM_SPACE(1) | amd_gpu.WAIT_REG_MEM_OPERATION(0) | amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | \
      amd_gpu.WAIT_REG_MEM_ENGINE(0), addr&0xFFFFFFFF, addr>>32, value, 0xffffffff, 4]
    return self._mark_command_end()

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
      address & 0xffffffff, address >> 32, value & 0xffffffff, value >> 32, cst]

  def timestamp(self, addr):
    self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=3, mem_int_sel=0, address=addr)
    return self._mark_command_end()

  def signal(self, signal:hsa.amd_signal_t, value=0):
    # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
    self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=1, mem_int_sel=2, address=ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET,
                      value=value, cache_flush=True)
    if signal.event_mailbox_ptr != 0:
      self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=1, mem_int_sel=2, address=signal.event_mailbox_ptr,
                        value=signal.event_id, cst=signal.event_id, cache_flush=True)
    return self._mark_command_end()

  def update_wait(self, cmd_idx, signal=None, value=None):
    assert self.q[self.cmd_offsets[cmd_idx]] == amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5), f"Command at index {cmd_idx} is not wait"
    if signal is not None: self._patch(self.cmd_offsets[cmd_idx] + 2, [*data64_le(ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET)])
    if value is not None: self.q[self.cmd_offsets[cmd_idx] + 4] = value
    return self

  def update_signal(self, cmd_idx, signal=None, value=None):
    assert self.q[self.cmd_offsets[cmd_idx]] == amd_gpu.PACKET3(amd_gpu.PACKET3_RELEASE_MEM, 6), f"Command at index {cmd_idx} is not signal"
    if signal is not None:
      self._patch(self.cmd_offsets[cmd_idx] + 3, [*data64_le(ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET)])
      if signal.event_mailbox_ptr != 0:
        self._patch(self.cmd_offsets[cmd_idx] + 8 + 3, [*data64_le(signal.event_mailbox_ptr), *data64_le(signal.event_id), signal.event_id])
    if value is not None: self._patch(self.cmd_offsets[cmd_idx] + 5, [*data64_le(value)])
    return self

  def bind(self, device: AMDDevice):
    self.binded_device = device
    self.hw_page = device._gpu_alloc(len(self.q) * 4, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    hw_view = to_mv(self.hw_page.va_addr, self.hw_page.size).cast("I")
    for i, value in enumerate(self.q): hw_view[i] = value

    self.indirect_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), self.hw_page.va_addr & 0xffffffff, self.hw_page.va_addr >> 32,
                         len(self.q) | amd_gpu.INDIRECT_BUFFER_VALID]
    self.q = hw_view # type: ignore

  def submit(self, device: AMDDevice):
    cmds = self.indirect_cmd if device == self.binded_device else self.q

    wptr = device.pm4_write_pointer[0]
    pm4_buffer_view = to_mv(device.pm4_ring.va_addr, device.pm4_ring.size).cast("I")
    for i, value in enumerate(cmds): pm4_buffer_view[(wptr+i)%(device.pm4_ring.size//4)] = value
    device.pm4_write_pointer[0] = wptr + len(cmds)
    device.pm4_doorbell[0] = wptr + len(cmds)
    return self

SDMA_MAX_COPY_SIZE = 0x400000
class HWCopyQueue(HWQueue):
  def __init__(self):
    self.internal_cmd_sizes = []
    super().__init__()

  def _q(self, arr):
    self.q += arr
    self.internal_cmd_sizes.append(len(arr))

  def copy(self, dest, src, copy_size):
    # Invalidate cache inv
    self._q([amd_gpu.SDMA_OP_GCR_REQ, 0, amd_gpu.SDMA_GCR_GLM_INV | amd_gpu.SDMA_GCR_GLK_INV | amd_gpu.SDMA_GCR_GLK_WB | amd_gpu.SDMA_GCR_GLV_INV | \
      amd_gpu.SDMA_GCR_GL1_INV | amd_gpu.SDMA_GCR_GL2_WB | amd_gpu.SDMA_GCR_GL2_INV, 0, 0])

    copied = 0
    copy_commands = (copy_size + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE
    for _ in range(copy_commands):
      step_copy_size = min(copy_size - copied, SDMA_MAX_COPY_SIZE)

      self._q([amd_gpu.SDMA_OP_COPY | amd_gpu.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(amd_gpu.SDMA_SUBOP_COPY_LINEAR),
        amd_gpu.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size - 1), 0, *data64_le(src + copied), *data64_le(dest + copied)])

      copied += step_copy_size

    # Invalidate cache wb
    self._q([amd_gpu.SDMA_OP_GCR_REQ, 0, amd_gpu.SDMA_GCR_GLK_WB | amd_gpu.SDMA_GCR_GL2_WB, 0, 0])

    return self._mark_command_end()

  def signal(self, signal: hsa.amd_signal_t, value=0):
    self._q([amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET), value])

    if signal.event_mailbox_ptr != 0:
      self._q([amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(signal.event_mailbox_ptr), signal.event_id])
      self._q([amd_gpu.SDMA_OP_TRAP, amd_gpu.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(signal.event_id)])

    return self._mark_command_end()

  def wait(self, signal: hsa.amd_signal_t, value=0):
    self._q([amd_gpu.SDMA_OP_POLL_REGMEM | amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) | \
      amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1), *data64_le(ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET), value, 0xffffffff,
      amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff)])

    return self._mark_command_end()

  def update_wait(self, cmd_idx, signal=None, value=None):
    assert self.q[self.cmd_offsets[cmd_idx]] & 0xf == amd_gpu.SDMA_OP_POLL_REGMEM, f"Command at index {cmd_idx} is not wait"
    if signal is not None: self._patch(self.cmd_offsets[cmd_idx] + 1, [*data64_le(ctypes.addressof(signal) + SIGNAL_VALUE_OFFSET)])
    if value is not None: self.q[self.cmd_offsets[cmd_idx] + 3] = value
    return self

  def submit(self, device:AMDDevice):
    read_ptr = device.sdma_read_pointer[0]
    if (device.sdma_doorbell_value-read_ptr) > device.sdma_ring.size: raise RuntimeError("SDMA queue overrun")

    sdma_buffer_view = to_mv(device.sdma_ring.va_addr, device.sdma_ring.size).cast("I")

    tail_blit_dword = 0
    for cmdsz in self.internal_cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= device.sdma_ring.size - device.sdma_doorbell_value % device.sdma_ring.size: break
      tail_blit_dword += cmdsz

    start_idx = (device.sdma_doorbell_value % device.sdma_ring.size) // 4
    sdma_buffer_view[start_idx : start_idx + tail_blit_dword] = array.array('I', self.q[:tail_blit_dword])
    device.sdma_doorbell_value += tail_blit_dword * 4

    if (rem_packet_cnt := len(self.q) - tail_blit_dword) > 0:
      zero_fill = device.sdma_ring.size - device.sdma_doorbell_value % device.sdma_ring.size
      ctypes.memset(device.sdma_ring.va_addr + (device.sdma_doorbell_value % device.sdma_ring.size), 0, zero_fill)
      device.sdma_doorbell_value += zero_fill

      sdma_buffer_view[0:rem_packet_cnt] = array.array('I', self.q[tail_blit_dword:])
      device.sdma_doorbell_value += rem_packet_cnt * 4

    device.sdma_write_pointer[0] = device.sdma_doorbell_value
    device.sdma_doorbell[0] = device.sdma_doorbell_value
    return self

SHT_PROGBITS, SHF_ALLOC = 0x1, 0x2
class AMDProgram:
  def __init__(self, device:AMDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]

    lib_gpu_size = round_up(max(sh[5]+sh[3] for sh in sections if sh[1] == SHT_PROGBITS), 0x1000)
    self.lib_gpu = self.device._gpu_alloc(lib_gpu_size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=True)
    lib_gpu_view = to_mv(self.lib_gpu.va_addr, lib_gpu_size)

    for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, _, _ in sections:
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC: lib_gpu_view[sh_addr:sh_addr+sh_size] = self.lib[sh_offset:sh_offset+sh_size]

    entry_point = min(sh[3] for sh in sections if sh[1] == SHT_PROGBITS and sh[2] & SHF_ALLOC)
    self.group_segment_size = lib_gpu_view.cast("I")[entry_point//4]
    self.private_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 1]
    self.kernargs_segment_size = lib_gpu_view.cast("I")[entry_point//4 + 2]
    self.kernargs_alloc_size = self.kernargs_segment_size
    self.kernargs_offset = 0

    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if lds_size > (self.device.properties['lds_size_in_kb'] * 1024) // 512: raise RuntimeError("Too many resources requsted: group_segment_size")
    if self.private_segment_size > self.device.max_private_segment_size: raise RuntimeError("Too many resources requsted: private_segment_size")

    code = hsa.amd_kernel_code_t.from_address(self.lib_gpu.va_addr + entry_point) # NOTE: this is wrong, it's not this object
    self.rsrc1 = code.compute_pgm_rsrc1
    self.rsrc2 = code.compute_pgm_rsrc2 | (lds_size << 15)

    if code.kernel_code_properties & 0x2 == 0x2: # ENABLE_SGPR_DISPATCH_PTR
      # Allocate space for the dispatch packet in the kernargs to pass it to the GPU.
      self.dispatch_packet_offset = self.kernargs_alloc_size
      self.kernargs_alloc_size += ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)

    assert code.kernel_code_properties & 0x400 == 0x400 # ENABLE_WAVEFRONT_SIZE32
    assert code.workitem_private_segment_byte_size == 0
    assert code.max_scratch_backing_memory_byte_size == 0
    assert code.kernel_code_prefetch_byte_size == 0

    self.prog_addr = self.lib_gpu.va_addr + entry_point + code.kernel_code_entry_byte_offset

    HWPM4Queue().memory_barrier().submit(self.device)

  # NOTE: no programs are ever freed
  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device._gpu_free(self.lib_gpu)

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if self.device.kernargs_ptr + self.kernargs_alloc_size > (self.device.kernargs.va_addr + self.device.kernargs.size):
      self.device.kernargs_ptr = self.device.kernargs.va_addr

    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
        raise RuntimeError(f"AMDProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")

    args_st = self.args_struct_t.from_address(self.device.kernargs_ptr)
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].va_addr)
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

    q = HWPM4Queue()
    q.wait(self.device.timeline_signal, self.device.timeline_value - 1).memory_barrier()
    if wait: q.timestamp(ctypes.addressof(self.device.timeline_signal) + getattr(hsa.amd_signal_t, 'start_ts').offset)
    q.exec(self, self.device.kernargs_ptr, global_size, local_size)
    if wait: q.timestamp(ctypes.addressof(self.device.timeline_signal) + getattr(hsa.amd_signal_t, 'end_ts').offset)
    q.signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
    self.device.timeline_value += 1
    self.device.kernargs_ptr += self.kernargs_alloc_size

    if wait:
      self.device._wait_signal(self.device.timeline_signal, self.device.timeline_value - 1)
      return (self.device.timeline_signal.end_ts - self.device.timeline_signal.start_ts) / 1e8

class AMDAllocator(LRUAllocator):
  def __init__(self, device:AMDDevice):
    self.device = device
    # NOTE: KFD_IOC_ALLOC_MEM_FLAGS_GTT doesn't work here for readinto
    self.b = [self.device._gpu_alloc(SDMA_MAX_COPY_SIZE, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True) for _ in range(16)]
    self.b_timeline = [0] * len(self.b)
    self.b_next = 0
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    try:
      if options.host: return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, public=True)
      return self.device._gpu_alloc(size, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM, public=options.cpu_access)
    except OSError as e:
      if e.errno == errno.ENOMEM: raise MemoryError("Cannot allocate memory") from e
      else: raise

  def _free(self, opaque, options:BufferOptions): self.device._gpu_free(opaque)

  def copyin(self, dest, src: memoryview):
    for i in range(0, src.nbytes, self.b[0].size):
      self.b_next = (self.b_next + 1) % len(self.b)
      AMDDevice._wait_signal(self.device.timeline_signal, self.b_timeline[self.b_next])
      ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].size, src.nbytes-i))
      HWCopyQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                   .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.b_timeline[self.b_next] = self.device.timeline_value
      self.device.timeline_value += 1

  def copy_from_disk(self, dest, src, size, disk):
    def _get_temp_buf():
      if self.b_timeline[cur_b_next:=(self.b_next + 1) % len(self.b)] <= self.device.timeline_signal.value:
        self.b_timeline[cur_b_next], self.b_next = (1 << 64), cur_b_next
        return (self.b[cur_b_next].va_addr, cur_b_next)
      return None

    for (batch_info, dst_off, src_off, copy_size) in disk.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=SDMA_MAX_COPY_SIZE):
      HWCopyQueue().copy(dest.va_addr + dst_off, batch_info[0] + src_off, copy_size) \
                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.b_timeline[batch_info[1]] = self.device.timeline_value
      self.device.timeline_value += 1

  def copyout(self, dest:memoryview, src):
    self.device.synchronize()
    for i in range(0, dest.nbytes, self.b[0].size):
      HWCopyQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                   .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i)) \
                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      AMDDevice._wait_signal(self.device.timeline_signal, self.device.timeline_value)
      self.device.timeline_value += 1

      ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest, src, sz:int, src_dev:AMDDevice, dest_dev:AMDDevice):
    src_dev._gpu_map(dest)
    HWCopyQueue().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                 .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                 .copy(dest.va_addr, src.va_addr, sz) \
                 .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
    HWPM4Queue().wait(src_dev.timeline_signal, src_dev.timeline_value).submit(dest_dev)
    src_dev.timeline_value += 1

  def offset(self, buf, size:int, offset:int): return type(buf)(va_addr=buf.va_addr + offset, size=size)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class AMDDevice(Compiled):
  kfd:int = -1
  event_page:Any = None  # TODO: fix types in kfd, Optional[kfd.struct_kfd_ioctl_alloc_memory_of_gpu_args]
  signals_page:Any = None
  signals_pool:List[hsa.amd_signal_t] = []
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
  def _set_signal(self, sig, value): sig.value = value

  @classmethod
  def _get_signal(self, value=0, sync_event=None) -> hsa.amd_signal_t:
    self._set_signal(ret := self.signals_pool.pop(), value)
    if sync_event is not None:
      ret.event_mailbox_ptr = AMDDevice.event_page.va_addr + sync_event.event_slot_index*8
      ret.event_id = sync_event.event_id
    else: ret.event_mailbox_ptr = ret.event_id = 0
    return ret

  @classmethod
  def _wait_signal(self, signal:hsa.amd_signal_t, value=0, timeout=10000):
    assert signal.event_id != 0, "can't wait on this signal"
    evt_arr = (kfd.struct_kfd_event_data)(event_id=signal.event_id)

    # Wait active for 5s, then going to sleep.
    start_time = time.time() * 1000
    while (time_spent:=time.time() * 1000 - start_time) < timeout:
      if signal.value >= value: return
      if time_spent > 5000: kio.wait_events(AMDDevice.kfd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=1, timeout=1000)
    raise RuntimeError(f"wait_signal: not set to {value}, but {signal.value}, {timeout} ms TIMEOUT!")

  def __init__(self, device:str=""):
    if AMDDevice.kfd == -1:
      AMDDevice.kfd = os.open("/dev/kfd", os.O_RDWR)
      AMDDevice.gpus = [g.parent for g in pathlib.Path("/sys/devices/virtual/kfd/kfd/topology/nodes").glob("*/gpu_id") if is_usable_gpu(g)]
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    with open(f"{AMDDevice.gpus[self.device_id]}/gpu_id", "r") as f: self.gpu_id = int(f.read())
    with open(f"{AMDDevice.gpus[self.device_id]}/properties", "r") as f: self.properties = {line.split()[0]: int(line.split()[1]) for line in f}
    self.drm_fd = os.open(f"/dev/dri/renderD{self.properties['drm_render_minor']}", os.O_RDWR)
    target = int(self.properties['gfx_target_version'])
    self.arch = "gfx%d%x%x" % (target // 10000, (target // 100) % 100, target % 100)
    kio.acquire_vm(AMDDevice.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    if AMDDevice.event_page is None:
      AMDDevice.signals_page = self._gpu_alloc(SIGNAL_SIZE*SIGNAL_COUNT, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      AMDDevice.event_page = self._gpu_alloc(0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
      for off in range(0, AMDDevice.signals_page.size, SIGNAL_SIZE):
        AMDDevice.signals_pool.append(hsa.amd_signal_t.from_address(AMDDevice.signals_page.va_addr + off))
      sync_event = kio.create_event(AMDDevice.kfd, event_page_offset=AMDDevice.event_page.handle, auto_reset=1)
    else:
      self._gpu_map(AMDDevice.signals_page)
      self._gpu_map(AMDDevice.event_page)
      sync_event = kio.create_event(AMDDevice.kfd, auto_reset=1)

    self.timeline_value: int = 1
    self.timeline_signal = AMDDevice._get_signal(sync_event=sync_event)
    self._shadow_timeline_signal = AMDDevice._get_signal(sync_event=kio.create_event(AMDDevice.kfd, auto_reset=1))

    self.kernargs = self._gpu_alloc(0x1000000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.kernargs_ptr = self.kernargs.va_addr

    # scratch setup
    max_cu_id = self.properties['simd_count'] // self.properties['simd_per_cu'] - 1
    max_wave_id = self.properties['max_waves_per_simd'] * self.properties['simd_per_cu'] - 1
    self.max_private_segment_size = 4096
    wave_scratch_len = round_up(((max_wave_id + 1) * self.max_private_segment_size), 256) # gfx11 requires alignment of 256
    self.scratch_len = (max_cu_id + 1) * self.properties['max_slots_scratch_cu'] * wave_scratch_len
    self.scratch = self._gpu_alloc(self.scratch_len, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    engines = self.properties['array_count'] // self.properties['simd_arrays_per_engine']
    self.tmpring_size = (wave_scratch_len // 256) << 12 | (self.scratch_len // (wave_scratch_len * engines))

    # SDMA Queue
    self.sdma_gart = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.sdma_ring = self._gpu_alloc(0x100000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.sdma_queue = kio.create_queue(AMDDevice.kfd, ring_base_address=self.sdma_ring.va_addr, ring_size=self.sdma_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_SDMA, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      write_pointer_address=self.sdma_gart.va_addr, read_pointer_address=self.sdma_gart.va_addr+8)

    # doorbell page
    self.doorbells_base = self.sdma_queue.doorbell_offset & (~0x1fff)  # doorbell is two pages
    self.doorbells = libc.mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, AMDDevice.kfd, self.doorbells_base)

    self.sdma_read_pointer = to_mv(self.sdma_queue.read_pointer_address, 8).cast("Q")
    self.sdma_write_pointer = to_mv(self.sdma_queue.write_pointer_address, 8).cast("Q")
    self.sdma_doorbell = to_mv(self.doorbells + self.sdma_queue.doorbell_offset - self.doorbells_base, 8).cast("Q")
    self.sdma_doorbell_value = 0

    # PM4 Queue
    self.pm4_ctx_save_restore_address = self._gpu_alloc(0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.pm4_eop_buffer = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
    self.pm4_gart = self._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.pm4_ring = self._gpu_alloc(0x100000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    self.pm4_queue = kio.create_queue(AMDDevice.kfd, ring_base_address=self.pm4_ring.va_addr, ring_size=self.pm4_ring.size, gpu_id=self.gpu_id,
      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=self.pm4_eop_buffer.va_addr, eop_buffer_size=self.pm4_eop_buffer.size,
      ctx_save_restore_address=self.pm4_ctx_save_restore_address.va_addr, ctx_save_restore_size=self.pm4_ctx_save_restore_address.size,
      ctl_stack_size = 0xa000, write_pointer_address=self.pm4_gart.va_addr, read_pointer_address=self.pm4_gart.va_addr+8)

    self.pm4_read_pointer = to_mv(self.pm4_queue.read_pointer_address, 8).cast("Q")
    self.pm4_write_pointer = to_mv(self.pm4_queue.write_pointer_address, 8).cast("Q")
    self.pm4_doorbell = to_mv(self.doorbells + self.pm4_queue.doorbell_offset - self.doorbells_base, 8).cast("Q")

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, AMDAllocator(self), AMDRenderer(), AMDCompiler(self.arch),
                     functools.partial(AMDProgram, self), functools.partial(HCQGraph, AMDDevice, HWPM4Queue, HWCopyQueue))

  def synchronize(self):
    AMDDevice._wait_signal(self.timeline_signal, self.timeline_value - 1)

    # reset kernargs
    self.kernargs_ptr = self.kernargs.va_addr
    if self.timeline_value > (1 << 31):
      self.timeline_signal, self._shadow_timeline_signal = self._shadow_timeline_signal, self.timeline_signal
      self.timeline_signal.value, self.timeline_value = 0, 1
      cast(AMDAllocator, self.allocator).b_timeline = [0] * len(cast(AMDAllocator, self.allocator).b)
