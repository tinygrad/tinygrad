from __future__ import annotations
from typing import cast
import os, ctypes, struct, hashlib, functools, importlib, mmap, errno, array, contextlib, sys, weakref, itertools, collections, atexit
assert sys.platform != 'win32'
from dataclasses import dataclass
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, HCQ2Buffer, HWQueue, HCQSignal, FileIOInterface
from tinygrad.runtime.support.hcq2 import MMIOInterface, BumpAllocator, hcq_filter_visible_devices, hcq_profile
from tinygrad.uop.ops import sint, UOp
from tinygrad.device import Compiled, BufferSpec, Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, round_up, data64_le, DEBUG, PROFILE, ProfileEvent, lo32, hi32, colored, prod, ContextVar, TracingKey
from tinygrad.helpers import VIZ, ceildiv, unwrap, pluralize
from tinygrad.renderer.cstyle import HIPRenderer, HIPCCRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import kfd, hsa, sqtt, amdgpu_kd, amdgpu_drm
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import AMDReg, AMDIP, import_module, import_soc, import_pmc
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta, USBPCIDevice, MAP_FIXED, MAP_NORESERVE
from tinygrad.runtime.support.usb import USB3
from tinygrad.runtime.support.memory import AddrSpace
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

SQTT = ContextVar("SQTT", abs(VIZ.value)>=2)
SQTT_ITRACE_SE_MASK, SQTT_LIMIT_SE, SQTT_SIMD_SEL, SQTT_TOKEN_EXCLUDE = \
  ContextVar("SQTT_ITRACE_SE_MASK", 0b11), ContextVar("SQTT_LIMIT_SE", 0), ContextVar("SQTT_SIMD_SEL", 0), ContextVar("SQTT_TOKEN_EXCLUDE", 0)
PMC = ContextVar("PMC", abs(VIZ.value)>=2)
EVENT_INDEX_PARTIAL_FLUSH = 4 # based on a comment in nvd.h
WAIT_REG_MEM_FUNCTION_EQ  = 3 # ==
WAIT_REG_MEM_FUNCTION_NEQ = 4 # !=
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

# new queue submition
from tinygrad.runtime.support.hcq2 import HCQ2LowerCtx
from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import Ops, UPat, PatternMatcher

def pm4_pkt3(ctx:HCQ2LowerCtx, cmd, *vals): ctx.append(ctx.dev.pm4.PACKET3(cmd, len(vals) - 1), *vals)

def pm4_wreg(ctx:HCQ2LowerCtx, reg, *args):
  pm4 = ctx.dev.pm4
  if pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < pm4.PACKET3_SET_SH_REG_END:
    pm4_pkt3(ctx, pm4.PACKET3_SET_SH_REG, reg.addr[0] - pm4.PACKET3_SET_SH_REG_START, *args)
  elif pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
    pm4_pkt3(ctx, pm4.PACKET3_SET_UCONFIG_REG, reg.addr[0] - pm4.PACKET3_SET_UCONFIG_REG_START, *args)
  else: raise RuntimeError(f'Cannot set {reg.name}')

def pm4_wait(ctx:HCQ2LowerCtx, x:UOp):
  pm4 = ctx.dev.pm4
  wrm = pm4.WAIT_REG_MEM_MEM_SPACE(1) | pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | pm4.WAIT_REG_MEM_ENGINE(0)
  pm4_pkt3(ctx, pm4.PACKET3_WAIT_REG_MEM, wrm, *data64_le(ctx.addr(x.src[0])), x.src[1], 0xffffffff, 4)

def pm4_barrier(ctx:HCQ2LowerCtx, x:UOp):
  pm4, nbio = ctx.dev.pm4, ctx.dev.nbio
  pf = '' if nbio.version[0] == 2 else '0' if nbio.version[:2] != (7, 11) else '1'
  req_reg = getattr(nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr[0]
  done_reg = getattr(nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr[0]
  wrm = pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | pm4.WAIT_REG_MEM_ENGINE(0) | pm4.WAIT_REG_MEM_OPERATION(1)
  pm4_pkt3(ctx, pm4.PACKET3_WAIT_REG_MEM, wrm, req_reg, done_reg, 0xffffffff, 0xffffffff, 4)
  if ctx.dev.target[0] != 9:
    cache_flags = pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(1)
    pm4_pkt3(ctx, pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le((1 << 64) - 1), *data64_le(0), 0, cache_flags)
  else:
    cp_coher = pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(1) \
             | pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(1) \
             | pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(1) \
             | pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(1) \
             | pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(1)
    pm4_pkt3(ctx, pm4.PACKET3_ACQUIRE_MEM, cp_coher, *data64_le((1 << 64) - 1), *data64_le(0), 0x0000000A)

def pm4_program(ctx:HCQ2LowerCtx, x:UOp):
  data, info = x.arg
  lib_gpu, args = x.src
  prog_addr = ctx.addr(lib_gpu) + data.entry_point_offset
  pm4, gc, soc = ctx.dev.pm4, ctx.dev.gc, ctx.dev.soc

  if ctx.dev.target[0] != 9:
    cache_flags = pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(1) \
                | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(1) | pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(1)
    pm4_pkt3(ctx, pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le((1 << 64) - 1), *data64_le(0), 0, cache_flags)
  else:
    cp_coher = pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(1) | pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(1)
    pm4_pkt3(ctx, pm4.PACKET3_ACQUIRE_MEM, cp_coher, *data64_le((1 << 64) - 1), *data64_le(0), 0x0000000A)

  user_regs = []
  if data.enable_private_segment_sgpr:
    scratch_hilo = data64_le(ctx.dev.scratch.va_addr)
    user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000]
  if data.enable_dispatch_ptr: user_regs += [*data64_le(ctx.addr(args) + data.kernargs_segment_size)]
  user_regs += [*data64_le(ctx.addr(args))]

  pm4_wreg(ctx, gc.regCOMPUTE_PGM_LO, *data64_le(prog_addr >> 8))
  pm4_wreg(ctx, gc.regCOMPUTE_PGM_RSRC1, data.rsrc1, data.rsrc2)
  pm4_wreg(ctx, gc.regCOMPUTE_PGM_RSRC3, data.rsrc3)
  pm4_wreg(ctx, gc.regCOMPUTE_TMPRING_SIZE, ctx.dev.tmpring_size)

  for xcc_id in range(ctx.dev.xccs):
    scratch_base = ctx.dev.scratch.va_addr + (ctx.dev.scratch.size // ctx.dev.xccs * xcc_id)
    pm4_wreg(ctx, gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, *data64_le(scratch_base >> 8))

  pm4_wreg(ctx, gc.regCOMPUTE_RESTART_X, 0, 0, 0)
  pm4_wreg(ctx, gc.regCOMPUTE_USER_DATA_0, *user_regs)
  pm4_wreg(ctx, gc.regCOMPUTE_RESOURCE_LIMITS, gc.regCOMPUTE_RESOURCE_LIMITS.encode(waves_per_sh=getenv("WAVES_PER_SH")))
  pm4_wreg(ctx, gc.regCOMPUTE_START_X, 0, 0, 0, *(info.local_size or (1, 1, 1)), 0, 0)

  dispatch_init = gc.regCOMPUTE_DISPATCH_INITIATOR.encode(**({'cs_w32_en': int(data.wave32)} if ctx.dev.target[0] != 9 else {}),
                                                          force_start_at_000=1, compute_shader_en=1)
  pm4_pkt3(ctx, pm4.PACKET3_DISPATCH_DIRECT, *info.global_size, dispatch_init)
  pm4_pkt3(ctx, pm4.PACKET3_EVENT_WRITE, pm4.EVENT_TYPE(soc.CS_PARTIAL_FLUSH) | pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))

def pm4_store(ctx:HCQ2LowerCtx, x:UOp):
  pm4 = ctx.dev.pm4
  if ctx.dev.target[0] != 9:
    cache_flags = pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | pm4.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                | pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                | pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | pm4.PACKET3_RELEASE_MEM_GCR_GL2_WB | pm4.PACKET3_RELEASE_MEM_GCR_SEQ
    event_dw = pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
             | pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(pm4.event_index__mec_release_mem__end_of_pipe)
    memsel_dw = pm4.PACKET3_RELEASE_MEM_DATA_SEL(pm4.data_sel__mec_release_mem__send_32_bit_low) \
              | pm4.PACKET3_RELEASE_MEM_INT_SEL(pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm) \
              | pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
  else:
    cache_flags = pm4.EOP_TC_WB_ACTION_EN | pm4.EOP_TC_NC_ACTION_EN
    event_dw = pm4.EVENT_TYPE(pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | pm4.EVENT_INDEX(pm4.event_index__mec_release_mem__end_of_pipe)
    memsel_dw = pm4.DATA_SEL(pm4.data_sel__mec_release_mem__send_32_bit_low) \
              | pm4.INT_SEL(pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm)
  pm4_pkt3(ctx, pm4.PACKET3_RELEASE_MEM, event_dw | cache_flags, memsel_dw, *data64_le(ctx.addr(x.src[0])), x.src[1], 0, 0)

amd_pm4_encode = PatternMatcher([
  (UPat(Ops.WAIT,    name="x"), pm4_wait),
  (UPat(Ops.BARRIER, name="x"), pm4_barrier),
  (UPat(Ops.PROGRAM, name="x"), pm4_program),
  (UPat(Ops.STORE,   name="x"), pm4_store),
])

def pm4_submit(ctx:HCQ2LowerCtx, blob:UOp) -> UOp:
  q = ctx.dev.compute_queue
  ring, wptr, doorbell, put_ptr = (ctx.param(b) for b in (q.ring, q.write_ptr, q.doorbell, q.put_value))
  size, ring_dwords = UOp.const(dtypes.uint32, blob.dtype.size), q.ring.size

  put = put_ptr[0]
  i = UOp.range(size, 0, dtype=dtypes.int)
  next_put = put + size.cast(put.dtype)
  ring_idx = ((put + i.cast(put.dtype)) % ring_dwords).cast(dtypes.int)

  copy_to_ring = ring[ring_idx].store(blob[i]).end(i)
  bump_put_ptr = put_ptr[0].store(next_put)
  bump_wptr = wptr[0].store(next_put)
  flush = UOp.barrier(copy_to_ring, bump_put_ptr, bump_wptr)
  return doorbell.after(flush)[0].store(next_put)

amd_hcq_routines = PatternMatcher([
  (UPat(Ops.PARAM, name="blob"), pm4_submit),
])



class AMDSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 100})

  def _sleep(self, time_spent_since_last_sleep_ms:int):
    # Reasonable to sleep for long workloads (which take more than 200ms) and only timeline signals.
    if time_spent_since_last_sleep_ms > 200 and self.owner is not None: self.owner.iface.sleep(200)

class AMDComputeQueue(HWQueue):
  def __init__(self, dev:AMD2Device):
    self.dev, self.soc, self.pm4, self.gc, self.nbio = dev, dev.soc, dev.pm4, dev.gc, dev.nbio
    super().__init__()

  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True, uncached=True))

  def pkt3(self, cmd, *vals): self.q(self.pm4.PACKET3(cmd, len(vals) - 1), *vals)

  def wreg(self, reg:AMDReg, *args:sint, **kwargs:int):
    if bool(args) == bool(kwargs): raise RuntimeError('One (and only one) of *args or **kwargs must be specified')
    if self.pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_SH_REG_END:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_SH_REG, self.pm4.PACKET3_SET_SH_REG_START
    elif self.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_UCONFIG_REG, self.pm4.PACKET3_SET_UCONFIG_REG_START
    else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr[0]}) via pm4 packet')
    self.pkt3(set_packet, reg.addr[0] - set_packet_start, *(args or (reg.encode(**kwargs),)))

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg=None, reg_done=0, op=WAIT_REG_MEM_FUNCTION_GEQ):
    wrm_info_dw = self.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | self.pm4.WAIT_REG_MEM_OPERATION(int(mem is None and reg_done > 0)) \
                | self.pm4.WAIT_REG_MEM_FUNCTION(op) | self.pm4.WAIT_REG_MEM_ENGINE(0)

    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, wrm_info_dw, *(data64_le(mem) if mem is not None else (reg, reg_done)), value, mask, 4)
    return self

  def acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    if self.dev.target[0] != 9:
      cache_flags_dw = self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)

      self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)
    else:
      cp_coher_cntl = self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(gli) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(glk) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(gl2) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(gl1) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(gl2)
      self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, cp_coher_cntl, *data64_le(sz), *data64_le(addr), 0x0000000A)
    return self

  def release_mem(self, address=0x0, value=0, data_sel=0, int_sel=2, ctxid=0, cache_flush=False):
    if self.dev.target[0] != 9:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_WB | self.pm4.PACKET3_RELEASE_MEM_GCR_SEQ)

      event_dw = self.pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
               | self.pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)

      memsel_dw = self.pm4.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | self.pm4.PACKET3_RELEASE_MEM_INT_SEL(int_sel) \
                | self.pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
    else:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.EOP_TC_WB_ACTION_EN | self.pm4.EOP_TC_NC_ACTION_EN)

      event_dw = self.pm4.EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | self.pm4.EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)

      memsel_dw = self.pm4.DATA_SEL(data_sel) | self.pm4.INT_SEL(int_sel)

      ctxid = 0

    self.pkt3(self.pm4.PACKET3_RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, *data64_le(address), *data64_le(value), ctxid)
    return self

  def memory_barrier(self):
    pf = '' if self.nbio.version[0] == 2 else '0' if self.nbio.version[:2] != (7, 11) else '1'
    self.wait_reg_mem(reg=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr[0],
                      reg_done=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr[0], value=0xffffffff)
    return self.acquire_mem()

  def wait(self, signal:AMDSignal, value:sint=0): return self.wait_reg_mem(mem=signal.value_addr, value=value, mask=0xffffffff)

  def timestamp(self, signal:AMDSignal):
    self.release_mem(cache_flush=False) # ensure all prior writes are done
    self.release_mem(signal.timestamp_addr, 0, self.pm4.data_sel__mec_release_mem__send_gpu_clock_counter, self.pm4.int_sel__mec_release_mem__none)
    self.acquire_mem() # ensure timestamp is written

  def write(self, b:HCQ2Buffer, val:sint, b64:bool=False):
    data_sel = self.pm4.data_sel__mec_release_mem__send_64_bit_data if b64 else self.pm4.data_sel__mec_release_mem__send_32_bit_low
    return self.release_mem(b.va_addr, val, data_sel, self.pm4.int_sel__mec_release_mem__none)

  def poll_bit(self, b:HCQ2Buffer, val:sint, mask:int): return self.wait_reg_mem(val, mask=mask, mem=b.va_addr, op=WAIT_REG_MEM_FUNCTION_EQ)

  def signal(self, signal:AMDSignal, value:sint=0):
    # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
    self.release_mem(signal.value_addr, value, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                      self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

    if (dev:=signal.owner) is not None and signal.is_timeline and not dev.is_am():
      self.release_mem(dev.queue_event_mailbox_ptr, dev.queue_event.event_id, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                        self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, ctxid=dev.queue_event.event_id)
    return self

  def bind(self, dev:AMD2Device):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    self.indirect_cmd = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), *data64_le(self.hw_page.va_addr),
                         len(self._q) | self.pm4.INDIRECT_BUFFER_VALID]
    self._q = hw_view
    return self

  def _submit(self, dev:AMD2Device):
    cmds = self.indirect_cmd if dev == self.binded_device else self._q
    q = dev.compute_queue
    # WORKAROUND: PACKET3_PRED_EXEC doesn't work in rings, only in IBs, create a fake IB inside a ring to work around that
    if self.dev.xccs > 1 and dev != self.binded_device:
      ib_end = ((q.put + 5) % q.ring.size) + len(cmds)
      ib_pad = q.ring.size - (ib_end - len(cmds)) if ib_end > q.ring.size else 0
      ib_ptr = q.ring._buf.va_addr + ((q.put + 5 + ib_pad) % q.ring.size) * 4
      cmds = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), *data64_le(ib_ptr), len(cmds) | self.pm4.INDIRECT_BUFFER_VALID,
              self.pm4.PACKET3(self.pm4.PACKET3_NOP, ib_pad + len(cmds) - 1), *((0,) * ib_pad), *cmds]

    ring_mv = q.ring_mv
    for i, value in enumerate(cmds): ring_mv[(q.put + i) % q.ring.size] = value

    q.put += len(cmds)
    q.signal_doorbell(dev)

class AMDCopyQueue(HWQueue):
  def __init__(self, dev, max_copy_size=0x40000000, queue_idx=0):
    self.dev, self.sdma, self.internal_cmd_sizes, self.max_copy_size, self.queue_idx = dev, dev.sdma, [], max_copy_size, queue_idx
    super().__init__()

  def q(self, *arr):
    super().q(*arr)
    self.internal_cmd_sizes.append(len(arr))

  def copy(self, dest:HCQ2Buffer, src:HCQ2Buffer, copy_size:int):
    copied, copy_commands = 0, (copy_size + self.max_copy_size - 1) // self.max_copy_size

    for _ in range(copy_commands):
      step_copy_size = min(copy_size - copied, self.max_copy_size)

      self.q(self.sdma.SDMA_OP_COPY | self.sdma.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_COPY_LINEAR),
        self.sdma.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size - 1), 0, *data64_le(src.va_addr + copied), *data64_le(dest.va_addr + copied))

      copied += step_copy_size
    return self

  def signal(self, signal:AMDSignal, value:sint=0):
    fence_flags = self.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if self.dev.target[0] != 9 else 0
    self.q(self.sdma.SDMA_OP_FENCE | fence_flags, *data64_le(signal.value_addr), value)

    if (dev:=signal.owner) is not None and signal.is_timeline and not dev.is_am():
      self.q(self.sdma.SDMA_OP_FENCE | fence_flags, *data64_le(dev.queue_event_mailbox_ptr), dev.queue_event.event_id)
      self.q(self.sdma.SDMA_OP_TRAP, self.sdma.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(dev.queue_event.event_id))
    elif dev is not None and dev.is_am(): self.q(self.sdma.SDMA_OP_TRAP, 0)

    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.q(self.sdma.SDMA_OP_POLL_REGMEM | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) | \
           self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1), *data64_le(signal.value_addr), value, 0xffffffff,
           self.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | self.sdma.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))
    return self

  def timestamp(self, signal:AMDSignal):
    self.q(self.sdma.SDMA_OP_TIMESTAMP | self.sdma.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL),
           *data64_le(signal.timestamp_addr))
    return self

  def write(self, b:HCQ2Buffer, val:sint, b64:bool=False):
    self.q(self.sdma.SDMA_OP_WRITE, *data64_le(b.va_addr), 1 if b64 else 0, lo32(val), *([hi32(val)] if b64 else []))
    return self

  def bind(self, dev:AMD2Device):
    if not getenv("AMD_SDMA_BIND", 0) or not dev.is_am(): return

    self.binded_device = dev
    self.hw_page = dev.allocator.alloc((qsz:=round_up(len(self._q), 8)) * 4, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i in range(qsz): hw_view[i] = self._q[i] if i < len(self._q) else 0

    self.indirect_cmd = [self.sdma.SDMA_OP_INDIRECT | self.sdma.SDMA_PKT_INDIRECT_HEADER_VMID(0), *data64_le(self.hw_page.va_addr), qsz,
                         *data64_le(0)]
    self._q, self.cmd_sizes = hw_view, [len(self.indirect_cmd)]

  def _submit(self, dev:AMD2Device):
    q = dev.sdma_queue(self.queue_idx)
    nbytes = q.ring.nbytes
    if self.binded_device == dev:
      # An IB packet must end on a 8 DW boundary.
      add = (8 - (((q.put % 32) // 4) + len(self.indirect_cmd) % 8)) % 8
      cmds, cmd_sizes = ([0] * add) + self.indirect_cmd, [len(self.indirect_cmd) + add]

      if len(cmds) * 4 >= (nbytes - q.put % nbytes):
        cmds, cmd_sizes = [0, 0] + self.indirect_cmd, [8]
    else: cmds, cmd_sizes = self._q, self.internal_cmd_sizes

    tail_blit_dword = 0
    for cmdsz in cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= nbytes - q.put % nbytes: break
      tail_blit_dword += cmdsz

    # Force align of submits to hit our usb layer write cache.
    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0 and dev.is_usb(): tail_blit_dword = 0

    # USB devices run in single-step mode, so they can't overrun the queue.
    total_bytes = (tail_blit_dword * 4 if rem_packet_cnt == 0 else -q.put % nbytes) + rem_packet_cnt * 4
    assert total_bytes < nbytes, "SDMA queue overrun"
    while not dev.is_usb() and q.put + total_bytes - q.rptr_mv[0] > nbytes: pass

    ring_mv = q.ring_mv
    start_idx = (q.put % nbytes) // 4
    ring_mv[start_idx : start_idx + tail_blit_dword] = array.array('I', cmds[:tail_blit_dword])
    q.put += tail_blit_dword * 4

    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0:
      zero_fill = nbytes - q.put % nbytes
      ring_mv.view(q.put % nbytes, zero_fill, fmt='B')[:] = bytes(zero_fill)
      q.put += zero_fill

      ring_mv[0:rem_packet_cnt] = array.array('I', cmds[tail_blit_dword:])
      q.put += rem_packet_cnt * 4

    q.signal_doorbell(dev)

@dataclass(frozen=True)
class AMDProgramData:
  entry_point_offset:int; rsrc1:int; rsrc2:int; rsrc3:int; wave32:bool
  kernargs_segment_size:int; kernargs_alloc_size:int
  enable_dispatch_ptr:int; enable_private_segment_sgpr:int

_amd_program_cache:dict[tuple[bytes,str], tuple[AMDProgramData,Buffer]] = {}

def amd_build_program(ctx:HCQ2LowerCtx, prg:UOp) -> UOp:
  if (cached:=_amd_program_cache.get(key:=(lib:=prg.src[4].arg, ctx.dev.device))) is None:
    image, sections, relocs = elf_loader(lib)
    rodata = next(sh.header.sh_addr for sh in sections if sh.name == ".rodata")
    for off, sym, typ, addent in relocs:
      assert typ == 5, f"unknown AMD reloc {typ}"  # R_AMDGPU_REL64
      image[off:off+8] = struct.pack('<q', sym - off + addent)
    lib_gpu = Buffer(ctx.dev.device, round_up(image.nbytes, 0x1000), dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    ctx.dev.allocator._copyin(lib_gpu._buf, image)
    ctx.dev.synchronize()
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata:rodata+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
    if (lds:=((desc.group_segment_fixed_size+511)//512)&0x1FF) > (ctx.dev.iface.props['lds_size_in_kb']*1024)//512:
      raise RuntimeError("Too many resources requested: group_segment_size")
    ctx.dev._ensure_has_local_memory(desc.private_segment_fixed_size)
    edp = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    cached = _amd_program_cache[key] = (AMDProgramData(
      entry_point_offset=rodata + desc.kernel_code_entry_byte_offset,
      rsrc1=desc.compute_pgm_rsrc1 | ((1<<20) if ctx.dev.target[0]==11 else 0),  # priv=1 on gfx11 for cwsr
      rsrc2=desc.compute_pgm_rsrc2 | (lds<<15), rsrc3=desc.compute_pgm_rsrc3,
      wave32=bool(desc.kernel_code_properties & 0x400),
      kernargs_segment_size=desc.kernarg_size,
      kernargs_alloc_size=desc.kernarg_size + (ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if edp else 0),
      enable_dispatch_ptr=edp,
      enable_private_segment_sgpr=desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
    ), lib_gpu)
  data, lib_gpu = cached
  return prg.replace(src=(UOp.from_buffer(lib_gpu, ctx.dev.device),), arg=(data, prg.arg))

amd_pm_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"), amd_build_program),
])

class AMDAllocator(HCQAllocator['AMD2Device']):
  def __init__(self, dev:AMD2Device):
    super().__init__(dev, copy_bufs=getattr(dev.iface, 'copy_bufs', None), max_copyout_size=0x1000 if dev.is_usb() else None,
                     supports_copy_from_disk=dev.has_sdma_queue, supports_transfer=dev.has_sdma_queue and not dev.is_usb())

  def _alloc(self, size:int, options:BufferSpec) -> HCQ2Buffer:
    return self.dev.iface.alloc(size, host=True, uncached=options.uncached, cpu_access=True)

  def _do_free(self, opaque, options:BufferSpec): self.dev.iface.free(opaque)

  def _do_map(self, buf:HCQ2Buffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    if not self.dev.is_usb(): return super()._copyout(dest, src)
    if not self.dev.iface.pci_dev.usb.usb.is_custom: return super()._copyout(dest, src)
    self.dev.synchronize()

    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=TracingKey(f"{self.dev.device} -> TINY", ret=dest.nbytes), enabled=PROFILE,
                     dev_suff="SDMA:0"):
      for i in range(0, dest.nbytes, cp_size:=self.b[0].size):
        self.dev.iface.pci_dev.usb.scsi_read_arm(lsize:=min(cp_size, dest.nbytes - i))
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(self.b[0], src.offset(i), lsize) \
                                  .write(self.dev.iface.cq_buf.offset(12), 0) \
                                  .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        dest.cast('B')[i:i+lsize] = self.b[0].cpu_view().view(size=lsize, fmt='B')[:]

@dataclass
class AMDQueueDesc:
  ring: Buffer        # uint32[ring_size//4]
  read_ptr: Buffer    # uint64[1]
  write_ptr: Buffer   # uint64[1]
  doorbell: Buffer    # uint64[1]
  put_value: Buffer   # uint64[1]
  params: tuple|None = None  # setup_ring params for recovery

  @property
  def ring_mv(self) -> MMIOInterface: return self.ring._buf.view.view(fmt='I')
  @property
  def rptr_mv(self) -> MMIOInterface: return self.read_ptr._buf.view.view(fmt='Q')
  @property
  def wptr_mv(self) -> MMIOInterface: return self.write_ptr._buf.view.view(fmt='Q')
  @property
  def doorbell_mv(self) -> MMIOInterface: return self.doorbell._buf.view.view(fmt='Q')
  @property
  def put(self) -> int: return self.put_value._buf.view.view(fmt='Q')[0]
  @put.setter
  def put(self, v:int): self.put_value._buf.view.view(fmt='Q')[0] = v

  def signal_doorbell(self, dev, doorbell_value:int|None=None):
    try:
      self.wptr_mv[0] = self.put
      System.memory_barrier()
      if dev.is_am() and not dev.is_usb(): dev.iface.dev_impl.gmc.flush_hdp()
      self.doorbell_mv[0] = self.put if doorbell_value is None else doorbell_value
    except Exception as e:
      dev.error_state = e
      raise

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    super().__init__(dev, dev_id, vendor=0x1002, devices=((0xffff, (0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), vram_bar=0,
      va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size, dev_impl_t=AMDev)
    self._compute_props()

  def p2p_paddrs(self, paddrs:list[tuple[int,int]]) -> tuple[list[tuple[int,int]], AddrSpace]:
    return ([(self.dev_impl.paddr2xgmi(p), sz) for p, sz in paddrs], AddrSpace.PEER) if self.dev_impl.is_hive() else super().p2p_paddrs(paddrs)

  def require_profile_mode(self): return True
  def is_wgp_active(self, xcc, se, sa, wgp) -> bool: return True # TODO: account for WGP disablement on some asics.

  def _compute_props(self):
    self.ip_versions = self.dev_impl.ip_ver

    gfxver = int(f"{self.dev_impl.ip_ver[am.GC_HWIP][0]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][1]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][2]:02d}")
    if self.dev_impl.gc_info.header.version_major == 2:
      cu_per_sa = self.dev_impl.gc_info.gc_num_cu_per_sh
      max_sh_per_se = self.dev_impl.gc_info.gc_num_sh_per_se
    else:
      cu_per_sa = 2 * (self.dev_impl.gc_info.gc_num_wgp0_per_sa + self.dev_impl.gc_info.gc_num_wgp1_per_sa)
      max_sh_per_se = self.dev_impl.gc_info.gc_num_sa_per_se

    array_count = max_sh_per_se * self.dev_impl.gc_info.gc_num_se * self.dev_impl.gfx.xccs
    self.props = {'cu_per_simd_array': cu_per_sa, 'simd_count': 2 * cu_per_sa * array_count, 'simd_per_cu': 2, 'array_count': array_count,
      'max_slots_scratch_cu': self.dev_impl.gc_info.gc_max_scratch_slots_per_cu, 'max_waves_per_simd': self.dev_impl.gc_info.gc_max_waves_per_simd,
      'simd_arrays_per_engine': max_sh_per_se, 'lds_size_in_kb': self.dev_impl.gc_info.gc_lds_size, 'num_xcc': self.dev_impl.gfx.xccs,
      'gfx_target_version': {90403: 90402}.get(gfxver, gfxver)}

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    assert cwsr_buffer is None, "no cwsr buffer for am"

    rcvr_params: tuple
    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
      doorbell_index = self.dev_impl.sdma.setup_ring(*(rcvr_params:=(ring.va_addr, ring.size, gart.va_addr+rptr, gart.va_addr+wptr, idx)))
    else:
      doorbell_index = self.dev_impl.gfx.setup_ring(*(rcvr_params:=(ring.va_addr, ring.size, gart.va_addr+rptr, gart.va_addr+wptr,
        eop_buffer.va_addr, eop_buffer.size, is_aql:=(queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL), is_aql)))

    ext = lambda addr,n,dt: Buffer("CPU", n, dt, options=BufferSpec(external_ptr=addr), preallocate=True)
    return AMDQueueDesc(ring=ext(ring.va_addr, ring.size//4, dtypes.uint32),
      doorbell=ext(self.dev_impl.doorbell64.addr + doorbell_index*8, 1, dtypes.uint64),
      read_ptr=ext(gart.va_addr+rptr, 1, dtypes.uint64), write_ptr=ext(gart.va_addr+wptr, 1, dtypes.uint64),
      put_value=Buffer("CPU", 1, dtypes.uint64, preallocate=True), params=rcvr_params)

  def _collect_interrupts(self, reset=False, drain_only=False):
    devs:list[AMD2Device] = [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, AMD2Device) and d.is_am()]
    for d in devs:
      if drain_only: d.iface.dev_impl.ih.drain()
      else: d.iface.dev_impl.ih.interrupt_handler()

      if reset and d.iface.dev_impl.recover(force=d.error_state is not None):
        d.compute_queue.put = d.compute_queue.rptr_mv[0] = d.compute_queue.wptr_mv[0] = 0
        d.iface.dev_impl.gfx.setup_ring(*d.compute_queue.params)
        d.timeline_signal.value = d.timeline_value - 1
        d.error_state = None

  def sleep(self, timeout):
    if hasattr(self.pci_dev, 'irq_poller') and self.pci_dev.irq_poller is not None and (events_cnt:=len(self.pci_dev.irq_poller.poll(timeout))):
      self.pci_dev.irq_fd.read(8 * events_cnt)
    self._collect_interrupts()
    if self.dev_impl.is_err_state: raise RuntimeError("Device is in error state")

  def on_device_hang(self):
    self._collect_interrupts(reset=True)
    raise RuntimeError("Device hang detected")

  def device_fini(self): self.dev_impl.fini()

def _mock(iface, name=None): return type(name or f"MOCK{iface.__name__}", (iface,), {})

class AMD2Device(HCQ2Compiled):
  ifaces = [PCIIface]

  def is_am(self) -> bool: return isinstance(self.iface, (PCIIface,))
  def is_usb(self) -> bool: return False

  hcq_routines = amd_hcq_routines

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0

    self.iface = self._select_iface()

    self.target:tuple[int, ...] = ((trgt:=self.iface.props['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
    self.arch = "gfx%d%x%x" % self.target
    assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), f"Unsupported arch: {self.arch}"
    if DEBUG >= 1: print(f"AMD2Device: opening {self.device_id} with target {self.target} arch {self.arch}")

    self.xccs = self.iface.props.get('num_xcc', 1)
    self.se_cnt = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine'] // self.xccs
    self.cu_cnt = self.iface.props['simd_count'] // self.iface.props['simd_per_cu'] // self.xccs
    self.waves_per_cu = self.iface.props['max_waves_per_simd'] * self.iface.props['simd_per_cu']
    self.wave_cnt = (self.cu_cnt * self.waves_per_cu) if self.target[0] != 9 else min(self.cu_cnt * 40, self.se_cnt * self.xccs * 512)

    self.ip_off = importlib.import_module(f"tinygrad.runtime.autogen.am.{'vega' if self.target[0] == 9 else 'navi'}_offsets")
    self.soc = import_soc(self.target)
    self.pm4 = importlib.import_module(f"tinygrad.runtime.autogen.am.pm4_{'soc15' if self.target[0] == 9 else 'nv'}")
    self.sdma = import_module('sdma', min(self.iface.ip_versions[am.SDMA0_HWIP], (6, 0, 0)))
    self.gc = AMDIP('gc', self.iface.ip_versions[am.GC_HWIP],
                    bases={i: tuple(getattr(self.ip_off, f'GC_BASE__INST{i}_SEG{s}', 0) for s in range(6)) for i in range(6)})

    self.nbio = AMDIP('nbio' if self.target[0] < 12 else 'nbif', self.iface.ip_versions[am.NBIF_HWIP],
                      bases={i: tuple(getattr(self.ip_off, f'NBIO_BASE__INST{i}_SEG{s}', 0) for s in range(9)) for i in range(6)})

    self.is_aql = getenv("AMD_AQL", int(self.xccs > 1))
    if self.is_aql:
      self.pm4_ibs = self.iface.alloc(0x2000 if self.is_usb() else (16 << 20), uncached=True, cpu_access=True)
      self.pm4_ib_alloc = BumpAllocator(self.pm4_ibs.size, wrap=True)

    self.max_copy_size = 0x40000000 if self.iface.ip_versions[am.SDMA0_HWIP][0] >= 5 else 0x400000
    self.sdma_queues:dict = {}
    self.has_sdma_queue = False #self.sdma_queue(0) is not None

    super().__init__(device, AMDAllocator(self), [HIPRenderer, AMDLLVMRenderer, HIPCCRenderer], None, AMDSignal,
                     functools.partial(AMDComputeAQLQueue if self.is_aql else AMDComputeQueue, self),
                     functools.partial(AMDCopyQueue, self, max_copy_size=self.max_copy_size) if self.has_sdma_queue else None,
                     kernargs_size=(8 << 10) if self.is_usb() else (16 << 20), sigalloc_size=0x100 if self.is_usb() else 0x1000,
                     can_recover=self.is_am(), arch=self.arch)

    # Scratch setup
    self.max_private_segment_size = 0
    self._ensure_has_local_memory(128) # set default scratch size to 128 bytes per thread

    # HCQ2 device-specific pattern matchers
    self.pm_program, self.pm_encode = amd_pm_program, amd_pm4_encode

    # HCQ2 helpers: timeline signal/value as UOps so HCQ2Graph rewrites can reference them.
    # self.timeline_signal_uop = self._wrap_buffer_uop(self.timeline_signal.base_buf)
    # self.timeline_value_var = UOp.variable(f"_hcq2_tl_{self.device.replace(':','_')}", 0, 1<<48, dtype=dtypes.uint64)

    # opt-in HCQ2 graph (PCI/PM4 AMD only).
    # if getenv("AMD_HCQ2") and isinstance(self.iface, PCIIface) and not self.is_aql:
    #   from tinygrad.runtime.graph.hcq2 import HCQ2Graph
    #   self.graph = HCQ2Graph

    self.pmc_enabled:bool = PROFILE > 0 and PMC > 0
    if self.pmc_enabled:
      self.iface.require_profile_mode()

      self.pmc_sched:list[PMCSample] = []
      self.pmc_counters = import_pmc(self.target)

      # validate counters: SQ for SIMD busy/instruction counts, LDS stats, GRBM for GPU cycles, L2 cache hits/misses
      l2, lds = ("TCC", "SQ") if self.target[0] == 9 else ("GL2C", "SQC")
      pmc_default = f"SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,{lds}_LDS_IDX_ACTIVE,{lds}_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,{l2}_HIT,{l2}_MISS"
      for k in (PMC_COUNTERS:=getenv("PMC_COUNTERS", pmc_default).split(",")):
        if k not in self.pmc_counters: raise RuntimeError(f"PMC counter {k} is not supported. Available: {','.join(self.pmc_counters.keys())}")

      cast(AMDComputeQueue, unwrap(self.hw_compute_queue_t)()).pmc_start([(k, *self.pmc_counters[k]) for k in PMC_COUNTERS]).submit(self)
      self.pmc_buffer = self.allocator.alloc(self.pmc_sched[-1].off + self.pmc_sched[-1].size, BufferSpec(nolru=True, uncached=True))
      self.allocator._copyin(self.pmc_buffer, memoryview(bytearray(self.pmc_buffer.size))) # zero pmc buffers, some counters have only lo part.

    # SQTT is disabled by default because of runtime overhead and big file sizes (~200mb to Tensor.full() two 4096x4096 tensors and matmul them)
    self.sqtt_enabled:bool = PROFILE > 0 and SQTT > 0
    if self.sqtt_enabled:
      self.iface.require_profile_mode()

      SQTT_BUFFER_SIZE = getenv("SQTT_BUFFER_SIZE", 256) # in mb, per shader engine
      self.sqtt_buffers = [self.allocator.alloc(SQTT_BUFFER_SIZE<<20, BufferSpec(nolru=True, uncached=True)) for _ in range(self.se_cnt * self.xccs)]
      self.sqtt_wptrs = self.allocator.alloc(round_up(self.se_cnt * self.xccs * 4, 0x1000), BufferSpec(cpu_access=True, nolru=True))
      self.sqtt_next_cmd_id = itertools.count(0)

  @functools.cached_property
  def compute_queue(self) -> AMDQueueDesc:
    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, hwreg_size_per_cu = 0x4000, 0x1000
    lds_size_per_cu = self.iface.props["lds_size_in_kb"] << 10 if self.target[:2] == (9,5) else 0x10000
    vgpr_size_per_cu = 0x60000 if self.target in {(11,0,0), (11,0,1), (11,5,1), (12,0,0), (12,0,1)} else 0x80000 if self.target[0] == 9 else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * self.cu_cnt, mmap.PAGESIZE)
    ctl_stack_size = round_up((12 if self.target[0] != 9 else 8) * self.wave_cnt + 8 + 40, mmap.PAGESIZE)
    return self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL if self.is_aql else kfd.KFD_IOC_QUEUE_TYPE_COMPUTE,
      0x2000 if self.is_usb() else (16 << 20), eop_buffer_size=0x1000,
      ctx_save_restore_size=0 if self.is_am() else wg_data_size + ctl_stack_size, ctl_stack_size=ctl_stack_size,
      debug_memory_size=round_up(self.wave_cnt * 32, 64))

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0, idx=0):
    ring = self.iface.alloc(ring_size, uncached=True, cpu_access=True)
    gart = self.iface.alloc(0x100, uncached=True, cpu_access=True)

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL:
      self.aql_gart = gart
      self.aql_desc = hsa.amd_queue_t(queue_properties=hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
        read_dispatch_id_field_base_byte_offset=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
        max_cu_id=(self.cu_cnt * self.xccs) - 1, max_wave_id=self.waves_per_cu - 1)
      self.aql_gart.cpu_view().view(fmt='B')[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size) * self.xccs, mmap.PAGESIZE)
    cwsr_buffer = self.iface.alloc(cwsr_buffer_size) if ctx_save_restore_size else None
    eop_buffer = self.iface.alloc(eop_buffer_size) if eop_buffer_size else None

    return (self.iface.create_queue(queue_type, ring, gart, rptr=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
            wptr=getattr(hsa.amd_queue_t, 'write_dispatch_id').offset, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer,
            ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size, idx=idx))

  def sdma_queue(self, idx:int):
    if getenv("AMD_DISABLE_SDMA"): return None
    if idx in self.sdma_queues: return self.sdma_queues[idx]
    with contextlib.suppress(OSError):
      self.sdma_queues[idx] = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x200 if self.is_usb() else (16 << 20), idx=idx)
    return self.sdma_queues.get(idx, None)

  def _ensure_has_local_memory(self, private_segment_size):
    if self.max_private_segment_size >= private_segment_size: return

    lanes_per_wave = 64 # wave64
    mem_alignment_size = 256 if self.target[0] != 9 else 1024
    size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
    size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt
    self.scratch, ok = self._realloc(getattr(self, 'scratch', None), size_per_xcc * self.xccs)
    if ok:
      # NOTE: xcc logic is correct only for GFX9.
      max_scratch_waves = self.cu_cnt * self.iface.props['max_slots_scratch_cu'] * self.xccs
      wave_scratch = ceildiv(lanes_per_wave * size_per_thread, mem_alignment_size)
      num_waves = (size_per_xcc // (wave_scratch * mem_alignment_size)) // (self.se_cnt if self.target[0] != 9 else 1)

      tmpring_t = getattr(hsa, f'union_COMPUTE_TMPRING_SIZE{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')
      self.tmpring_size = int.from_bytes(tmpring_t(WAVES=min(num_waves, max_scratch_waves), WAVESIZE=wave_scratch), 'little')
      self.max_private_segment_size = private_segment_size

      if hasattr(self, 'aql_desc'):
        gfx9_rsrc = {'NUM_FORMAT':hsa.BUF_NUM_FORMAT_UINT, 'DATA_FORMAT':hsa.BUF_DATA_FORMAT_32, 'ELEMENT_SIZE':1, 'INDEX_STRIDE':3}
        rsrc = {'DST_SEL_X':hsa.SQ_SEL_X, 'DST_SEL_Y':hsa.SQ_SEL_Y, 'DST_SEL_Z':hsa.SQ_SEL_Z, 'DST_SEL_W':hsa.SQ_SEL_W, 'ADD_TID_ENABLE':1,
                'TYPE':hsa.SQ_RSRC_BUF, **(gfx9_rsrc if self.target[0] == 9 else {'FORMAT':hsa.BUF_FORMAT_32_UINT, 'OOB_SELECT':2})}
        rsrc1_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD1{"_GFX11" if self.target[0] != 9 else ""}_bitfields')
        rsrc3_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD3{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')

        self.aql_desc.scratch_backing_memory_location = int(self.scratch.va_addr)
        self.aql_desc.scratch_wave64_lane_byte_size = self.max_private_segment_size * lanes_per_wave // 64
        self.aql_desc.scratch_resource_descriptor[:] = [lo32(self.scratch.va_addr),
          int.from_bytes(rsrc1_t(BASE_ADDRESS_HI=hi32(self.scratch.va_addr), SWIZZLE_ENABLE=1), 'little'),
          lo32(size_per_xcc), int.from_bytes(bytes(rsrc3_t(**rsrc)), 'little')]
        self.aql_desc.compute_tmpring_size = self.tmpring_size
        self.aql_gart.cpu_view()[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

  def invalidate_caches(self):
    unwrap(self.hw_compute_queue_t)().memory_barrier().signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()

  def on_device_hang(self): self.iface.on_device_hang()

  def device_props(self): return self.iface.props
