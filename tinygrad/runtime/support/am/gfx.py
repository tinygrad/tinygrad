import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.helpers import to_mv, mv_address, getenv
from tinygrad.runtime.support.am.amring import AMRing

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c
  AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 0x003

  def __init__(self, adev):
    self.adev = adev

  def init(self):
    print("GFX: init")

    self.wait_for_rlc_autoload()
    assert self.gb_addr_config() == 0x545 # gfx11 is the same

    self.config_gfx_rs64()
    self.adev.gmc.init_gfxhub()
    self.init_golden_registers()
    self.constants_init()
    self.nbio_v4_3_gc_doorbell_init()
    self.cp_resume()

  def soc21_grbm_select(self, me=0, pipe=0, queue=0, vmid=0):
    self.adev.regGRBM_GFX_CNTL.write((me << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__MEID__SHIFT) |
      (pipe << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__PIPEID__SHIFT) | (vmid << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__VMID__SHIFT) |
      (queue << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__QUEUEID__SHIFT))

  def wait_for_rlc_autoload(self):
    while True:
      bootload_ready = (self.adev.regRLC_RLCS_BOOTLOAD_STATUS.read() & amdgpu_gc_11_0_0.RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) != 0
      if self.adev.regCP_STAT.read() == 0 and bootload_ready: break

  def gb_addr_config(self): return self.adev.regGB_ADDR_CONFIG.read()
  def init_golden_registers(self): self.adev.regTCP_CNTL.write(self.adev.regTCP_CNTL.read() | 0x20000000)

  def constants_init(self):
    self.adev.regGRBM_CNTL.write(0xff)

    # TODO: Read configs here
    for i in range(8, 16):
      self.soc21_grbm_select(vmid=i)
      self.adev.regSH_MEM_CONFIG.write(0xc00c)
      self.adev.regSH_MEM_BASES.write(0x10002)
    self.soc21_grbm_select()

  def cp_set_doorbell_range(self):
    self.adev.regCP_RB_DOORBELL_RANGE_LOWER.write(0x458)
    self.adev.regCP_RB_DOORBELL_RANGE_UPPER.write(0x7f8)
    self.adev.regCP_MEC_DOORBELL_RANGE_LOWER.write(0x0)
    self.adev.regCP_MEC_DOORBELL_RANGE_UPPER.write(0x450)

  def cp_compute_enable(self):
    self.adev.regCP_MEC_RS64_CNTL.write(0x3c000000)
    time.sleep(0.5)

  def nbio_v4_3_gc_doorbell_init(self):
    self.adev.regS2A_DOORBELL_ENTRY_0_CTRL.write(0x30000007)
    self.adev.regS2A_DOORBELL_ENTRY_3_CTRL.write(0x3000000d)

  def config_gfx_rs64(self):
    for pipe in range(2):
      self.soc21_grbm_select(pipe=pipe)
      self.adev.regCP_PFP_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_PFP_PRGRM_CNTR_START_HI.write(0x1c000)

    # TODO: write up!
    self.soc21_grbm_select()
    self.adev.regCP_ME_CNTL.write(0x153c0000)
    self.adev.regCP_ME_CNTL.write(0x15300000)

    for pipe in range(2):
      self.soc21_grbm_select(pipe=pipe)
      self.adev.regCP_ME_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_ME_PRGRM_CNTR_START_HI.write(0x1c000)

    self.soc21_grbm_select()
    self.adev.regCP_ME_CNTL.write(0x15300000)
    self.adev.regCP_ME_CNTL.write(0x15000000)

    for pipe in range(4):
      self.soc21_grbm_select(me=1, pipe=pipe)
      self.adev.regCP_MEC_RS64_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_MEC_RS64_PRGRM_CNTR_START_HI.write(0x1c000)

    self.soc21_grbm_select()
    self.adev.regCP_MEC_RS64_CNTL.write(0x400f0000)
    self.adev.regCP_MEC_RS64_CNTL.write(0x40000000)

  def cp_gfx_enable(self):
    self.adev.regCP_ME_CNTL.write(0x1000000)
    self.adev.wait_reg(self.adev.regCP_STAT, value=0x0)

  def hqd_load(self, ring):
    self.soc21_grbm_select(me=1, pipe=ring.pipe, queue=ring.queue)

    mqd_mv = ring.mqd_mv.cast('I')
    for i, reg in enumerate(range(self.adev.regCP_MQD_BASE_ADDR.regoff, self.adev.regCP_HQD_PQ_WPTR_HI.regoff + 1)):
      self.adev.wreg(reg, mqd_mv[0x80 + i])

    self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.write(ring.mqd.cp_hqd_pq_doorbell_control)
    self.adev.regCP_HQD_ACTIVE.write(0x1)

    self.soc21_grbm_select(0, 0, 0, 0)

  def kcq_init(self):
    self.kcq_ring = AMRing(self.adev, size=0x100000, me=1, pipe=0, queue=1, vmid=0, doorbell_index=((self.AMDGPU_NAVI10_DOORBELL_MEC_RING0) << 1))
    self.hqd_load(self.kcq_ring)
    time.sleep(5)

  def cp_resume(self):
    self.cp_set_doorbell_range()
    self.cp_compute_enable()
    self.cp_gfx_enable()
    self.kcq_init()
