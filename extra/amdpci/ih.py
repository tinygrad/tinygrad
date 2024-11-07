import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_osssys_6_0_0, amdgpu_nbio_4_3_0
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

AMDGPU_NAVI10_DOORBELL_IH = 0x178

class IHRing():
  def __init__(self, adev, doorbell_index, is_1=False):
    self.adev = adev
    self.doorbell_index = doorbell_index << 1

    self.gpu_vaddr = self.adev.vmm.alloc_vram(262144)
    self.gpu_paddr = self.adev.vmm.vaddr_to_paddr(self.gpu_vaddr)
    self.gpu_mc_addr = self.adev.vmm.paddr_to_mc(self.gpu_paddr)

    self.rptr_gpu_vaddr = self.adev.vmm.alloc_vram(0x1000)
    self.wptr_gpu_vaddr = self.adev.vmm.alloc_vram(0x1000)

    self.rptr_gpu_paddr = self.adev.vmm.vaddr_to_paddr(self.rptr_gpu_vaddr)
    self.wptr_gpu_paddr = self.adev.vmm.vaddr_to_paddr(self.wptr_gpu_vaddr)

    self.rptr_cpu_addr = self.adev.vmm.paddr_to_cpu_addr(self.rptr_gpu_paddr)
    self.wptr_cpu_addr = self.adev.vmm.paddr_to_cpu_addr(self.wptr_gpu_paddr)
    self.rptr_cpu_view = self.adev.vmm.paddr_to_cpu_mv(self.rptr_gpu_paddr, 8)
    self.wptr_cpu_view = self.adev.vmm.paddr_to_cpu_mv(self.wptr_gpu_paddr, 8)

    if not is_1:
      self.ih_rb_base = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_BASE, amdgpu_osssys_6_0_0.regIH_RB_BASE_BASE_IDX)
      self.ih_rb_base_hi = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_BASE_HI, amdgpu_osssys_6_0_0.regIH_RB_BASE_HI_BASE_IDX)
      self.ih_rb_cntl = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_CNTL, amdgpu_osssys_6_0_0.regIH_RB_CNTL_BASE_IDX)
      self.ih_rb_wptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_WPTR, amdgpu_osssys_6_0_0.regIH_RB_WPTR_BASE_IDX)
      self.ih_rb_rptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_RPTR, amdgpu_osssys_6_0_0.regIH_RB_RPTR_BASE_IDX)
      self.ih_doorbell_rptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_DOORBELL_RPTR, amdgpu_osssys_6_0_0.regIH_DOORBELL_RPTR_BASE_IDX)
      self.ih_rb_wptr_addr_lo = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_WPTR_ADDR_LO, amdgpu_osssys_6_0_0.regIH_RB_WPTR_ADDR_LO_BASE_IDX)
      self.ih_rb_wptr_addr_hi = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_WPTR_ADDR_HI, amdgpu_osssys_6_0_0.regIH_RB_WPTR_ADDR_HI_BASE_IDX)
      self.psp_reg_id = 0
    else:
      self.ih_rb_base = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_BASE_RING1, amdgpu_osssys_6_0_0.regIH_RB_BASE_RING1_BASE_IDX)
      self.ih_rb_base_hi = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_BASE_HI_RING1, amdgpu_osssys_6_0_0.regIH_RB_BASE_HI_RING1_BASE_IDX)
      self.ih_rb_cntl = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_CNTL_RING1, amdgpu_osssys_6_0_0.regIH_RB_CNTL_RING1_BASE_IDX)
      self.ih_rb_wptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_WPTR_RING1, amdgpu_osssys_6_0_0.regIH_RB_WPTR_RING1_BASE_IDX)
      self.ih_rb_rptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RB_RPTR_RING1, amdgpu_osssys_6_0_0.regIH_RB_RPTR_RING1_BASE_IDX)
      self.ih_doorbell_rptr = self.adev.reg_off("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_DOORBELL_RPTR_RING1, amdgpu_osssys_6_0_0.regIH_DOORBELL_RPTR_RING1_BASE_IDX)
      self.psp_reg_id = 1

class IH_IP:
  def __init__(self, adev):
    self.adev = adev

  def init(self):
    print("IH init")
    self.create_rings()
    self.ih_v6_0_irq_init()

  def create_rings(self):
    self.ih = IHRing(self.adev, AMDGPU_NAVI10_DOORBELL_IH)
    self.ih1 = IHRing(self.adev, AMDGPU_NAVI10_DOORBELL_IH + 1, is_1=True)

  def ih_v6_0_enable_ring(self, ring, is_1):
    self.adev.wreg(ring.ih_rb_base, (ring.gpu_vaddr >> 8) & 0xffffffff)
    self.adev.wreg(ring.ih_rb_base_hi, (ring.gpu_vaddr >> 40) & 0xff)

    cntr = 0xC0310120 if not is_1 else 0xC0100320
    self.adev.wreg(ring.ih_rb_cntl, cntr)

    if not is_1:
      self.adev.wreg(ring.ih_rb_wptr_addr_lo, ring.wptr_gpu_vaddr & 0xffffffff)
      self.adev.wreg(ring.ih_rb_wptr_addr_hi, ring.wptr_gpu_vaddr >> 32)

    self.adev.wreg(ring.ih_rb_wptr, 0)
    self.adev.wreg(ring.ih_rb_rptr, 0)

    self.adev.wreg(ring.ih_doorbell_rptr, ring.doorbell_index | 0x10000000) # enable

  def ih_v6_0_toggle_ring_interrupts(self, ring, is_1):
    self.adev.wreg(ring.ih_rb_cntl, 0x40330121 if not is_1 else 0x40100221)

  def ih_v6_0_toggle_interrupts(self):
    self.ih_v6_0_toggle_ring_interrupts(self.ih, is_1=False)
    self.ih_v6_0_toggle_ring_interrupts(self.ih1, is_1=True)

  def ih_v6_0_irq_init(self):
    self.ih_v6_0_enable_ring(self.ih, is_1=False)
    self.ih_v6_0_enable_ring(self.ih1, is_1=True)

    # toggle the doorbell for ih0
    # WREG32_SOC15(NBIO, 0, regS2A_DOORBELL_ENTRY_1_CTRL, ih_doorbell_range);
    self.adev.wreg_ip("NBIO", 0, amdgpu_nbio_4_3_0.regS2A_DOORBELL_ENTRY_1_CTRL, amdgpu_nbio_4_3_0.regS2A_DOORBELL_ENTRY_1_CTRL_BASE_IDX, 0x00057801)

    # CLIENT18_IS_STORM_CLIENT = 1
    self.adev.wreg_ip("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_STORM_CLIENT_LIST_CNTL, amdgpu_osssys_6_0_0.regIH_STORM_CLIENT_LIST_CNTL_BASE_IDX, 0x40000)

    self.adev.wreg_ip("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_INT_FLOOD_CNTL, amdgpu_osssys_6_0_0.regIH_INT_FLOOD_CNTL_BASE_IDX, 0x8)
    self.adev.wreg_ip("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_MSI_STORM_CTRL, amdgpu_osssys_6_0_0.regIH_MSI_STORM_CTRL_BASE_IDX, 0x3)
    self.adev.wreg_ip("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RING1_CLIENT_CFG_INDEX, amdgpu_osssys_6_0_0.regIH_RING1_CLIENT_CFG_INDEX_BASE_IDX, 0x0)
    self.adev.wreg_ip("OSSSYS", 0, amdgpu_osssys_6_0_0.regIH_RING1_CLIENT_CFG_DATA, amdgpu_osssys_6_0_0.regIH_RING1_CLIENT_CFG_DATA_BASE_IDX, 0x1000a)

    self.ih_v6_0_toggle_interrupts()
