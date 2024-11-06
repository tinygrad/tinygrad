import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_ip_offset, amdgpu_mp_13_0_0, amdgpu_nbio_4_3_0
from tinygrad.helpers import to_mv, mv_address

def check(x): assert x == 0

check(libpciaccess.pci_system_init())

pci_iter = libpciaccess.pci_id_match_iterator_create(None)
print(pci_iter)

pcidev = None
while True:
  pcidev = libpciaccess.pci_device_next(pci_iter)
  if not pcidev: break
  dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
  print(dev_fmt, hex(pcidev.contents.vendor_id), hex(pcidev.contents.device_id))
  
  if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c:
    dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
    if dev_fmt == "0000:03:00.0": continue # skip it, use for kernel hacking.
    if dev_fmt == "0000:86:00.0": continue # skip it, use for kernel hacking.
    if dev_fmt == "0000:c6:00.0": continue # skip it, use for kernel hacking.
    if dev_fmt == "0000:44:00.0": continue # skip it, use for kernel hacking.
    if dev_fmt == "0000:83:00.0": continue # skip it, use for kernel hacking.
    # if dev_fmt == "0000:c3:00.0": continue # skip it, use for kernel hacking.
    # print(dev_fmt)
    # exit(0)
    break

assert pcidev is not None
pcidev = pcidev.contents

libpciaccess.pci_device_probe(ctypes.byref(pcidev))

class AMDDev:
  def __init__(self, pcidev):
    self.usec_timeout = 10000000
    
    self.pcidev = pcidev
    libpciaccess.pci_device_enable(ctypes.byref(pcidev))

    aper_base = pcidev.regions[0].base_addr
    aper_size = pcidev.regions[0].size
    libpciaccess.pci_device_map_range(ctypes.byref(pcidev), aper_base, aper_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(vram_bar_mem:=ctypes.c_void_p()))
    self.vram_cpu_addr = vram_bar_mem.value
    self.raw_vram = to_mv(vram_bar_mem, 24 << 30)

    doorbell_bar_region_addr = pcidev.regions[2].base_addr
    doorbell_bar_region_size = pcidev.regions[2].size
    x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), doorbell_bar_region_addr, doorbell_bar_region_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(doorbell_bar_mem:=ctypes.c_void_p()))
    self.doorbell = to_mv(doorbell_bar_mem, doorbell_bar_region_size).cast('I')
    self.doorbell64 = to_mv(doorbell_bar_mem, doorbell_bar_region_size).cast('Q')

    pci_region_addr = pcidev.regions[5].base_addr
    pci_region_size = pcidev.regions[5].size
    x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), pci_region_addr, pci_region_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))

    self.pci_mmio = to_mv(pcimem, pci_region_size).cast('I')

    from extra.amdpci.vmm import VMM
    self.vmm = VMM(self) # gmc ip like

    from extra.amdpci.smu import SMU_IP
    self.smu = SMU_IP(self) # soc21

    from extra.amdpci.psp import PSP_IP
    self.psp = PSP_IP(self)

    # Issue a gpu reset...
    if self.psp.is_sos_alive():
      print("sOS is alive, issue mode1 reset")
      self.smu.mode1_reset()

    from extra.amdpci.soc21 import SOC21_IP
    self.soc21 = SOC21_IP(self) # soc21
    self.soc21.init()

    self.vmm.init()

    from extra.amdpci.ih import IH_IP
    self.ih = IH_IP(self)
    self.ih.init()

    from extra.amdpci.psp import PSP_IP
    self.psp = PSP_IP(self)
    self.psp.init()

    exit(1)

    from extra.amdpci.gfx import GFX_IP
    self.gfx = GFX_IP(self)

  def pcie_index_offset(self): return self.reg_off("NBIO", 0, amdgpu_nbio_4_3_0.regBIF_BX_PF0_RSMU_INDEX, amdgpu_nbio_4_3_0.regBIF_BX_PF0_RSMU_INDEX_BASE_IDX)
  def pcie_data_offset(self): return self.reg_off("NBIO", 0, amdgpu_nbio_4_3_0.regBIF_BX_PF0_RSMU_DATA, amdgpu_nbio_4_3_0.regBIF_BX_PF0_RSMU_DATA_BASE_IDX)

  def indirect_rreg(self, reg):
    self.wreg(self.pcie_index_offset(), reg)
    self.rreg(self.pcie_index_offset())
    return self.rreg(self.pcie_data_offset())

  def indirect_wreg(self, reg, val):
    self.wreg(self.pcie_index_offset(), reg)
    self.rreg(self.pcie_index_offset())
    self.wreg(self.pcie_data_offset(), val)
    self.rreg(self.pcie_data_offset())

  def rreg(self, reg):
    if reg > len(self.pci_mmio): return self.indirect_rreg(reg)
    return self.pci_mmio[reg]

  def wreg(self, reg, val):
    if reg > len(self.pci_mmio): self.indirect_wreg(reg, val)
    else: self.pci_mmio[reg] = val

  def ip_base(self, ip, inst, seg):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    return off

  def reg_off(self, ip, inst, reg, seg):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    return off + reg

  def rreg_ip(self, ip, inst, reg, seg, offset=0):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    return self.rreg(off + reg + offset)

  def wreg_ip(self, ip, inst, reg, seg, val, offset=0):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    self.wreg(off + reg + offset, val)

  # def setup(self):
  #   from extra.amdpci.rlc import replay_rlc

  #   self.hw_init()

  def hw_init(self): # gfx_v11_0_hw_init
    replay_rlc(self) # TODO: Split into functions, but this is the same regs setup...

    self.gfx_v11_0_wait_for_rlc_autoload_complete()
    self.get_gb_addr_config() # raises if something wrong
    self.gfx_v11_0_gfxhub_enable()

  def soc21_grbm_select(self, me, pipe, queue, vmid):
    regGRBM_GFX_CNTL = 0xa900 # (adev->reg_offset[GC_HWIP][0][1] + 0x0900)
    GRBM_GFX_CNTL__PIPEID__SHIFT=0x0
    GRBM_GFX_CNTL__MEID__SHIFT=0x2
    GRBM_GFX_CNTL__VMID__SHIFT=0x4
    GRBM_GFX_CNTL__QUEUEID__SHIFT=0x8

    grbm_gfx_cntl = (me << GRBM_GFX_CNTL__MEID__SHIFT) | (pipe << GRBM_GFX_CNTL__PIPEID__SHIFT) | (vmid << GRBM_GFX_CNTL__VMID__SHIFT) | (queue << GRBM_GFX_CNTL__QUEUEID__SHIFT)
    self.wreg(regGRBM_GFX_CNTL, grbm_gfx_cntl)

  def hdp_v6_0_flush_hdp(self): self.wreg(0x1fc00, 0x0)
  def gmc_v11_0_flush_gpu_tlb(self):
    self.wreg(0x291c, 0xf80001)
    while self.rreg(0x292e) != 1: pass

  def gfx_v11_0_gfxhub_enable(self):
    from extra.amdpci.gfxhub import gfxhub_v3_0_gart_enable
    if DEBUG >= 2: print("start gfx_v11_0_gfxhub_enable")
    # gfxhub_v3_0_gart_enable(self)
    # self.hdp_v6_0_flush_hdp()

    self.gmc_v11_0_flush_gpu_tlb()
    self.hdp_v6_0_flush_hdp()
    print("start gfx_v11_0_gfxhub_enable")
  
  def get_gb_addr_config(self):
    gb_addr_config = self.rreg(regGB_ADDR_CONFIG)
    if gb_addr_config == 0: raise RuntimeError("error in get_gb_addr_config: gb_addr_config is 0")

  def gfx_v11_0_wait_for_rlc_autoload_complete(self):
    while True:
      cp_status = self.rreg(regCP_STAT)
      # TODO: some exceptions here for other gpus
      if True:
        bootload_status = self.rreg(regRLC_RLCS_BOOTLOAD_STATUS)

      if cp_status == 0 and ((bootload_status & RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) >> RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE__SHIFT) == 1:
        break

adev = AMDDev(pcidev)