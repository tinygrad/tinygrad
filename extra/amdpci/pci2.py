import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_ip_offset, amdgpu_mp_13_0_0_offset
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
  print(dev_fmt)
  
  if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c:
    dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
    # if dev_fmt == "0000:03:00.0": continue # skip it, use for kernel hacking.
    break

assert pcidev is not None
pcidev = pcidev.contents

libpciaccess.pci_device_probe(ctypes.byref(pcidev))

class AMDDev:
  def __init__(self, pcidev):
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

    pci_region_addr = pcidev.regions[5].base_addr
    pci_region_size = pcidev.regions[5].size
    x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), pci_region_addr, pci_region_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))

    self.pci_mmio = to_mv(pcimem, pci_region_size).cast('I')

    from extra.amdpci.vmm import VMM
    self.vmm = VMM(self)

    regMMVM_CONTEXT0_CNTL = 0x0740
    regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x07ab
    regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x07ac
    regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x07cb
    regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x07cc
    regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x07eb
    regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x07ec
    # print("ccc", hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_CNTL, 0)))
    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0)))
    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0)))

    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0)))
    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, 0)))

    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0)))
    # print(hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0)))

    # just reset
    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (512 << 20) - 1)
    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, 0, 0)

    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, self.vmm.pdb0_base & 0xffffffff)
    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.vmm.pdb0_base >> 32) & 0xffffffff)

    v = self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_CNTL, 0)
    # print(v, "CC")
    self.wreg_ip("MMHUB", 0, regMMVM_CONTEXT0_CNTL, 0, 0x1fffe03)
    # hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_CNTL, 0))
    # print("ccc", hex(self.rreg_ip("MMHUB", 0, regMMVM_CONTEXT0_CNTL, 0)))

    from extra.amdpci.psp import PSP_IP
    self.psp = PSP_IP(self)

    from extra.amdpci.gfx import GFX_IP
    self.gfx = GFX_IP(self)

  def rreg(self, reg): return self.pci_mmio[reg]
  def wreg(self, reg, val): self.pci_mmio[reg] = val

  def rreg_ip(self, ip, inst, reg, seg, offset=0):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    return self.rreg(off + reg + offset)

  def wreg_ip(self, ip, inst, reg, seg, val, offset=0):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    self.wreg(off + reg + offset, val)

  def setup(self):
    from extra.amdpci.rlc import replay_rlc

    self.hw_init()

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
    gfxhub_v3_0_gart_enable(self)
    self.hdp_v6_0_flush_hdp()

    self.gmc_v11_0_flush_gpu_tlb()
    self.hdp_v6_0_flush_hdp()
  
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