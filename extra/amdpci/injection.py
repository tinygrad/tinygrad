import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_ip_offset, amdgpu_mp_13_0_0
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
    if dev_fmt != "0000:c3:00.0": continue # skip it, use for kernel hacking.
    break

assert pcidev is not None
pcidev = pcidev.contents

libpciaccess.pci_device_probe(ctypes.byref(pcidev))

class AMDDev:
  def __init__(self, pcidev):
    self.pcidev = pcidev
    # libpciaccess.pci_device_enable(ctypes.byref(pcidev))

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

    # while True:
    #   regGCVM_CONTEXT1_CNTL = 0x28e9
    #   regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x2975
    #   regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 = 0x2976
    #   regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x2995
    #   regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 = 0x2996
    #   ctx_distance = 1
    #   ctx_addr_distance = 2
    #   for vmid in range(15):
    #     self.wreg(regGCVM_CONTEXT1_CNTL + ctx_distance * vmid, 0x1fffe07)
    #     self.wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + ctx_addr_distance * vmid, 0)
    #     self.wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 + ctx_addr_distance * vmid, 0)
    #     self.wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + ctx_addr_distance * vmid, 0xffffffff)
    #     self.wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 + ctx_addr_distance * vmid, 0xf)
      
    #   print(self.rreg(regGCVM_CONTEXT1_CNTL + ctx_distance * vmid))
    #   print(self.rreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + ctx_addr_distance * vmid))
    #   print(self.rreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 + ctx_addr_distance * vmid))
    #   print(self.rreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + ctx_addr_distance * vmid))
    #   print(self.rreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 + ctx_addr_distance * vmid))

    # for i in range(16):
    #   self.wreg(0x28e8 + i, 0x1fffe01)
    #   print("vmid", i, self.rreg(0x28e8 + i))
    #   print("pt", i, self.rreg(0x28e8 + i))

    self.raw_vram_q = self.raw_vram.cast('Q')
    self.raw_vram_q[0] = 0xdeaddeaddeaddead
    print(hex(self.raw_vram_q[0]))
    # for i in range(4171503, (24 << 30) // 8):
    #   print("\r", i, end="")
    #   if self.raw_vram_q[i] == 0xdeaddeaddeaddead:
    #     print(hex(self.raw_vram_q[i]))
    # print("vram done")

    # self.wreg(0x1fc00, 0x0)

    regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x2953
    regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x2954
    regGCVM_CONTEXT0_CNTL = 0x28e8
    regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x2973
    regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x2974
    regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x2993
    regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x2994
    print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32))) # regCP_HQD_VMID
    print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32))) # regCP_HQD_ACTIVE
    while True:
      vmid = 9
      # if self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32+vmid*2) != 0:
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32+vmid*2)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32+vmid*2)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_CNTL+vmid)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32+vmid*2)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32+vmid*2)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32+vmid*2)))
      #   print(hex(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32+vmid*2)))

      regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_LO = 0x1260 + 0x15e5
      regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_HI = 0x1260 + 0x15e6
      regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_LO = 0x1260 + 0x15e7
      regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI = 0x1260 + 0x15e8
      regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_CNTL = 0xA000 + 0x5e44

      # print(hex(self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_CNTL)))

      # print(self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32+vmid*2))
      if self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32+vmid*2) != 0:
        paddrs = self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32+vmid*2)
        paddrs |= self.rreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32+vmid*2) << 32

        for i in range(512):
          if (self.raw_vram_q[paddrs // 8 + i] != 0 and (self.raw_vram_q[paddrs // 8 + i] & 0x1 == 1)):
            print(i, hex(self.raw_vram_q[paddrs // 8 + i]))

        # self.wreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_CNTL, 1)

        # addr = 0xddeadb000
        
        # resp = (1 << 30)
        # print(hex(resp))
        # self.wreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_HI, resp)
        # v = self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI)
        # print("--", hex(v))
        
        # self.wreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_LO, addr & 0xffffffff)
        # print(hex(self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_LO)))

        # v = self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI)
        # # while v == 0:
        # #   v = self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI)
        # #   print(v)
        # x = self.rreg(regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_LO)
        # if v != 0: print(hex(v), hex(x))

      # self.wreg(0xc040, 0xCAFEDEA0)
      # # self.wreg(0x1fc00, 0x0)
      # print(hex(self.rreg(0x320c))) # regCP_HQD_VMID
      # print(hex(self.rreg(0x320b))) # regCP_HQD_ACTIVE

      # if self.rreg(0xc040) != 0:
      #   print(hex(self.rreg(0xc040)))

  def rreg(self, reg): return self.pci_mmio[reg]
  def wreg(self, reg, val): self.pci_mmio[reg] = val

  def rreg_ip(self, ip, inst, reg, seg):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    return self.rreg(off + reg)

  def wreg_ip(self, ip, inst, reg, seg, val):
    off = amdgpu_ip_offset.__dict__.get(f"{ip}_BASE__INST{inst}_SEG{seg}")
    self.wreg(off + reg, val)

adev = AMDDev(pcidev)
