import os, ctypes, collections, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0, amdgpu_nbio_4_3_0, amdgpu_discovery, amdgpu_mmhub_3_0_0
from tinygrad.helpers import to_mv, mv_address

class AMDDev:
  def __init__(self, pcidev):
    self.usec_timeout = 10000000
    self.regs_offset = collections.defaultdict(dict)

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

    self.init_discovery()

    from extra.amdpci.smu import SMU_IP
    self.smu = SMU_IP(self) # soc21

    from extra.amdpci.psp import PSP_IP
    self.psp = PSP_IP(self)

    # Issue a gpu reset...
    if self.psp.is_sos_alive():
      print("sOS is alive, issue mode1 reset, 1s to sleep")
      self.smu.mode1_reset()
      time.sleep(1)

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

    self.smu.init()

    from extra.amdpci.gfx import GFX_IP
    self.gfx = GFX_IP(self)

    from extra.amdpci.mes import MES_IP
    self.mes = MES_IP(self)

    self.gfx.init()

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
    # print("read from", hex(reg))
    if reg > len(self.pci_mmio): return self.indirect_rreg(reg)
    return self.pci_mmio[reg]

  def wreg(self, reg, val):
    # print("write to", hex(reg), hex(val))
    if reg > len(self.pci_mmio): self.indirect_wreg(reg, val)
    else: self.pci_mmio[reg] = val

  def ip_base(self, ip, inst, seg):
    ipid = amdgpu_discovery.__dict__.get(f"{ip}_HWIP")
    try: x = self.regs_offset[ipid][inst][seg]
    except (KeyError, IndexError): x = 0
    return x

  def reg_off(self, ip, inst, reg, seg):
    off = self.ip_base(ip, inst, seg)
    return off + reg

  def rreg_ip(self, ip, inst, reg, seg, offset=0):
    off = self.ip_base(ip, inst, seg)
    return self.rreg(off + reg + offset)

  def wreg_ip(self, ip, inst, reg, seg, val, offset=0):
    off = self.ip_base(ip, inst, seg)
    self.wreg(off + reg + offset, val)

  def init_discovery(self):
    mmIP_DISCOVERY_VERSION = 0x16A00
    mmRCC_CONFIG_MEMSIZE = 0xde3
    mmMP0_SMN_C2PMSG_33 = 0x16061
    DISCOVERY_TMR_OFFSET = (64 << 10)
    DISCOVERY_TMR_SIZE = (10 << 10)

    # Wait for IFWI init to complete.
    for i in range(1000):
      msg = self.rreg(mmMP0_SMN_C2PMSG_33)
      if msg & 0x80000000: break
      time.sleep(0.001)

    vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    print("Detected VRAM size", vram_size)

    pos = vram_size - DISCOVERY_TMR_OFFSET
    self.discovery_blob = self.vmm.paddr_to_cpu_mv(pos, DISCOVERY_TMR_SIZE, allow_high=True)

    bhdr = amdgpu_discovery.struct_binary_header.from_buffer(self.discovery_blob)
    
    ip_offset = bhdr.table_list[amdgpu_discovery.IP_DISCOVERY].offset
    ihdr = amdgpu_discovery.struct_ip_discovery_header.from_address(ctypes.addressof(bhdr) + ip_offset)
    # print(hex(ihdr.signature))
    assert ihdr.signature == amdgpu_discovery.DISCOVERY_TABLE_SIGNATURE

    hw_id_map = {}
    for x,y in amdgpu_discovery.hw_id_map:
      hw_id_map[amdgpu_discovery.__dict__[x]] = int(y)
    # print(hw_id_map)

    num_dies = ihdr.num_dies
    for num_die in range(num_dies):
      die_offset = ihdr.die_info[num_die].die_offset
      dhdr = amdgpu_discovery.struct_die_header.from_address(ctypes.addressof(bhdr) + die_offset)
      num_ips = dhdr.num_ips
      ip_offset = die_offset + ctypes.sizeof(dhdr)

      for num_ip in range(num_ips):
        ip = amdgpu_discovery.struct_ip_v4.from_address(ctypes.addressof(bhdr) + ip_offset)
        num_base_address = ip.num_base_address

        assert not ihdr.base_addr_64_bit
        base_addresses = []
        ba = (ctypes.c_uint32 * num_base_address).from_address(ctypes.addressof(bhdr) + ip_offset + 8)

        for hw_ip in range(1, amdgpu_discovery.MAX_HWIP):
          if hw_ip in hw_id_map and hw_id_map[hw_ip] == ip.hw_id:
            self.regs_offset[hw_ip][ip.instance_number] = [x for x in ba]
            # print("set ip instance", hw_ip, ip.instance_number, [hex(x) for x in ba])

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * num_base_address


if __name__ == "__main__":
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
      # if dev_fmt == "0000:c6:00.0": continue # skip it, use for kernel hacking.
      if dev_fmt == "0000:44:00.0": continue # skip it, use for kernel hacking.
      if dev_fmt == "0000:83:00.0": continue # skip it, use for kernel hacking.
      if dev_fmt == "0000:c3:00.0": continue # skip it, use for kernel hacking.
      # print(dev_fmt)
      # exit(0)
      break

  assert pcidev is not None
  pcidev = pcidev.contents

  libpciaccess.pci_device_probe(ctypes.byref(pcidev))

  adev = AMDDev(pcidev)