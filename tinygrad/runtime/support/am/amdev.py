import os, ctypes, collections, time
from tinygrad.runtime.autogen import libpciaccess
from tinygrad.runtime.autogen import amdgpu_2, amdgpu_mp_13_0_0, amdgpu_nbio_4_3_0, amdgpu_discovery, amdgpu_mmhub_3_0_0, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

from tinygrad.runtime.support.am.mm import MM, PhysicalMemory
from tinygrad.runtime.support.am.firmware import Firmware
from tinygrad.runtime.support.am.gmc import GMC_IP
from tinygrad.runtime.support.am.soc21 import SOC21_IP
from tinygrad.runtime.support.am.smu import SMU_IP
from tinygrad.runtime.support.am.psp import PSP_IP
from tinygrad.runtime.support.am.gfx import GFX_IP
# from tinygrad.runtime.support.am.mes import MES_IP

class AMRegister:
  def __init__(self, adev, regoff): self.adev, self.regoff = adev, regoff
  def write(self, value, inst=0): return self.adev.wreg(self.regoff, value)
  def read(self, inst=0): return self.adev.rreg(self.regoff)

class AMDev:
  def __init__(self, pcidev):
    self.pcidev = pcidev
    self.usec_timeout = 1000000

    self.vram_cpu_addr, self.vram = self._map_pci_range(bar=0, cast='B')
    self.doorbell_cpu_addr, self.doorbell64 = self._map_pci_range(bar=2, cast='Q')
    self.mmio_cpu_addr, self.mmio = self._map_pci_range(bar=5, cast='I')

    self.start_discovery()
    self._prepare_registers([("MP0", amdgpu_mp_13_0_0), ("NBIO", amdgpu_nbio_4_3_0), ("MMHUB", amdgpu_mmhub_3_0_0), ("GC", amdgpu_gc_11_0_0)])

    # Memory manager & firmware
    self.mm = MM(self, self.vram_size)
    self.fw = Firmware(self)

    # Initialize IP blocks
    self.gmc = GMC_IP(self)
    self.soc21 = SOC21_IP(self)
    self.smu = SMU_IP(self)
    self.psp = PSP_IP(self)
    self.gfx = GFX_IP(self)
    # self.mes = MES_IP(self)

    if self.psp.is_sos_alive():
      print("sOS is alive, issue mode1 reset...")
      self.smu.mode1_reset()
      time.sleep(1)

    self.soc21.init()
    self.gmc.init(self.mm.root_pt)
    self.psp.init()
    self.smu.init()
    self.gfx.init()

  def _map_pci_range(self, bar, cast='I'):
    ret = libpciaccess.pci_device_map_range(ctypes.byref(self.pcidev), self.pcidev.regions[bar].base_addr, size:=self.pcidev.regions[bar].size,
      libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))
    return pcimem.value, to_mv(pcimem.value, size).cast(cast)

  def _prepare_registers(self, modules):
    for base, m in modules:
      for k, regval in m.__dict__.items():
        if k.startswith("reg") and not k.endswith("_BASE_IDX") and (base_idx:=getattr(m, f"{k}_BASE_IDX", None)) is not None:
          setattr(self, k, AMRegister(self, self.reg_off(base, 0, regval, base_idx)))

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

  def reg_off(self, ip, inst, reg, seg): return self.ip_base(ip, inst, seg) + reg
  def rreg(self, reg): return self.indirect_rreg(reg) if reg > len(self.mmio) else self.mmio[reg]
  def wreg(self, reg, val):
    if reg > len(self.mmio): self.indirect_wreg(reg, val)
    else: self.mmio[reg] = val

  def wait_reg(self, reg, value, mask=0xffffffff):
    v = reg.read()
    while v & mask != value: v = reg.read()

    # for i in range(100):
    #   val = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_STAT, amdgpu_gc_11_0_0.regCP_STAT_BASE_IDX)
    #   if val == 0: return
    #   time.sleep(0.00001)
    # raise Exception('gfx_v11_0_cp_gfx_enable timeout')

  def wdoorbell64(self, index, val): self.doorbell64[index//2] = val

  def ip_base(self, ip, inst, seg):
    ipid = amdgpu_discovery.__dict__.get(f"{ip}_HWIP")
    try: x = self.regs_offset[ipid][inst][seg]
    except (KeyError, IndexError): x = 0
    return x

  def start_discovery(self):
    mmIP_DISCOVERY_VERSION = 0x16A00
    mmRCC_CONFIG_MEMSIZE = 0xde3
    mmMP0_SMN_C2PMSG_33 = 0x16061
    DISCOVERY_TMR_OFFSET = (64 << 10)
    DISCOVERY_TMR_SIZE = (10 << 10)

    # Wait for IFWI init to complete.
    for i in range(1000):
      if self.rreg(mmMP0_SMN_C2PMSG_33) & 0x80000000: break
      time.sleep(0.001)

    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    self.discovery_pm = PhysicalMemory(self, self.vram_size - DISCOVERY_TMR_OFFSET, DISCOVERY_TMR_SIZE)
    print("Detected VRAM size", self.vram_size)

    bhdr = amdgpu_discovery.struct_binary_header.from_address(self.discovery_pm.cpu_addr())

    ip_offset = bhdr.table_list[amdgpu_discovery.IP_DISCOVERY].offset
    ihdr = amdgpu_discovery.struct_ip_discovery_header.from_address(ctypes.addressof(bhdr) + ip_offset)
    assert ihdr.signature == amdgpu_discovery.DISCOVERY_TABLE_SIGNATURE

    hw_id_map = {}
    for x,y in amdgpu_discovery.hw_id_map:
      hw_id_map[amdgpu_discovery.__dict__[x]] = int(y)

    self.regs_offset = collections.defaultdict(dict)

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

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * num_base_address
