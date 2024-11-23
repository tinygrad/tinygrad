import os, ctypes, collections, time
from tinygrad.runtime.autogen import libpciaccess
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0
from tinygrad.helpers import to_mv, mv_address
from tinygrad.runtime.support.am.mm import MM, PhysicalMemory
from tinygrad.runtime.support.am.firmware import Firmware
from tinygrad.runtime.support.am.gmc import GMC_IP
from tinygrad.runtime.support.am.amring import AMRegister
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GFX, AM_PSP, AM_SMU

class AMDev:
  def __init__(self, pcidev):
    self.pcidev = pcidev

    self.vram_cpu_addr, self.vram = self._map_pci_range(bar=0, cast='B')
    self.doorbell_cpu_addr, self.doorbell64 = self._map_pci_range(bar=2, cast='Q')
    self.mmio_cpu_addr, self.mmio = self._map_pci_range(bar=5, cast='I')

    self.start_discovery()
    self._build_regs()

    # Memory manager & firmware
    self.mm = MM(self, self.vram_size)
    self.fw = Firmware(self)

    # Initialize IP blocks
    self.gmc = GMC_IP(self)
    self.soc21 = AM_SOC21(self)
    self.smu = AM_SMU(self)
    self.psp = AM_PSP(self)
    self.gfx = AM_GFX(self)

    if self.psp.is_sos_alive():
      print("sOS is alive, issue mode1 reset...")
      self.smu.mode1_reset()
      time.sleep(0.5)

    self.soc21.init()
    self.gmc.init(self.mm.root_pt)
    self.psp.init()
    self.smu.init()
    self.gfx.init()

  def _map_pci_range(self, bar, cast='I'):
    ret = libpciaccess.pci_device_map_range(ctypes.byref(self.pcidev), self.pcidev.regions[bar].base_addr, size:=self.pcidev.regions[bar].size,
      libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))
    return pcimem.value, to_mv(pcimem.value, size).cast(cast)

  def _build_regs(self):
    mods = [("MP0", mp_13_0_0), ("MP1", mp_11_0, "mmMP1"), ("NBIO", nbio_4_3_0), ("MMHUB", mmhub_3_0_0), ("GC", gc_11_0_0), ("OSSSYS", osssys_6_0_0)]
    for info in mods:
      base, module, rpref = info if len(info) == 3 else (*info, "reg")
      reg_names: Set[str] = set(k[len(rpref):] for k in module.__dict__.keys() if k.startswith(rpref) and not k.endswith("_BASE_IDX"))
      reg_fields: Dict[str, List[int, int]] = collections.defaultdict(dict)
      for k, val in module.__dict__.items():
        if k.endswith("_MASK") and ((rname:=k.split("__")[0]) in reg_names):
          reg_fields[rname][k[2+len(rname):-5].lower()] = (val, module.__dict__.get(f"{k[:-5]}__SHIFT", val.bit_length() - 1))

      for k, regval in module.__dict__.items():
        if k.startswith(rpref) and not k.endswith("_BASE_IDX") and (base_idx:=getattr(module, f"{k}_BASE_IDX", None)) is not None:
          setattr(self, k, AMRegister(self, self.reg_off(base, 0, regval, base_idx), reg_fields.get(k[len(rpref):], {})))

  def indirect_rreg(self, reg):
    self.regBIF_BX_PF0_RSMU_INDEX.write(reg)
    assert self.regBIF_BX_PF0_RSMU_INDEX.read() == reg
    return self.regBIF_BX_PF0_RSMU_DATA.read()

  def indirect_wreg(self, reg, val):
    self.regBIF_BX_PF0_RSMU_INDEX.write(reg)
    assert self.regBIF_BX_PF0_RSMU_INDEX.read() == reg
    self.regBIF_BX_PF0_RSMU_DATA.write(val)
    assert self.regBIF_BX_PF0_RSMU_DATA.read() == val

  def ip_base(self, ip:str, inst:int, seg:int) -> int:
    try: return self.regs_offset[am.__dict__.get(f"{ip}_HWIP")][inst][seg]
    except (KeyError, IndexError): return 0

  def reg_off(self, ip:str, inst:int, reg:int, seg:int): return self.ip_base(ip, inst, seg) + reg
  def rreg(self, reg:int): return self.indirect_rreg(reg * 4) if reg > len(self.mmio) else self.mmio[reg]
  def wreg(self, reg:int, val:int):
    if reg > len(self.mmio): self.indirect_wreg(reg * 4, val)
    else: self.mmio[reg] = val

  def wreg64(self, reg:AMRegister, reg_hi:AMRegister, val:int):
    reg.write(val & 0xffffffff)
    reg_hi.write(val >> 32)

  def wait_reg(self, reg:AMRegister, value:int, mask=0xffffffff):
    for _ in range(10000):
      if ((rval:=reg.read()) & mask) == value: return rval
      time.sleep(0.001)
    raise Exception(f'wait_reg timeout reg=0x{reg.regoff:X} mask=0x{mask:X} value=0x{value:X} last_val=0x{rval}')

  def start_discovery(self):
    mmIP_DISCOVERY_VERSION = 0x16A00
    mmRCC_CONFIG_MEMSIZE = 0xde3
    mmMP0_SMN_C2PMSG_33 = 0x16061
    DISCOVERY_TMR_OFFSET = (64 << 10)
    DISCOVERY_TMR_SIZE = (10 << 10)

    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    self.discovery_pm = PhysicalMemory(self, self.vram_size - DISCOVERY_TMR_OFFSET, DISCOVERY_TMR_SIZE)

    bhdr = am.struct_binary_header.from_address(self.discovery_pm.cpu_addr())

    ip_offset = bhdr.table_list[am.IP_DISCOVERY].offset
    ihdr = am.struct_ip_discovery_header.from_address(ctypes.addressof(bhdr) + ip_offset)
    assert ihdr.signature == am.DISCOVERY_TABLE_SIGNATURE

    hw_id_map = {am.__dict__[x]: int(y) for x,y in am.hw_id_map}
    self.regs_offset = collections.defaultdict(dict)

    num_dies = ihdr.num_dies
    for num_die in range(num_dies):
      die_offset = ihdr.die_info[num_die].die_offset
      dhdr = am.struct_die_header.from_address(ctypes.addressof(bhdr) + die_offset)
      num_ips = dhdr.num_ips
      ip_offset = die_offset + ctypes.sizeof(dhdr)

      for num_ip in range(num_ips):
        ip = am.struct_ip_v4.from_address(ctypes.addressof(bhdr) + ip_offset)
        num_base_address = ip.num_base_address

        assert not ihdr.base_addr_64_bit
        base_addresses = []
        ba = (ctypes.c_uint32 * num_base_address).from_address(ctypes.addressof(bhdr) + ip_offset + 8)

        for hw_ip in range(1, am.MAX_HWIP):
          if hw_ip in hw_id_map and hw_id_map[hw_ip] == ip.hw_id:
            self.regs_offset[hw_ip][ip.instance_number] = [x for x in ba]

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * num_base_address
