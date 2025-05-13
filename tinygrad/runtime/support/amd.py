import functools, importlib
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import getbits, round_up, getenv
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.usb import ASM24Controller

@dataclass(frozen=True)
class AMDRegBase:
  name: str
  offset: int
  segment: int
  fields: dict[str, tuple[int, int]]
  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

def collect_registers(module, cls=AMDRegBase) -> dict[str, AMDRegBase]:
  def _split_name(name): return name[:(pos:=next((i for i,c in enumerate(name) if c.isupper()), len(name)))], name[pos:]
  offsets = {k:v for k,v in module.__dict__.items() if _split_name(k)[0] in {'reg', 'mm'} and not k.endswith('_BASE_IDX')}
  bases = {k[:-len('_BASE_IDX')]:v for k,v in module.__dict__.items() if _split_name(k)[0] in {'reg', 'mm'} and k.endswith('_BASE_IDX')}
  fields: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
  for field_name,field_mask in module.__dict__.items():
    if not ('__' in field_name and field_name.endswith('_MASK')): continue
    reg_name, reg_field_name = field_name[:-len('_MASK')].split('__')
    fields[reg_name][reg_field_name.lower()] = ((field_mask & -field_mask).bit_length()-1, field_mask.bit_length()-1)
  # NOTE: Some registers like regGFX_IMU_FUSESTRAP in gc_11_0_0 are missing base idx, just skip them
  return {reg:cls(name=reg, offset=off, segment=bases[reg], fields=fields[_split_name(reg)[1]]) for reg,off in offsets.items() if reg in bases}

def import_module(name:str, version:tuple[int, ...], version_prefix:str=""):
  for ver in [version, version[:2]+(0,), version[:1]+(0, 0)]:
    try: return importlib.import_module(f"tinygrad.runtime.autogen.am.{name}_{version_prefix}{'_'.join(map(str, ver))}")
    except ImportError: pass
  raise ImportError(f"Failed to load autogen module for {name.upper()} {'.'.join(map(str, version))}")

def setup_pci_bars(usb:ASM24Controller, gpu_bus:int, mem_base:int, pref_mem_base:int) -> dict[int, tuple[int, int]]:
  try: need_reset = (usb.pcie_cfg_req(pci.PCI_VENDOR_ID, bus=gpu_bus, dev=0, fn=0, size=2) != 0x1002)
  except RuntimeError: need_reset = True

  if need_reset or getenv("USB_RESCAN_BUS", 0) == 1:
    for bus in range(gpu_bus):
      # All 3 values must be written at the same time.
      buses = (0 << 0) | ((bus+1) << 8) | ((gpu_bus) << 16)
      usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=buses, size=4)

      usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=mem_base>>16, size=2)
      usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xf000, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=pref_mem_base>>16, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=mem_base>>16, size=2)

      usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  mem_space_addr, bar_off, bars = [mem_base, pref_mem_base], 0, {}
  while bar_off < 24:
    cfg = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4)
    bar_mem, bar_space = bool(cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH), cfg & pci.PCI_BASE_ADDRESS_SPACE

    if bar_space == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
      usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=0xffffffff, size=4)
      bar_size = 0xffffffff - (usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4) & 0xfffffff0) + 1

      usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=mem_space_addr[bar_mem], size=4)
      bars[bar_off // 4] = (mem_space_addr[bar_mem], bar_size)
      mem_space_addr[bar_mem] += round_up(bar_size, 2 << 20)

    # 64bit bar, zero out the upper 32 bits
    if bar_space == pci.PCI_BASE_ADDRESS_MEM_TYPE_64: usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, dev=0, fn=0, value=0,size=4)
    bar_off += 8 if cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64 else 4

  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=gpu_bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)
  return bars
