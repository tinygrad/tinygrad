import functools, importlib, time
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import getbits, round_up
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.usb import ASMController

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

def setup_pci_bars(usb:ASMController, gpu_bus:int, mem_base:int, perf_mem_base:int):
  try: is_cfg_ok = usb.pcie_cfg_req(pci.PCI_VENDOR_ID, bus=gpu_bus, dev=0, fn=0, size=2) == 0x1002
  except RuntimeError as e:
    for bus in range(gpu_bus):
      time.sleep(0.4)
      usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
      usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
      usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

      usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=mem_base>>16, size=2)
      usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xf000, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=perf_mem_base>>16, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=mem_base>>16, size=2)

      usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
      time.sleep(0.1)
      usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=0x0, size=1)
      usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  bar_next_addr, bar_off, bars = [mem_base, perf_mem_base], 0, {}
  for bar_id in range(4):
    bar_cfg = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4)
    if (bar_cfg & pci.PCI_BASE_ADDRESS_SPACE) == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
      usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=0xffffffff, size=4)
      bar_size = 0xffffffff - (usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4) & 0xFFFFFFF0) + 1
      if bar_id in {0, 1, 3}:
        is_pref = int(bool(bar_cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH))
        usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=bar_next_addr[is_pref], size=4)
        bars[bar_off // 4] = (bar_next_addr[is_pref], bar_size)
        bar_next_addr[is_pref] += round_up(bar_size, 2 << 20)
    if bar_cfg & pci.PCI_BASE_ADDRESS_SPACE == pci.PCI_BASE_ADDRESS_MEM_TYPE_64:
      usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, dev=0, fn=0, value=0x0, size=4)
    bar_off += 8 if bar_cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64 else 4

  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=gpu_bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)
  return bars
