from __future__ import annotations
import time, functools, tinygrad.runtime.autogen.nv_regs
from typing import cast
from tinygrad.helpers import getenv, DEBUG, getbits, round_up, ceildiv
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, VirtMapping, AddrSpace, MMIOInterface
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.system import PCIDevice, USBPCIDevice, System
from tinygrad.runtime.support.usb import SLOT

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVUSBPCIDevice(USBPCIDevice):
  def __init__(self, dev, pcibus):
    super().__init__("NV", dev, pcibus, gpu_bus=2)
    if (pci_id:=self.read_config(pci.PCI_VENDOR_ID, 4)) != 0x220410DE: raise RuntimeError(f"Expected RTX 3090, found PCI ID {pci_id:#010x}")

  def reset(self) -> None:
    self.usb.set_pcie_power(False)
    self.usb.set_pcie_power(True)
    self._bar_info = System.pci_setup_usb_bars(self.usb, self.gpu_bus, mem_base=0x10000000, pref_mem_base=(32 << 30))

  def stream_gsp_boot(self, flcn:NV_FLCN, base:int) -> None:
    image = flcn.nvdev.gsp.gsp_image
    ring_start, ring_size, metadata_size = flcn.nvdev.gsp.boot_layout
    slots, total = ring_size // SLOT, ceildiv(len(image), SLOT)
    util_ctrl, rx_bytes = 0xA44, 0xA4C # NV_XVE_PCIE_UTIL_CTRL / NV_XVE_PCIE_UTIL_RX_BYTES
    mps = 128 << getbits(self.read_config(0x78 + pci.PCI_EXP_DEVCTL, 2), 5, 7)
    tlp_header = 3 * 4 # configuration requests and completions have three-dword headers
    host_bytes = 2 * tlp_header + 4 # start_cpu reads CPUCTL, then writes the start bit
    sent, armed, consumed = 0, 0, 0

    def upload(end:int):
      nonlocal sent
      while sent < end: self.usb.usb.bulk_write(image[sent*SLOT:(sent:=sent+1)*SLOT].ljust(SLOT, b'\x00'))

    self.write_config(util_ctrl, 7, 4) # enable counting and reset TX/RX byte counters before starting SEC2
    while sent < total:
      while armed < min(total, slots + consumed // SLOT):
        self.usb.usb.control_write(0xF2, value=SLOT // 512, index=(ring_start // SLOT + armed % slots) | (1 << 8))
        armed += 1
      if not sent: flcn.start_cpu(base)
      upload((end:=armed if armed == total else armed - 2) - 1) # retain an active barrier and a spare
      self.usb._f0_out(0x05, 0x0F, (self.gpu_bus << 24) | rx_bytes, 0) # queue counter read before releasing the active slot
      host_bytes += tlp_header
      if end > sent: upload(end)
      elif sent + 1 < armed and not self.usb.read(0xB296, 1)[0] & 3: upload(sent + 1)
      consumed = (((self.usb._f0_in()[0] & 0x7fffffff) * 1024 - host_bytes) * mps // (mps + tlp_header)) - metadata_size
    flcn.wait_cpu_halted(base)
    self.write_config_flush(util_ctrl, 0, 4)

class NVReg:
  def __init__(self, nvdev, base, off, fields=None): self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields

  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return NVReg(self.nvdev, base + self.base, self.off, self.fields)

  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv): self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def _is_dual_pde(self) -> bool: return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12, aperture=2 if aspace is AddrSpace.SYS else 0, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))

    if self._is_dual_pde(): self.entries[2*entry_id], self.entries[2*entry_id+1] = x & 0xffffffffffffffff, x >> 64
    else: self.entries[entry_id] = x

  def entry(self, entry_id:int) -> int:
    return (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True
  def supports_huge_page(self, paddr:int): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id:int) -> int:
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_visible_pa_allocator = None
    if self.dev.is_usb:
      pa_base, cpu_visible_limit = self.pa_allocator.base, self.dev.pci_dev.bar_info(1)[1]
      self.cpu_visible_pa_allocator = TLSFAllocator(cpu_visible_limit - pa_base, base=pa_base)
      self.pa_allocator = TLSFAllocator(self.vram_size - cpu_visible_limit, base=cpu_visible_limit)

  def palloc_cpu_visible(self, size:int, zero=True) -> int:
    paddr = cast(TLSFAllocator, self.cpu_visible_pa_allocator).alloc(round_up(size, 0x1000), 0x1000)
    if zero: self.dev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def valloc_cpu_visible(self, size:int, uncached=False, zero=True) -> VirtMapping:
    palloc = self.palloc_cpu_visible if self.cpu_visible_pa_allocator is not None else self.palloc
    va = self.alloc_vaddr(size:=round_up(size, 0x1000))
    paddr = palloc(size, zero=zero)
    return self.map_range(va, size, [(paddr, size)], aspace=AddrSpace.PHYS, uncached=uncached)

  def pfree(self, paddr:int, ptable=False):
    cpu_visible = self.cpu_visible_pa_allocator
    if cpu_visible is not None and cpu_visible.base <= paddr < cpu_visible.base + cpu_visible.size: cpu_visible.free(paddr)
    else: super().pfree(paddr, ptable)

  def on_range_mapped(self): self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')
    self.is_usb = isinstance(pci_dev, NVUSBPCIDevice)
    self.smi_dev, self.is_booting, self.is_err_state, self.finalized = False, True, False, False
    self._early_ip_init()
    self._early_mmu_init()

    # No booting state, gsp client is reinited every run.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def fini(self):
    if self.finalized: return
    self.gsp.fini_hw()
    self.finalized = True

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")

    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
      if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.", flush=True)
      self.pci_dev.reset()
      time.sleep(0.1) # wait until device can respond again

    self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)

    self.flcn:NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    self.flcn.wait_for_reset()

  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")

    # MMU Init
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE',
                                                                                f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

    self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
    self.large_bar = self.vram.nbytes >= self.vram_size

    # UVM depth   HW level                            VA bits
    # 0           PDE4                                56:56 (hopper+)
    # 1           PDE3                                55:47
    # 2           PDE2                                46:38
    # 3           PDE1 (or 512M PTE)                  37:29
    # 4           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 5           PTE_64K / PTE_4K                    20:16 / 20:12
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])

    # tail vram reserved for falcon structs
    self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
      va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not self.large_bar)

  def _alloc_boot_mem(self, size:int, data:bytes|None=None, contiguous:bool=False, sysmem:bool|None=None) -> tuple[MMIOInterface,int|None,list[int]]:
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar and not self.is_usb):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      paddr = self.mm.palloc_cpu_visible(sz) if self.is_usb else self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      sysaddr = [self.pci_dev.bar_info(1)[0] + paddr + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name:str, arch:str):
    for k,v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v
