from __future__ import annotations
import time, functools, struct, tinygrad.runtime.autogen.nv_regs
from typing import cast
from tinygrad.helpers import getenv, DEBUG, getbits, round_up, ceildiv
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, VirtMapping, AddrSpace
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.system import PCIDevice, USBPCIDevice
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.usb import CustomASM24Controller, USBMMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVASM24Controller(CustomASM24Controller):
  def __init__(self, usb):
    super().__init__(usb)
    self.write(0xB267, b'\x08')
    self.write(0xB26F, b'\x28')

  def pcie_mem_write(self, address:int, data:bytes):
    if not data: return
    self._f0_out(0x60 if address >> 32 else 0x40, 0x0F, address, len(data) // 4, mode=1)
    self.usb.bulk_write(data)

  def pcie_mem_read(self, address:int, nbytes:int) -> memoryview:
    fmt_type = 0x20 if address >> 32 else 0x00
    if nbytes == 4: return memoryview(struct.pack("<I", self.pcie_request(fmt_type, address)))
    self._f0_out(fmt_type, 0x0F, address, nbytes // 4, mode=2)
    return self.usb.bulk_read(nbytes, timeout=30000)

class NVUSBMMIOInterface(USBMMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True, sram_start_slot:int=0):
    super().__init__(usb, addr, size, fmt, pcimem)
    self.sram_start_slot = sram_start_slot

  def __getitem__(self, index):
    off, sz = self._off_from_index(index)
    if self.pcimem:
      start, end = self.addr + off, self.addr + off + sz
      aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
      data = self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start)[start-aligned_start:end-aligned_start]
    else: data = self.usb.scsi_read(sz) if self.addr == 0xf000 else self.usb.read(self.addr + off, sz)
    if isinstance(index, slice): return data if self.fmt == 'B' else data.cast(self.fmt).tolist()
    return int.from_bytes(data, "little")

  def __setitem__(self, index, data):
    off, _ = self._off_from_index(index)
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    if not self.pcimem: self.usb.scsi_write(data, slot_start=self.sram_start_slot) if self.addr == 0xf000 else self.usb.write(self.addr + off, data)
    else:
      start, end = self.addr + off, self.addr + off + len(data)
      aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
      if start == aligned_start and end == aligned_end: aligned = data
      else:
        aligned = bytearray(self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start))
        aligned[start-aligned_start:end-aligned_start] = data
      self.usb.pcie_mem_write(aligned_start, aligned)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return NVUSBMMIOInterface(self.usb, self.addr+offset, self.nbytes-offset if size is None else size, fmt=fmt or self.fmt, pcimem=self.pcimem,
                              sram_start_slot=self.sram_start_slot)

class ASM24GSPQueueInterface(MMIOInterface):
  PAGE_SIZE, SLOT_SIZE, SRAM_SIZE = 0x1000, 0x4000, 0x80000
  PAGE_PADDRS = (0x213000, 0x27F000, 0x27B000, 0x27C000, 0x27D000, 0x27E000,
                 0x828000, 0x820000, 0x200000, 0x820000, 0x200000)

  def __init__(self, usb, size:int, fmt='B', offset:int=0, root:ASM24GSPQueueInterface|None=None):
    self.usb, self.offset, self.nbytes, self.fmt, self.el_sz = usb, offset, size, fmt, struct.calcsize(fmt)
    self._mirror:bytearray = bytearray(self.SRAM_SIZE) if root is None else root._mirror

  def __len__(self): return self.nbytes // self.el_sz

  def _off_from_index(self, index):
    if isinstance(index, slice):
      start, stop = index.start or 0, index.stop if index.stop is not None else len(self)
      return start * self.el_sz, (stop - start) * self.el_sz
    return index * self.el_sz, self.el_sz

  def _page_mapping(self, logical_page:int) -> tuple[str, int]:
    paddr = self.PAGE_PADDRS[logical_page]
    if paddr == 0x200000: return "xdata", 0xF000
    if 0x200000 <= paddr < 0x280000: return "sram", paddr - 0x200000
    return "xdata", {0x820000: 0xA000, 0x828000: 0xB800}[paddr]

  def _pieces(self, offset:int, size:int):
    end = offset + size
    while offset < end:
      page, page_off = divmod(offset, self.PAGE_SIZE)
      chunk = min(end - offset, self.PAGE_SIZE - page_off)
      kind, mapped = self._page_mapping(page)
      yield kind, mapped + page_off, chunk
      offset += chunk

  def __getitem__(self, index):
    off, size = self._off_from_index(index)
    out = bytearray()
    for kind, mapped, chunk in self._pieces(self.offset + off, size):
      out += self.usb.read(mapped, chunk) if kind == "xdata" else self._mirror[mapped:mapped+chunk]
    if isinstance(index, slice): return bytes(out) if self.fmt == 'B' else memoryview(out).cast(self.fmt).tolist()
    return int.from_bytes(out, "little")

  def __setitem__(self, index, data):
    off, size = self._off_from_index(index)
    raw = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    dirty_slots:dict[int, int] = {}
    pos = 0
    for kind, mapped, chunk in self._pieces(self.offset + off, size):
      if kind == "xdata": self.usb.write(mapped, raw[pos:pos+chunk])
      else:
        self._mirror[mapped:mapped+chunk] = raw[pos:pos+chunk]
        for slot in range(mapped // self.SLOT_SIZE, ceildiv(mapped + chunk, self.SLOT_SIZE)):
          dirty_slots[slot] = max(dirty_slots.get(slot, 0), min(mapped + chunk, (slot + 1) * self.SLOT_SIZE))
      pos += chunk

    for slot, hi in sorted(dirty_slots.items()):
      slot_base = slot * self.SLOT_SIZE
      self.usb.scsi_write(bytes(self._mirror[slot_base:round_up(hi, 512)]), slot_start=slot)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return ASM24GSPQueueInterface(self.usb, self.nbytes-offset if size is None else size, fmt or self.fmt, self.offset+offset, self)

class NVUSBPCIDevice(USBPCIDevice):
  def __init__(self, dev, pcibus):
    super().__init__("NV", dev, pcibus, controller_t=NVASM24Controller, gpu_bus=2)
    self._gsp_args:dict[int, bytes] = {}

  def function_level_reset(self) -> None:
    cap = 0x78
    command = self.read_config(pci.PCI_COMMAND, 2)
    bars = [(off, self.read_config(off, 4)) for off in (0x10, 0x18, 0x1c, 0x20)]
    devctl = self.read_config(cap + pci.PCI_EXP_DEVCTL, 2)
    self.write_config_flush(pci.PCI_COMMAND, command & ~pci.PCI_COMMAND_MASTER, 2)
    self.write_config(cap + pci.PCI_EXP_DEVCTL, devctl | pci.PCI_EXP_DEVCTL_BCR_FLR, 2)
    time.sleep(0.1)
    for off, value in bars: self.write_config(off, value, 4)
    self.write_config(cap + pci.PCI_EXP_DEVCTL, devctl, 2)
    self.write_config_flush(pci.PCI_COMMAND, command, 2)

  def dma_view(self, ctrl_addr, size, start_slot=0):
    return NVUSBMMIOInterface(self.usb, ctrl_addr, size, fmt='B', pcimem=False, sram_start_slot=start_slot)

  def alloc_gsp_queues(self, size:int) -> tuple[MMIOInterface, list[int]]:
    self.gsp_queues = ASM24GSPQueueInterface(self.usb, size)
    return self.gsp_queues, list(self.gsp_queues.PAGE_PADDRS)

  def stage_gsp_args(self, data:bytes, offset:int) -> int:
    page = data + bytes(0x100 - len(data))
    self._gsp_args[offset] = page
    self.usb.write(0xB800 + offset, page)
    return 0x828000 + offset

  def retrain_pcie(self, generation:int):
    for bus, cap in ((self.gpu_bus - 1, 0x80), (self.gpu_bus, 0x78)):
      ctl2 = self.usb.pcie_cfg_req(cap + 0x30, bus=bus, size=2)
      self.usb.pcie_cfg_req(cap + 0x30, bus=bus, value=(ctl2 & ~0xF) | generation, size=2)
      linkctl = self.usb.pcie_cfg_req(cap + 0x10, bus=bus, size=2)
      self.usb.pcie_cfg_req(cap + 0x10, bus=bus, value=linkctl & ~0x3, size=2)
    linkctl = self.usb.pcie_cfg_req(0x90, bus=self.gpu_bus - 1, size=2)
    self.usb.pcie_cfg_req(0x90, bus=self.gpu_bus - 1, value=linkctl | 0x20, size=2)
    time.sleep(0.1)

  def stream_gsp_boot(self, image:bytes, launched_at:float):
    ring_page, ring_pages, batch_pages = 44, 84, 28
    ring_size, batch_size = ring_pages * 0x1000, batch_pages * 0x1000
    for i, off in enumerate(range(ring_size, len(image), batch_size)):
      deadline = launched_at + 0.003 + (off-ring_size) / ring_size * 0.0014
      if i == 0:
        while time.perf_counter() < deadline: pass
      slot = ring_page // 4 + i % (ring_pages // batch_pages) * batch_pages // 4
      self.usb.usb.control_write(0xF2, value=batch_size // 512, index=slot | (batch_pages // 4 << 8))
      while time.perf_counter() < deadline: pass
      self.usb.usb.bulk_write(image[off:off+batch_size].ljust(batch_size, b'\x00'))
    # Restore queues and arguments after SEC2 verifies the image.
    while time.perf_counter() < launched_at + 0.270: pass
    self.usb.scsi_write(bytes(self.gsp_queues._mirror))
    for offset, page in self._gsp_args.items(): self.usb.write(0xB800 + offset, page)

  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    bar_addr, bar_size = self.bar_info(bar)
    size = bar_size - off if size is None else size
    return NVUSBMMIOInterface(self.usb, bar_addr + off, size, fmt)

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
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000) # global for all devices.

  def __init__(self, *args, cpu_visible_limit:int|None=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_visible_pa_allocator = None
    if cpu_visible_limit is not None:
      pa_base = self.pa_allocator.base
      self.cpu_visible_pa_allocator = TLSFAllocator(cpu_visible_limit - pa_base, base=pa_base)
      self.pa_allocator = TLSFAllocator(self.vram_size - cpu_visible_limit, base=cpu_visible_limit)

  def palloc_cpu_visible(self, size:int, align:int=0x1000, zero=True) -> int:
    paddr = cast(TLSFAllocator, self.cpu_visible_pa_allocator).alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def valloc_cpu_visible(self, size:int, align=0x1000, uncached=False, zero=True) -> VirtMapping:
    palloc = self.palloc_cpu_visible if self.cpu_visible_pa_allocator is not None else self.palloc
    va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)
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
    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()

    # No booting state, gsp client is reinited every run.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def _recover_stale_wpr(self):
    cast(NVUSBPCIDevice, self.pci_dev).function_level_reset()
    self.mmio = self.pci_dev.map_bar(0, fmt='I')
    self.flcn.wait_for_reset()

  def fini(self):
    if not self.is_usb:
      self.gsp.fini_hw()
      return
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() == 0: return

    self.gsp.fini_hw()
    cast(NV_FLCN, self.flcn).shutdown_booter()

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")

    recover_wpr = False
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      if self.is_usb: recover_wpr = True
      else:
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

    if recover_wpr: self._recover_stale_wpr()
    else: self.flcn.wait_for_reset()

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
    cpu_visible_limit = self.pci_dev.bar_info(1)[1] if self.is_usb else None
    self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
      va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not self.large_bar,
      cpu_visible_limit=cpu_visible_limit)

  def _alloc_boot_mem(self, size:int, data:bytes|None=None, contiguous:bool=False, sysmem:bool|None=None) -> tuple[MMIOInterface,int|None,list[int]]:
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar and not self.is_usb):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      cpu_visible = self.is_usb
      paddr = self.mm.palloc_cpu_visible(sz) if cpu_visible else self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      base = paddr if cpu_visible else self.pci_dev.bar_info(1)[0] + paddr
      sysaddr = [base + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name:str, arch:str):
    for k,v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v
