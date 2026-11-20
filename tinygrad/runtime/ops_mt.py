from __future__ import annotations
import ctypes, dataclasses, os, pathlib, struct, subprocess, tempfile, time
from typing import Any, cast
from tinygrad.device import Compiled, Allocator, BufferStorage, BufferSpec, Buffer
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.renderer.cstyle import CUDARenderer, CStyleLanguage
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG, pluralize, round_up
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, KernelInfo
from tinygrad.runtime.support.hcq2 import HWQueue, patch, get_time_ms, to_name, ccall
from tinygrad.runtime.support.usb import USB3, usb_link, usb_stream, usb_ctrl, usb_bulk, usb_stack, usb_fail, usb_poke, HOST_SIZE
from tinygrad.runtime.support.system import PCIDevice, USBPCIDevice, System, filter_visible_devices
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.runtime.autogen import mtfw, mtregs, mtpds, libc

def mt_off(typ, name:str) -> int: return next(f[2] for f in typ._real_fields_ if f[0] == name)

# **** mtdev ****

MT_VENDOR_ID = 0x1ed5
MT_USB_GPU_BUS = 2
MT_SYS_BAR, MT_DDR_BAR = 0, 2
MT_SRAM_OFFSET = 0x30000
MT_DISTRIBUTOR_OFFSET = 0x83000
MT_SOCIF_OFFSET, MT_CORE_STRIDE, MT_CORE_COUNT = 0xc00000, 0x100000, 8
MT_CORE_ID = 0x20

MT_IPC_TX_RING, MT_IPC_RX_RING = MT_SRAM_OFFSET, MT_SRAM_OFFSET + 0x218
MT_IPC_IRQ = 2
MT_IPC_NODE_SMC, MT_IPC_NODE_PCIE = 0, 2
MT_IPC_EVENT_GET_FW_VERSION = 11
MT_IPC_EVENT_SET_POWER_STATE = 91
MT_PSTATE_P0, MT_PSTATE_P12 = 0, 12


def _ipc_header(event_id:int, data_size:int, msg_id:int=0, msg_type:int=0, event_type:int=0,
                target:int=MT_IPC_NODE_SMC, response:bool=True) -> int:
  return (event_type | (event_id << 5) | (msg_id << 14) | (msg_type << 22) | (int(response) << 25) |
          (MT_IPC_NODE_PCIE << 29) | (target << 37) | (data_size << 45))


class _Ring:
  def __init__(self, smc:MTSMC, ctrl:int):
    self.smc, self.ctrl = smc, ctrl
    self.size = smc._r32(ctrl + 0x0c)
    self.data = smc._r32(ctrl + 0x10)
    if self.size == 0 or self.size & 3: raise RuntimeError(f"Invalid MT IPC ring size {self.size:#x} at {ctrl:#x}")
    if (self.data & 0xfff) != ((ctrl + 0x18) & 0xfff): raise RuntimeError(f"Invalid MT IPC ring data pointer {self.data:#x}")

  def _data_len(self) -> int:
    rd, wr = self.smc._r32(self.ctrl), self.smc._r32(self.ctrl + 4)
    rp, wp = rd >> 1, wr >> 1
    if ((wr ^ rd) & ~1) == 0: return self.size if (rd ^ wr) & 1 else 0
    return wp - rp if rp < wp else self.size - (rp - wp)

  def _advance(self, ptr:int, nbytes:int) -> int:
    pos, parity = ptr >> 1, ptr & 1
    if nbytes >= self.size - pos: return ((nbytes - (self.size - pos)) * 2) | (parity ^ 1)
    return ((pos + nbytes) * 2) | parity

  def read(self, pos:int, nbytes:int) -> bytes:
    first = min(nbytes, self.size - pos)
    return self.smc._read(self.data + pos, first) + (self.smc._read(self.data, nbytes - first) if first != nbytes else b'')

  def write(self, payload:bytes):
    if len(payload) > self.size - self._data_len(): raise RuntimeError("MT IPC ring is full")
    wr = self.smc._r32(self.ctrl + 4)
    pos = wr >> 1
    first = min(len(payload), self.size - pos)
    self.smc._write(self.data + pos, payload[:first])
    if first != len(payload): self.smc._write(self.data, payload[first:])
    self.smc._w32(self.ctrl + 4, self._advance(wr, len(payload)))

  def read_message(self) -> tuple[int, bytes, int]:
    rd = self.smc._r32(self.ctrl)
    header = struct.unpack('<Q', self.read(rd >> 1, 8))[0]
    data_size = (header >> 45) & 0xff
    if not 1 <= data_size <= 64: raise RuntimeError(f"Invalid MT IPC response size {data_size} dwords")
    nbytes = 8 + data_size * 4
    payload = self.read(rd >> 1, nbytes)[8:]
    return header, payload, self._advance(rd, nbytes)


class MTSMC:
  def __init__(self, mmio:MMIOInterface):
    self.mmio = mmio
    self.tx, self.rx = _Ring(self, MT_IPC_TX_RING), _Ring(self, MT_IPC_RX_RING)
    self.msg_id = 0

  def _read(self, offset:int, nbytes:int) -> bytes:
    # The ASM24 bridge returns short reads for some MT register windows above 64 bytes.
    return b''.join(bytes(self.mmio[off:off + min(64, offset + nbytes - off)]) for off in range(offset, offset + nbytes, 64))

  def _write(self, offset:int, data:bytes):
    assert offset % 4 == 0 and len(data) % 4 == 0
    self.mmio[offset:offset + len(data)] = data

  def _r32(self, offset:int) -> int: return struct.unpack('<I', self._read(offset, 4))[0]
  def _w32(self, offset:int, value:int): self._write(offset, struct.pack('<I', value))

  def _kick(self, event_id:int, target:int, msg_id:int):
    # GIC SGIR: write-1-to-set, target list, event ID and SGI interrupt source.
    value = 0x40000 | 1 | (0x100 << target) | ((event_id * 2) & 0xff)
    self._w32(MT_DISTRIBUTOR_OFFSET + 0x1f8 + MT_IPC_IRQ * 4, value)

  def _receive(self, old_wr:int, msg_id:int, event_id:int, timeout:float) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if (wr:=self._r32(self.rx.ctrl + 4)) != old_wr:
        header, payload, next_rd = self.rx.read_message()
        self._w32(self.rx.ctrl, next_rd)
        old_wr = wr
        if (header >> 14) & 0xff != msg_id or (header >> 5) & 0x7f != event_id: continue  # asynchronous message
        if not (header >> 22) & 1: raise RuntimeError("MT IPC response has the request bit clear")
        if (error := (header >> 53) & 3): raise RuntimeError(f"MT IPC event {event_id} failed with error {error}")
        return payload
      time.sleep(0.001)
    raise TimeoutError(f"MT IPC event {event_id} timed out")

  def call(self, event_id:int, payload:bytes, timeout=5.0) -> bytes:
    assert len(payload) % 4 == 0 and 4 <= len(payload) <= 256
    msg_id, self.msg_id = self.msg_id, (self.msg_id + 1) & 0xff
    header = _ipc_header(event_id, len(payload) // 4, msg_id=msg_id)
    old_wr = self._r32(self.rx.ctrl + 4)
    self.tx.write(struct.pack('<Q', header) + payload)
    self._kick(event_id, MT_IPC_NODE_SMC, msg_id)
    return self._receive(old_wr, msg_id, event_id, timeout)

  def rtos_version(self) -> list[tuple[str, int]]:
    payload = self.call(MT_IPC_EVENT_GET_FW_VERSION, bytes(108))
    if len(payload) % 12: raise RuntimeError("Invalid MT RTOS version response")
    return [(payload[off:off + 8].rstrip(b'\0').decode('ascii'), struct.unpack_from('<I', payload, off + 8)[0])
            for off in range(0, len(payload), 12)]

  def set_pstate(self, level:int):
    # OS (Linux), requested P-state, reserved.
    return self.call(MT_IPC_EVENT_SET_POWER_STATE, struct.pack('<III', 0, level, 0))


class MTDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    self.mmio = pci_dev.map_bar(MT_SYS_BAR, fmt='B')
    self.vram_size = pci_dev.bar_info(MT_DDR_BAR)[1]
    self.smc = MTSMC(self.mmio)
    self.fw_versions = self.smc.rtos_version()
    if DEBUG >= 1: print(f"mt {self.devfmt}: firmware {self.fw_versions}")
    self.smc.set_pstate(MT_PSTATE_P0)
    self.core_ids = self._read_core_ids()
    if not any(self.core_ids): raise RuntimeError("MT GPU cores did not power on")
    if DEBUG >= 1: print(f"mt {self.devfmt}: {len(self.core_ids)} cores, BVNC {self.bvnc}, VRAM {self.vram_size >> 30} GiB")

  def _read_core_ids(self) -> list[int]:
    return [struct.unpack('<Q', self.smc._read(MT_SOCIF_OFFSET + core * MT_CORE_STRIDE + MT_CORE_ID, 8))[0]
            for core in range(MT_CORE_COUNT)]

  @property
  def bvnc(self) -> tuple[int, int, int, int]:
    value = self.core_ids[0]
    return ((value >> 48) & 0xffff, (value >> 32) & 0xffff, (value >> 16) & 0xffff, value & 0xffff)

  def close(self): self.smc.set_pstate(MT_PSTATE_P12)

  @classmethod
  def open_usb(cls, dev_id=0) -> MTDev:
    devices = filter_visible_devices(USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001), "MT")
    if dev_id >= len(devices): raise RuntimeError(f"MT:{dev_id} does not exist ({pluralize('device', len(devices))} available)")
    dev, pcibus = devices[dev_id]
    return cls(USBPCIDevice("MT", dev, pcibus, gpu_bus=MT_USB_GPU_BUS))

  @classmethod
  def open_pci(cls, dev_id=0) -> MTDev:
    pci_dev = System.pci_probe_device("MT", dev_id, MT_VENDOR_ID, ((0xff00, (0x0200, 0x0300)),), base_class=0x03)
    return cls(pci_dev)

# **** mmu ****

MT_PAGE_SIZE = 0x1000
MT_PTE_READ_ONLY = 0x3c00000000000001
MT_PTE_READ_WRITE = 0x3c0000000000000d

class MTMMU4:
  """Rogue MMU v4 page tables used by the MTT S80 (BVNC 35.4.1632.23)."""
  def __init__(self, root_pa:int, size:int=0x1000000):
    self.root_pa, self.root, self.next = root_pa, bytearray(size), 0x1000
    self.pd:dict[int,int] = {}
    self.pt:dict[tuple[int,int],int] = {}
    self.pages:dict[int,int] = {}
    self.page_flags:dict[int,int] = {}
    self.dirty_pages:set[int] = set()
    self.mapped_ranges:set[tuple[int,int,int,int]] = set()

  def _alloc(self) -> int:
    if self.next + MT_PAGE_SIZE > len(self.root): raise MemoryError("MT MMU root allocation exhausted")
    pa = self.next
    self.next += MT_PAGE_SIZE
    return self.root_pa + pa

  def map(self, va:int, pa:int, flags:int=MT_PTE_READ_ONLY) -> None:
    if va % MT_PAGE_SIZE or pa % MT_PAGE_SIZE: raise ValueError("MT MMU mappings must be 4 KiB aligned")
    if flags & 1 != 1: raise ValueError("MT MMU mapping must be present")
    if self.pages.get(va) == pa and self.page_flags.get(va) == flags: return
    if va in self.pages:
      self.mapped_ranges = {r for r in self.mapped_ranges if not (r[0] <= va < r[0]+r[2])}
    self.pages[va],self.page_flags[va] = pa,flags
    pci, pdi, pti = (va >> 30) & 0x3ff, (va >> 21) & 0x1ff, (va >> 12) & 0x1ff
    if pci not in self.pd:
      self.pd[pci] = self._alloc()
      self.root[pci*4:pci*4+4] = (((self.pd[pci] >> 12) << 4) | 1).to_bytes(4, 'little')
      self.dirty_pages.add((pci*4) & ~0xfff)
    if (pci, pdi) not in self.pt:
      self.pt[pci,pdi] = self._alloc()
      off = self.pd[pci] - self.root_pa + pdi * 8
      self.root[off:off+8] = (self.pt[pci,pdi] | 1).to_bytes(8, 'little')
      self.dirty_pages.add(off & ~0xfff)
    off = self.pt[pci,pdi] - self.root_pa + pti * 8
    self.root[off:off+8] = ((pa & 0xfffffffff000) | flags).to_bytes(8, 'little')
    self.dirty_pages.add(off & ~0xfff)

  def map_range(self, va:int, pa:int, size:int, flags:int=MT_PTE_READ_ONLY) -> None:
    if size % MT_PAGE_SIZE: raise ValueError("MT MMU range size must be 4 KiB aligned")
    key = (va,pa,size,flags)
    if key in self.mapped_ranges: return
    for off in range(0, size, MT_PAGE_SIZE): self.map(va + off, pa + off, flags)
    self.mapped_ranges.add(key)

# **** elf ****

class _MessagePack:
  def __init__(self, data:bytes): self.data, self.off = data, 0
  def take(self, n:int) -> bytes:
    if self.off + n > len(self.data): raise ValueError("Truncated MTGPU MessagePack data")
    ret = self.data[self.off:self.off+n]
    self.off += n
    return ret
  def uint(self, n:int) -> int: return int.from_bytes(self.take(n), 'big')
  def sint(self, n:int) -> int: return int.from_bytes(self.take(n), 'big', signed=True)
  def unpack(self):
    tag = self.uint(1)
    if tag <= 0x7f: return tag
    if tag >= 0xe0: return tag - 0x100
    if 0x80 <= tag <= 0x8f: return {self.unpack():self.unpack() for _ in range(tag & 0xf)}
    if 0x90 <= tag <= 0x9f: return [self.unpack() for _ in range(tag & 0xf)]
    if 0xa0 <= tag <= 0xbf: return self.take(tag & 0x1f).decode()
    if tag == 0xc0: return None
    if tag in (0xc2,0xc3): return tag == 0xc3
    if tag == 0xc4: return self.take(self.uint(1))
    if tag == 0xc5: return self.take(self.uint(2))
    if tag == 0xc6: return self.take(self.uint(4))
    if tag == 0xd9: return self.take(self.uint(1)).decode()
    if tag == 0xda: return self.take(self.uint(2)).decode()
    if tag == 0xdb: return self.take(self.uint(4)).decode()
    if tag == 0xcc: return self.uint(1)
    if tag == 0xcd: return self.uint(2)
    if tag == 0xce: return self.uint(4)
    if tag == 0xcf: return self.uint(8)
    if tag == 0xd0: return self.sint(1)
    if tag == 0xd1: return self.sint(2)
    if tag == 0xd2: return self.sint(4)
    if tag == 0xd3: return self.sint(8)
    if tag == 0xdc: return [self.unpack() for _ in range(self.uint(2))]
    if tag == 0xdd: return [self.unpack() for _ in range(self.uint(4))]
    if tag == 0xde: return {self.unpack():self.unpack() for _ in range(self.uint(2))}
    if tag == 0xdf: return {self.unpack():self.unpack() for _ in range(self.uint(4))}
    raise ValueError(f"Unsupported MTGPU MessagePack tag {tag:#x}")

@dataclasses.dataclass(frozen=True)
class MTImmediate:
  base: int
  offset: int
  value: int

@dataclasses.dataclass(frozen=True)
class MTLaunchMetadata:
  entry: int
  immediates: tuple[MTImmediate, ...]
  kernel: dict

  @classmethod
  def from_note(cls, note:bytes) -> MTLaunchMetadata:
    if len(note) < 0x14: raise ValueError("Invalid MTGPU note")
    namesz, descsz, _ = struct.unpack_from('<III', note)
    name_off = 0x0c
    desc_off = (name_off + namesz + 3) & ~3
    if note[name_off:name_off+namesz].rstrip(b'\0') != b'MTGPU': raise ValueError("Invalid MTGPU note name")
    metadata = _MessagePack(note[desc_off:desc_off+descsz]).unpack()
    kernels = metadata['mtgpu.kernels']
    if len(kernels) != 1: raise ValueError("Expected one MTGPU kernel")
    kernel = kernels[0]
    immediates = []
    for item in kernel.get('.imms', kernel.get('imms', [])):
      base_data = item.get('.immBase', item.get('immBase', {}))
      offset = base_data.get('.offset', base_data.get('offset', item.get('.offset', item.get('offset'))))
      value = base_data.get('.value', base_data.get('value', item.get('.value', item.get('value'))))
      immediates.append(MTImmediate(int(item.get('.base', item.get('base', 0))), int(offset), int(value)))
    return cls(0, tuple(immediates), kernel)

@dataclasses.dataclass(frozen=True)
class MTKernelImage:
  entry: int
  text: bytes
  metadata: MTLaunchMetadata

  @classmethod
  def from_elf(cls, blob:bytes) -> MTKernelImage:
    _, sections, _ = elf_loader(blob)
    text = next(s for s in sections if s.name == '.text')
    note = next(s for s in sections if s.name == '.note')
    metadata = MTLaunchMetadata.from_note(note.content)
    return cls(int(text.header.sh_addr), text.content, dataclasses.replace(metadata, entry=int(text.header.sh_addr)))

# **** pds ****

# Rogue PDS instructions used by compute launches. These are the same encodings
# as pvr_pds_compute_shader(), restricted to features exposed by MTGPU metadata.
PDS_UNUSED = 0xffffffff

@dataclasses.dataclass(frozen=True)
class MTPDSProgram:
  code: bytes
  data: bytes
  temps_used: int = 3

  @classmethod
  def from_words(cls, code:list[int], data:list[int]) -> MTPDSProgram:
    return cls(struct.pack(f'<{len(code)}I',*code),struct.pack(f'<{len(data)}I',*data))

def _bra(addr:int) -> int:
  return mtpds.PVR_ROGUE_PDSINST_BRA_OPCODE_DEFAULT | (2<<mtpds.PVR_ROGUE_PDSINST_BRA_SRCC_SHIFT) | \
    mtpds.PVR_ROGUE_PDSINST_BRA_NEG_ENABLE | (8<<mtpds.PVR_ROGUE_PDSINST_BRA_SETC_SHIFT) | addr

def _dout(dst:int, src0:int=0, src1:int=0, cc:int=0, end:int=0) -> int:
  return mtpds.PVR_ROGUE_PDSINST_DOUT_OPCODE_DEFAULT | (cc<<mtpds.PVR_ROGUE_PDSINST_DOUT_CC_SHIFT) | \
    (end<<mtpds.PVR_ROGUE_PDSINST_DOUT_END_SHIFT) | (src1<<mtpds.PVR_ROGUE_PDSINST_DOUT_SRC1_SHIFT) | \
    (src0<<mtpds.PVR_ROGUE_PDSINST_DOUT_SRC0_SHIFT) | dst

def _doutu(src0:int=0, cc:int=0, end:int=1, flags:int=0) -> int:
  return _dout(mtpds.PVR_ROGUE_PDSINST_DOUT_DST_DOUTU,src0,cc=cc,end=end) | flags

def _doutw(src1:int, src0:int, end:bool=False) -> int: return _dout(mtpds.PVR_ROGUE_PDSINST_DOUT_DST_DOUTW,src0,src1,end=end)
def _doutd(src0:int, cc:int=0, end:int=0) -> int: return _dout(mtpds.PVR_ROGUE_PDSINST_DOUT_DST_DOUTD,src0,cc=cc,end=end)
def _doutc(src0:int) -> int: return _dout(5,src0,end=1)  # DOUTC, conditional in the Mesa headers

def _sp(op:int, cc:int=0, payload:int=0) -> int:
  return (mtpds.PVR_ROGUE_PDSINST_OPCODEC_SP<<28) | (op<<23) | (cc<<27) | payload

def _release() -> int: return _sp(mtpds.PVR_ROGUE_PDSINST_OPCODESP_RELEASE)
def _halt() -> int: return _sp(mtpds.PVR_ROGUE_PDSINST_OPCODESP_HALT)
def _ld(src0:int=0, flags:int=0) -> int: return _sp(mtpds.PVR_ROGUE_PDSINST_OPCODESP_LD,payload=src0|flags)
def _st(src0:int=0) -> int: return _sp(mtpds.PVR_ROGUE_PDSINST_OPCODESP_ST,payload=src0)
def _wdf(payload:int) -> int: return _sp(mtpds.PVR_ROGUE_PDSINST_OPCODESP_WDF,payload=payload)
def _vendor_sp9(cc:int, payload:int) -> int: return _sp(9,cc,payload)  # generation-1 resume ABI; not defined by Mesa

def _sftlp64(cc:int, lop:int, im:int, src0:int, src1:int, src2:int, dst:int) -> int:
  return mtpds.PVR_ROGUE_PDSINST_SFTLP64_OPCODE_DEFAULT | (cc<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_CC_SHIFT) | \
    (lop<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_LOP_SHIFT) | (im<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_IM_SHIFT) | \
    (src0<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_SRC0_SHIFT) | (src1<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_SRC1_SHIFT) | \
    (src2<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_SRC2_SHIFT) | (dst<<mtpds.PVR_ROGUE_PDSINST_SFTLP64_DST_SHIFT)

def _cmp(cc:int, cop:int, setcp:int, im:int, src0:int, src1:int) -> int:
  return mtpds.PVR_ROGUE_PDSINST_CMP_OPCODE_DEFAULT | (cc<<mtpds.PVR_ROGUE_PDSINST_CMP_CC_SHIFT) | \
    (cop<<mtpds.PVR_ROGUE_PDSINST_CMP_COP_SHIFT) | (setcp<<mtpds.PVR_ROGUE_PDSINST_CMP_SETCP_SHIFT) | \
    (im<<mtpds.PVR_ROGUE_PDSINST_CMP_IM_SHIFT) | (src0<<mtpds.PVR_ROGUE_PDSINST_CMP_SRC0_SHIFT) | (src1<<mtpds.PVR_ROGUE_PDSINST_CMP_SRC1_SHIFT)

def _ddmad(cc:int, end:int, src0:int, src1:int, src2:int, src3:int) -> int:
  return mtpds.PVR_ROGUE_PDSINST_DDMAD_OPCODE_DEFAULT | (cc<<mtpds.PVR_ROGUE_PDSINST_DDMAD_CC_SHIFT) | \
    (end<<mtpds.PVR_ROGUE_PDSINST_DDMAD_END_SHIFT) | (src0<<mtpds.PVR_ROGUE_PDSINST_DDMAD_SRC0_SHIFT) | \
    (src1<<mtpds.PVR_ROGUE_PDSINST_DDMAD_SRC1_SHIFT) | (src2<<mtpds.PVR_ROGUE_PDSINST_DDMAD_SRC2_SHIFT) | \
    (src3<<mtpds.PVR_ROGUE_PDSINST_DDMAD_SRC3_SHIFT)

def _control(dest:int, part:str, store:str, last:bool) -> int:
  masks = {'lower32':0,'upper32':1,'lower64':2}
  stores = {'unified':0,'common':1}
  return (dest<<13) | masks[part] | (stores[store]<<28) | (int(last)<<31)

def _reg_list(regs:tuple[int,int,int]) -> list[int]: return [i for i,reg in enumerate(regs) if reg != PDS_UNUSED]

def compute_pds(execution_address:int, local_regs:tuple[int,int,int]=(PDS_UNUSED,)*3,
                workgroup_regs:tuple[int,int,int]=(PDS_UNUSED,)*3, barrier_coefficient:int=PDS_UNUSED,
                coefficient_update:bool=False, prefix:tuple[int,...]=()) -> MTPDSProgram:
  """Build compute PDS code/data for one MTGPU kernel.

  Constant zero is the 64-bit USC task-control word. ID control words follow
  it in the same order as their DOUTW instructions. Workgroup/local IDs are
  sourced from CDM temp registers 64 and 65, as in pvr_pds_compute_shader().
  """
  code, data = [_bra(2)], list(prefix) or [execution_address & 0xffffffff,execution_address >> 32]
  local, workgroup = _reg_list(local_regs), _reg_list(workgroup_regs)
  if coefficient_update: data += [execution_address & 0xffffffff,execution_address >> 32]

  if barrier_coefficient != PDS_UNUSED:
    code.append(_doutw(4,1))  # barrier control at dword 4, zero64 at dword 2
    data += [0,0,_control(barrier_coefficient,'lower64','common',not workgroup)]

  if 0 in workgroup and 1 in workgroup:
    code.append(_doutw(len(data),64))
    data.append(_control(workgroup_regs[0],'lower64','common',2 not in workgroup))
  else:
    for i in (0,1):
      if i in workgroup:
        code.append(_doutw(len(data),64))
        data.append(_control(workgroup_regs[i],'lower32' if i == 0 else 'upper32','common',2 not in workgroup))
  if 2 in workgroup:
    code.append(_doutw(len(data),65))
    data.append(_control(workgroup_regs[2],'upper32','common',True))

  if coefficient_update: code.append(_doutu(1))
  code.append(_halt())
  branch_target = len(code)
  if local:
    if 0 in local and (1 in local or 2 in local): dest,part = local_regs[0],'lower64'
    elif 0 in local: dest,part = local_regs[0],'lower32'
    else: dest,part = local_regs[1] if 1 in local else local_regs[2],'upper32'
    code.append(_doutw(len(data),64))
    data.append(_control(dest,part,'unified',True))

  code[0] = _bra(branch_target)
  code += [_doutu(),_halt()]
  return MTPDSProgram.from_words(code,data)

# **** submission ****

SUBMISSION_VA = 0x8000420000
ARGS_VA = 0x81e1200000
GRID_VA = 0x81e1400000
RESOURCE_VA = 0xda00406000
TEXT_VA = 0xe000000000
RESUME_VA = 0xda00000000
PDS_MAIN_VA = 0xda00400000
TASK_VA = 0x100804000
CDM_EXECUTION_ADDRESS = 0x81e0e00000

def build_submission(launch_id:int=1, execution_address:int=CDM_EXECUTION_ADDRESS) -> bytes:
  """Build one compute submission with its CDM kick and error region."""
  head = mtfw.MTFW_SUBMISSION_HEAD()
  head.timestampAddr, head.fwCswBufer = 0x8000438200, 0x6800000000
  head.retryCount, head.subRegionDescStartOffset, head.subRegionDescNums = 0xffffffff, mtfw.MTFW_SUBMISSION_HEAD.SIZE, 2
  head.cbId = launch_id

  cdm = mtfw.MTFW_SUBMISSION_REGION_DESC()
  cdm.regionType, cdm.regionStart.offset, cdm.regionSize = mtfw.MTFW_SUB_REGION_CDM, 0xa0, 0x198
  error = mtfw.MTFW_SUBMISSION_REGION_DESC()
  error.regionType, error.regionStart.gpuAddr, error.regionSize = mtfw.MTFW_SUB_REGION_ERROR_BUFFER, execution_address, 8

  kick = bytearray(0x1a8)
  struct.pack_into('<QQ',kick,0x18,0x1800000000,0x2000000001)
  struct.pack_into('<Q',kick,0x20,0)
  struct.pack_into('<Q',kick,0x28,0x2000000000)
  struct.pack_into('<I',kick,0x30,1)
  struct.pack_into('<I',kick,0x38,0)
  struct.pack_into('<Q',kick,0x40,0x30d4000000178)
  struct.pack_into('<I',kick,0x48,0x14)
  struct.pack_into('<Q',kick,0xf8,0xff00000000)
  struct.pack_into('<Q',kick,0x118,0x10000001f0)
  struct.pack_into('<I',kick,0x120,2)
  struct.pack_into('<I',kick,0x128,0)
  struct.pack_into('<Q',kick,0x170,0x81e0840000)
  struct.pack_into('<Q',kick,0x178,0x8000420300)
  struct.pack_into('<Q',kick,0x180,0x8000420380)
  struct.pack_into('<I',kick,0x188,0x2c)
  struct.pack_into('<Q',kick,0x190,0x8000428300)
  struct.pack_into('<Q',kick,0x198,0x207820c000)
  struct.pack_into('<I',kick,0x1a0,0x200)

  raw = bytearray(bytes(head))
  struct.pack_into('<Q',raw,0x8,0)
  struct.pack_into('<Q',raw,0x10,0x8000438200)
  struct.pack_into('<Q',raw,0x18,0x6800000000)
  packet = bytearray((bytes(raw)+bytes(cdm)+bytes(error)+kick+bytes(8)).ljust(0x19250,b'\0'))
  struct.pack_into('<QQQQQQ',packet,0x300,0xe140000040010081,0x3606c1400002,0x3601,0,0x2000000,0x80000000)
  for off in range(0x380,0x1000,0x80): struct.pack_into('<Q',packet,off,0xb0)
  return bytes(packet)

# **** compute ****

FW_BASE, ROOT_BASE, INIT_SIZE, FW_DEVVA = 0x20000000, 0x30000000, 0x2000000, 0xe1c0000000
FWIF_OFF, META_BASE = 0xc0000, 0x90000000
SUB_PA, SUB_VA = FW_BASE + INIT_SIZE, FW_DEVVA + INIT_SIZE
FIRMWARE_PATHS = ('/lib/firmware/mthreads/mtfw-gen1.bin', '/opt/musa/lib/firmware/mthreads/mtfw-gen1.bin')
DEFAULT_FIRMWARE = next((path for path in FIRMWARE_PATHS if pathlib.Path(path).is_file()), FIRMWARE_PATHS[0])

class MTFirmwareLayout:
  def __init__(self, core_count:int=8):
    self.reg_dump_buffer_size = (core_count * 40 + 24 + 8) * 64
    self.vgpu_shared = (0xfffed2e0 + 0x97730 + core_count * 40) & 0xffffffff
    self.trace = (0xfffed2e0 + 0x26e10 + 0x40ee0 + FWIF_OFF) & 0xffffffff
    self.perf = FWIF_OFF + ((0xfffed2e0 + 0x26e10 + 0x70a50) & 0xffffffff)
    self.fault = self.perf + 0xe6018
    self.dump = self.fault + 0x88218
    self.regcfg = self.fault + 0xee240
    self.regbuf = self.regcfg + 0x30
    self.item = self.regbuf + self.reg_dump_buffer_size
    self.gpu = self.item + self.vgpu_shared
    self.watchdog = self.gpu + 0x4c0
    self.ccb = FWIF_OFF + 0x40ee0

def build_firmware(lay:MTFirmwareLayout) -> bytes:
  fw = mtfw.struct_MTFW_FWIF_TAG()
  fw.watchDog = META_BASE + lay.watchdog
  fw.ccbQueueArrayAddr, fw.ccbQueueArrayNum = META_BASE + lay.ccb, mtfw.MTFW_CCB_TYPE_MAX_NUM
  fw.traceBufAddr = META_BASE + lay.trace
  fw.metaPerfEventBufCtrlArray, fw.metaPerfEventBufCtrlNum = META_BASE + lay.perf, 1
  fw.faultDumpBuffer, fw.gpuUtilBufAddr = META_BASE + lay.fault, META_BASE + lay.gpu
  fw.socTimerFreq, fw.gpuFrequence = 25, 1600000000
  cfg = fw.fwCfgs.cfg
  for name,count,base in [('flStateAddrsSCG',2,0),('flStateAddrsMCG2',4,0x200),('flStateAddrsMCG3',6,0x600),
                          ('flStateAddrsMCG4',8,0xc00),('flStateAddrsMCG5',10,0x1400),('flStateAddrsMCG6',12,0x1e00),
                          ('flStateAddrsMCG7',14,0x2a00),('flStateAddrsMCG8',16,0x3800)]:
    for i in range(count): getattr(cfg, name)[i] = 0xd9bfff0000 | base + i * 0x100
  cfg.yuvCoefficientAddr = 0xe0ffe00000
  cfg.pdsHeapBaseAddr, cfg.uscHeapBaseAddr = 0xda00000000, 0xe000000000
  cfg.fbcdcHeapBaseAddr, cfg.fbcdcLargeHeapBaseAddr = 0xec00000000, 0xec40000000
  cfg.texStateHeapBaseAddr, cfg.componentCtrlHeapBaseAddr = 0xb000000000, 0xeb00000000
  cfg.dmKillShaderOffset = 0xfff80000
  return bytes(fw)

# Values programmed by MUSA 3.3.0's meta_fw_reset_sequence() and RGXStart() for BVNC 35.4.1632.23.
# These are device-specific reset/clock programs, not constants present in the register headers.
MT_MERCER_RESET_ASSERT = (0x1249249249249249,0x36db6db6db6db6db,mtregs.RGX_CR_MERCER_SOFT_RESET_MASKFULL)
MT_MERCER_RESET_DEASSERT = (0x36db6db6db6db6db,0x1249249249249249,0)
MT_SOFT_RESET_SEQUENCE = (0x386000000800,0x3860000088f2,0x20000080f2,mtregs.RGX_CR_SOFT_RESET_GARTEN_EN)
MT_CLK_CTRL0_PRE_JONES, MT_CLK_CTRL1_PRE_JONES = 0x5545010155151101, 0x5554151555455555
MT_CLK_CTRL0_POST_JONES, MT_CLK_CTRL1_POST_JONES = 0xaa8a0202aa2a2202, 0xaaa82a2aaa8aaaaa
MT_CLK_CTRL0_UPDATE_MASK, MT_CLK_CTRL0_UPDATE_BITS = 0xfcfffffff3fffcff, 0x100000004000100
MT_CLK_CTRL1_UPDATE_MASK, MT_CLK_CTRL1_UPDATE_BITS = 0xffffffffcf3fffff, 0x10400000
MT_MMU_PAGE_RANGES = ((mtregs.RGX_CR_MMU_PAGE_SIZE_RANGE_ONE,0),(mtregs.RGX_CR_MMU_PAGE_SIZE_RANGE_TWO,0xfff80800),
                      (mtregs.RGX_CR_MMU_PAGE_SIZE_RANGE_THREE,0),(mtregs.RGX_CR_MMU_PAGE_SIZE_RANGE_FOUR,0xfff80000))
MT_ACE_PROT_CTRL = 0
for _name, _value in mtregs.__dict__.items():
  if _name.startswith('RGX_CR_ACE_PROT_CTRL_OSID') and _name.endswith('_UNPRIVILEGED_NON_SECURE'): MT_ACE_PROT_CTRL |= _value
MT_ACE_CTRL = ((0xf << mtregs.RGX_CR_ACE_CTRL_PM_MMU_AXCACHE_SHIFT) | mtregs.RGX_CR_ACE_CTRL_MMU_AWCACHE_WRITE_BACK_WRITE_ALLOCATE |
               mtregs.RGX_CR_ACE_CTRL_MMU_ARCACHE_WRITE_BACK_READ_ALLOCATE | mtregs.RGX_CR_ACE_CTRL_COH_DOMAIN_OUTER_SHAREABLE)
MT_FW_STARTED_MAGIC = 0x4d

class MTFirmware:
  def __init__(self, dev:MTDev, path:str|None=None, core_count:int=8, prepare:bool=True, mmu:MTMMU4|None=None, v=None):
    self.dev, self.path, self.core_count = dev, path or os.environ.get('MT_FIRMWARE', DEFAULT_FIRMWARE), core_count
    if prepare and not pathlib.Path(self.path).is_file():
      raise FileNotFoundError(f'MT firmware not found at {self.path}; set MT_FIRMWARE to mtfw-gen1.bin')
    self.layout = MTFirmwareLayout(core_count)
    self.v = self.dev.pci_dev.map_bar(2, off=FW_BASE, size=INIT_SIZE, fmt='B') if v is None else v
    self.mmu = MTMMU4(ROOT_BASE) if mmu is None else mmu
    self.root_view = self.dev.pci_dev.map_bar(2,off=ROOT_BASE,size=len(self.mmu.root),fmt='B')
    if prepare: self._prepare()

  def _prepare(self) -> None:
    lay = self.layout
    self.dev.smc._write(0x1400000, struct.pack('<I', 1))
    self.dev.smc._write(0x1400004, struct.pack('<I', 1))
    with open(self.path, 'rb') as f: raw = f.read()
    self.v[:len(raw)] = raw
    for start,end in [(FWIF_OFF,lay.trace),(lay.trace,lay.trace+0x1000),(lay.perf,lay.perf+0x1000),
                      (lay.fault,lay.watchdog+0x1000),(0x400000,0x410000)]: self.v[start:end] = bytes(end-start)
    fw = build_firmware(lay)
    self.v[FWIF_OFF:FWIF_OFF+len(fw)] = fw
    self.v[lay.trace:lay.trace+128] = struct.pack('<32I', *([0xffffffff] * 32))
    self.v[lay.trace+0x17d20:lay.trace+0x17d20+128] = struct.pack('<32I', *([1] * 32))
    self.v[lay.fault+0x8208:lay.fault+0x8210] = struct.pack('<Q', META_BASE + lay.dump)
    self.v[lay.fault+0x8210:lay.fault+0x8214] = struct.pack('<I', 1)
    for i in range(64):
      self.v[lay.fault+i*0x208+0x1f0:lay.fault+i*0x208+0x1f8] = struct.pack('<Q', META_BASE+lay.regbuf+i*(lay.reg_dump_buffer_size>>6))
      self.v[lay.fault+i*0x208+0x1f8:lay.fault+i*0x208+0x1fc] = struct.pack('<I', lay.reg_dump_buffer_size>>9)
    self.v[lay.dump+8:lay.dump+12] = struct.pack('<I', 1)
    rc = bytearray(16)
    struct.pack_into('<I', rc, 0, 0x208e150)
    struct.pack_into('<I', rc, 5, 0x1000000)
    struct.pack_into('<H', rc, 10, self.core_count * 2)
    self.v[lay.regcfg:lay.regcfg+16] = rc
    self.mmu.map_range(FW_DEVVA, FW_BASE, INIT_SIZE)

  def _write(self, off:int, value:int, size:int=8) -> None: self.dev.smc._write(MT_SOCIF_OFFSET + off, value.to_bytes(size, 'little'))
  def _read(self, off:int, size:int=8) -> int: return int.from_bytes(self.dev.smc._read(MT_SOCIF_OFFSET + off, size), 'little')
  def _write_read(self, off:int, value:int, size:int=8) -> None:
    self._write(off,value,size)
    self._read(off,size)

  def upload_mmu(self) -> None:
    if not self.mmu.dirty_pages: return
    chunks:list[tuple[int,int]] = []
    for off in sorted(self.mmu.dirty_pages):
      if chunks and off == chunks[-1][1]: chunks[-1] = (chunks[-1][0],off+0x1000)
      else: chunks.append((off,off+0x1000))
    for start,end in chunks: self.root_view[start:end] = self.mmu.root[start:end]
    self.mmu.dirty_pages.clear()

  def start(self, timeout:float=10.0) -> None:
    self.upload_mmu()
    self._write_read(mtregs.RGX_CR_SYS_BUS_SECURE,0,4)
    for value in MT_MERCER_RESET_ASSERT: self._write_read(mtregs.RGX_CR_MERCER_SOFT_RESET,value)
    for off in [mtregs.RGX_CR_SWIFT_SOFT_RESET,mtregs.RGX_CR_TEXAS_SOFT_RESET]: self._write_read(off,0xffffffff,4)
    for value in MT_SOFT_RESET_SEQUENCE: self._write_read(mtregs.RGX_CR_SOFT_RESET,value)
    for off in [mtregs.RGX_CR_TEXAS_SOFT_RESET,mtregs.RGX_CR_SWIFT_SOFT_RESET]: self._write_read(off,0,4)
    for value in MT_MERCER_RESET_DEASSERT: self._write_read(mtregs.RGX_CR_MERCER_SOFT_RESET,value)
    self._read(mtregs.RGX_CR_SOFT_RESET)
    self._write(mtregs.RGX_CR_META_BOOT,mtregs.RGX_CR_META_BOOT_MODE_EN,4)
    self._write(mtregs.RGX_CR_MTS_GARTEN_WRAPPER_CONFIG,0)
    self._write(mtregs.RGX_CR_CLK_CTRL0,MT_CLK_CTRL0_PRE_JONES)
    self._write(mtregs.RGX_CR_CLK_CTRL1,MT_CLK_CTRL1_PRE_JONES)
    self._write(mtregs.RGX_CR_JONES_RAM_INIT_KICK,mtregs.RGX_CR_JONES_RAM_INIT_KICK_MASKFULL & 0xff,4)
    clock_deadline = time.monotonic() + min(timeout, 1.0)
    while self._read(mtregs.RGX_CR_JONES_RAM_STATUS) & 0xff != 0xff and time.monotonic() < clock_deadline: time.sleep(0.001)
    self._write(mtregs.RGX_CR_CLK_CTRL0,MT_CLK_CTRL0_POST_JONES)
    self._write(mtregs.RGX_CR_CLK_CTRL1,MT_CLK_CTRL1_POST_JONES)
    self._write_read(mtregs.RGX_CR_HMMU_BYPASS,mtregs.RGX_CR_HMMU_BYPASS_MASKFULL,4)
    self._write(mtregs.RGX_CR_FWCORE_ADDR_REMAP_CONFIG12,mtregs.RGX_CR_FWCORE_ADDR_REMAP_CONFIG12_FETCH_EN_EN | FW_BASE)
    self._write(mtregs.RGX_CR_FWCORE_ADDR_REMAP_CONFIG5,mtregs.RGX_CR_FWCORE_ADDR_REMAP_CONFIG5_LOAD_STORE_EN_EN |
                mtregs.RGX_CR_FWCORE_ADDR_REMAP_CONFIG5_FETCH_EN_EN | (FW_BASE + 0x100000))
    self._write(mtregs.RGX_CR_MTS_GARTEN_WRAPPER_CONFIG,0)
    self._write_read(mtregs.RGX_CR_MMU_CBASE_MAPPING_CONTEXT,0,4)
    self._write(mtregs.RGX_CR_MMU_CBASE_MAPPING,ROOT_BASE >> mtregs.RGX_CR_MMU_CBASE_MAPPING_BASE_ADDR_ALIGNSHIFT,4)
    for off,value in MT_MMU_PAGE_RANGES: self._write(off,value)
    self._write(mtregs.RGX_CR_ACE_PROT_CTRL,MT_ACE_PROT_CTRL)
    self._write(mtregs.RGX_CR_ACE_CTRL,MT_ACE_CTRL)
    self._write(mtregs.RGX_CR_CLK_CTRL0,(self._read(mtregs.RGX_CR_CLK_CTRL0) & MT_CLK_CTRL0_UPDATE_MASK) | MT_CLK_CTRL0_UPDATE_BITS)
    self._write(mtregs.RGX_CR_CLK_CTRL1,(self._read(mtregs.RGX_CR_CLK_CTRL1) & MT_CLK_CTRL1_UPDATE_MASK) | MT_CLK_CTRL1_UPDATE_BITS)
    time.sleep(3e-6)
    self._write_read(mtregs.RGX_CR_SOFT_RESET,0)
    time.sleep(3e-6)
    self._write(mtregs.RGX_CR_FWCORE_DMI_DMCONTROL,mtregs.RGX_CR_FWCORE_DMI_DMCONTROL_DMACTIVE_EN,4)
    self._write(mtregs.RGX_CR_FWCORE_BOOT,mtregs.RGX_CR_FWCORE_BOOT_MODE_EN,4)
    deadline = time.monotonic() + timeout
    started_off, build_off = FWIF_OFF + mt_off(mtfw.struct_MTFW_FWIF_TAG,'fwStarted'), FWIF_OFF + mt_off(mtfw.struct_MTFW_FWIF_TAG,'fwBuildVer')
    while True:
      started = struct.unpack('<I', self.v[started_off:started_off+4])[0]
      if started & 0xff == MT_FW_STARTED_MAGIC:
        self.started, self.build = started, bytes(self.v[build_off:build_off+mtfw.MTFWIF_FW_BUILD_VERSION_SIZE]).rstrip(b'\0')
        return
      if time.monotonic() > deadline:
        raise TimeoutError(f"MT firmware start timed out: {started:#x}, fault={self._read(mtregs.RGX_CR_MMU_FAULT_STATUS_META):#x}")
      time.sleep(0.01)

# **** CCB ****

MT_TOKEN_BASE = 0x103
MT_HOST_PRIVATE = 0x4d544e4f50
MT_CCB_ITEM_SIZE = mtfw.MTFW_CCB_ITEM.SIZE

def build_ccb_item(generation:int, submission_size:int) -> bytes:
  ci = mtfw.MTFW_CCB_ITEM()
  ci.submissionCmd, ci.hostPrivateData = mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, MT_HOST_PRIVATE
  sp = ci.ccbItemPara.submissionPara
  sp.pageTableRootAddr, sp.pageTableGeneration = ROOT_BASE, generation
  sp.submissionAddr, sp.submissionSize = SUBMISSION_VA, submission_size
  return bytes(ci)

@dataclasses.dataclass
class MTCCB:
  item_off:int
  queue_off:int

# **** launch ****

BACKING_PA, BACKING_SIZE = 0x40000000,0x10000000

class MTLaunch(MTFirmware):
  write_map:dict[int,bytes]
  packet:bytearray
  kernel_lib:bytes
  backing_offsets:dict[int,int]
  text_va:int
  text_offset:int
  pds_execution_address:int
  pds_code_off:int
  pds_data_offsets:list[int]
  generated_image:MTKernelImage
  _active:MTLaunch|None = None
  _loaded:MTLaunch|None = None

  def __init__(self, dev:MTDev):
    reuse = MTLaunch._active is not None and MTLaunch._active.dev is dev
    active = MTLaunch._active if reuse else None
    super().__init__(dev, prepare=not reuse, mmu=active.mmu if active is not None else None, v=active.v if active is not None else None)
    self.sv:Any = active.sv if active is not None else self.dev.pci_dev.map_bar(2, off=SUB_PA, size=INIT_SIZE, fmt='B')
    if not reuse: self.sv[:0x20000] = bytes(0x20000)
    # The firmware reserves a fixed output/status page even when user output VAs differ.
    self.zero_pa, self.status_pa = SUB_PA + 0x1000, SUB_PA + 0x3000
    self.mmu_dirty = True
    self.backing_pa = BACKING_PA
    self.backing:Any = active.backing if active is not None else self.dev.pci_dev.map_bar(2, off=self.backing_pa, size=BACKING_SIZE, fmt='B')
    self._map_generated()
    self._map_common()
    self._prepare_packet()
    self.ccb:MTCCB|None = None
    self.launch_id = 1
    self.generation = 1
    MTLaunch._loaded = self

  def _map_generated(self) -> None: raise NotImplementedError

  def _map_common(self) -> None:
    for base in [0xda00400000,0xe000400000]:
      for off in range(0,0x1000000,0x1000):
        if base+off not in self.mmu.pages: self.mmu.map(base+off,self.zero_pa,MT_PTE_READ_WRITE)
    for off in range(0x400000, 0x410000, 0x1000): self.mmu.map(FW_DEVVA + off, FW_BASE + off, MT_PTE_READ_WRITE)
    if 0x100000000 not in self.mmu.pages: self.mmu.map(0x100000000,SUB_PA+0x6000,MT_PTE_READ_WRITE)
    for i in range(18): self.mmu.map(0xd9bfff0000+i*0x1000,SUB_PA+0x10000+i*0x1000,MT_PTE_READ_WRITE)
    self.mmu.map(0xe1c2003000,self.status_pa,MT_PTE_READ_WRITE)
    aliases = [(0xc000000000,0x20400000),(0x22000000,0x20400000),(0x90400000,0x20400000),
               (0xe1c0401000,SUB_PA+0x1000),(0xe1c0402000,SUB_PA+0x8000),(0xe1c2001000,SUB_PA+0x1000),
               (0xe1c2002000,SUB_PA+0x8000),(0xe000002000,SUB_PA+0x8000),(0x90005000,SUB_PA+0x5000),
               (0x90008000,SUB_PA+0x8000),(0xdafffff000,SUB_PA+0x11000),
               (0x81e120000,self.mmu.pages[0x81e1200000]),(0x81e140000,self.mmu.pages[0x81e1400000])]
    for va,pa in aliases:
      if va not in self.mmu.pages: self.mmu.map(va,pa,MT_PTE_READ_WRITE)
    for off in range(0,0x100000,0x1000):
      if FW_DEVVA+0x200000+off not in self.mmu.pages: self.mmu.map(FW_DEVVA+0x200000+off,SUB_PA+off,MT_PTE_READ_WRITE)
    for off in range(0,0x1000000,0x1000):
      if 0xdb00000000+off not in self.mmu.pages: self.mmu.map(0xdb00000000+off,self.zero_pa,MT_PTE_READ_WRITE)

  def _prepare_packet(self) -> None:
    packet = bytearray(build_submission())
    struct.pack_into('<Q', packet, 0x38, 1)
    struct.pack_into('<I', packet, 0x40, 1)
    struct.pack_into('<Q', packet, 0x08, FW_DEVVA + 0x40b000)
    struct.pack_into('<Q', packet, 0xc0 + 0x88, FW_DEVVA + 0x409000)
    struct.pack_into('<Q', packet, 0xc0 + 0x90, FW_DEVVA + 0x40a000)
    previous = MTLaunch._loaded
    for off in range(0,len(packet),0x1000):
      chunk = bytes(packet[off:off+0x1000])
      unchanged = previous is not None and previous.packet[off:off+len(chunk)] == chunk
      if off == 0 and previous is not None:
        unchanged = previous.packet[:0x38] == packet[:0x38] and previous.packet[0x40:0x1000] == packet[0x40:0x1000]
      if not unchanged: self.sv[off:off+len(chunk)] = chunk
      if self.mmu.pages.get(SUBMISSION_VA+off) != SUB_PA+off: self.mmu.map(SUBMISSION_VA+off,SUB_PA+off,MT_PTE_READ_WRITE)
    self.packet = packet
    if previous is None: self.v[0x405000:0x405010] = struct.pack('<QII', ROOT_BASE, 0xffffffff, 0)

  def _write_backing(self, off:int, data:bytes|bytearray) -> None:
    for pos in range(off,off+len(data),0x1000):
      page_off = pos & ~0xfff
      start = max(off,page_off)
      end = min(off+len(data),page_off+0x1000)
      page = bytearray(self.write_map.get(page_off,bytes(0x1000)))
      page[start-page_off:end-page_off] = data[start-off:end-off]
      self.write_map[page_off] = bytes(page)

  def load_kernel(self, image:MTKernelImage) -> None:
    text_off = self.backing_offsets[self.text_va] + self.text_offset
    text_start = self.text_va + self.text_offset
    for page in range(text_start & ~0xfff,(text_start+len(image.text)+0xfff) & ~0xfff,0x1000):
      pa = self.backing_pa + self.backing_offsets[self.text_va] + page-self.text_va
      if self.mmu.pages.get(page) != pa:
        self.mmu.map(page,pa,MT_PTE_READ_WRITE)
        self.mmu_dirty = True
        self.generation += 1
    self._write_backing(text_off,image.text)
    const_off = self.backing_offsets[ARGS_VA]
    args = image.metadata.kernel['.args']
    user_args = [a for a in args if a['.argBase']['.name'] not in ['Global Spill Base','Core Number']
                 and not a['.argBase']['.name'].startswith('threadIdx')]
    clear_start = 0x48 if len(user_args) <= 1 else 0x50
    clear_end = max([0x80]+[imm.offset*4+4 for imm in image.metadata.immediates])
    self._write_backing(const_off+clear_start,bytes(clear_end-clear_start))
    for imm in image.metadata.immediates:
      if imm.offset >= 18: self._write_backing(const_off+imm.offset*4,struct.pack('<I',imm.value))

  def generate_pds(self, image:MTKernelImage) -> None:
    local = [PDS_UNUSED]*3
    workgroup = [PDS_UNUSED]*3
    for arg in image.metadata.kernel['.args']:
      kind,offset = arg['.kind'],arg['.argBase']['.offset']
      if 1 <= kind <= 3: local[kind-1] = offset
      elif 5 <= kind <= 7: workgroup[kind-5] = offset
    barrier = image.metadata.kernel['.barrier_coef'] if image.metadata.kernel['.has_barrier'] else PDS_UNUSED
    k = image.metadata.kernel
    temp_bytes = (k['.temp_reg_count']*4+7)&~7
    shared_blocks = (k['.shared_reg_count']+15)&~15
    prefix_words = (self.pds_execution_address,8,0,temp_bytes,shared_blocks,0x100000000,0,0x400d030001000078)
    prefix = tuple(struct.unpack('<16I',struct.pack('<8Q',*prefix_words)))
    program = compute_pds(self.pds_execution_address,(local[0],local[1],local[2]),(workgroup[0],workgroup[1],workgroup[2]),barrier,
                          coefficient_update=False,prefix=prefix)
    self._write_backing(self.pds_code_off+7,program.code)
    for off in self.pds_data_offsets: self._write_backing(off+0x78,program.data)

  def map_buffer(self, va:int, pa:int, size:int, flags:int=MT_PTE_READ_WRITE) -> None:
    self.mmu.map_range(va, pa, size, flags)
    self.mmu_dirty = True

  def set_spill(self, va:int) -> None:
    self._write_backing(self.backing_offsets[ARGS_VA],struct.pack('<Q',va))

  def set_grid(self, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> None:
    off = self.backing_offsets[GRID_VA]
    for i, field in enumerate([0x2c,0x30,0x34]): self._write_backing(off+field,struct.pack('<I',global_size[i]-1))
    block = bytearray(self.write_map.get(off,bytes(0x1000))[0x3c:0x40])
    block[0] = local_size[2] - 1
    block[1] = (local_size[1] - 1) << 2
    block[2:4] = struct.pack('<H', (local_size[0] - 1) << 4)
    self._write_backing(off+0x3c,block)
    k = self.generated_image.metadata.kernel
    has_block = any(a['.kind'] in (5,6,7) for a in k['.args'])
    local_mem = k['.shared_memory_size']
    blocks = max(1,(local_mem+63)//64)
    register_blocks = k['.shared_reg_count']//16+1
    for field,value in [(0x00,2),(0x04,0x406002),(0x08,0x3500),(0x18,0x2000000|register_blocks),
                        (0x20,0x101085),(0x24,0x406010),(0x28,0x3580),(0x38,(0x1000 if has_block else 0)+blocks),
                        (0x40,0x80000000)]: self._write_backing(off+field,struct.pack('<I',value))

  def start(self, timeout:float=10.0) -> None:
    if MTLaunch._active is not None and MTLaunch._active.dev is self.dev:
      active = MTLaunch._active
      assert active.ccb is not None
      self.v, self.layout, self.ccb = active.v, active.layout, active.ccb
      self.launch_id = active.launch_id + 1
      active.launch_id = self.launch_id
      if MTLaunch._loaded is not self:
        self._map_generated()
        self._map_common()
        self._prepare_packet()
        MTLaunch._loaded = self
      struct.pack_into('<Q', self.packet, 0x38, self.launch_id)
      self.generation = active.generation + 1
      active.generation = self.generation
      self.upload_mmu()
    else:
      super().start(timeout)
      MTLaunch._active = self
    self.mmu_dirty = False
    if self.ccb is None:
      self._write(mtregs.RGX_CR_MMU_FAULT_STATUS1,0)
      self._write(mtregs.RGX_CR_MMU_FAULT_STATUS2,0)
      self._write(mtregs.RGX_CR_EVENT_CLEAR,0xffffffff,4)
      self._write(mtregs.RGX_CR_MMU_FAULT_STATUS_META,0)
      self._write(mtregs.RGX_CR_MMU_FAULT_STATUS2_META,0)
      self.ccb = MTCCB(self.layout.item + mtfw.MTFW_CCB_TYPE_CDM * 4096 * MT_CCB_ITEM_SIZE,
                       self.layout.ccb + mtfw.MTFW_CCB_TYPE_CDM * mtfw.MTFW_CCB_QUEUE.SIZE)

# **** generated ****

# Device-VA to backing-PA layout used by the generated compute context. This is
# an allocation map, not captured launch state.
GENERATED_OFFSETS = {
  0x100000000:0x0,0x100400000:0x200000,0x100800000:0x400000,0x100a00000:0x600000,0x101a00000:0x1600000,
  **{0x102a00000+i*0x100000:0x2600000+i*0x100000 for i in range(33)},
  **{0x105a00000+i*0x100000:0x5600000+i*0x100000 for i in range(32)},
  0x8000000000:0x8600000,0x8000400000:0x8a00000,
  0x81e0802000:0x8e00000,0x81e0812000:0x8e10000,0x81e0814000:0x8e12000,0x81e0815000:0x8e13000,
  0x81e081e000:0x8e1c000,0x81e0820000:0x8e1e000,0x81e0830000:0x8e1f000,0x81e0840000:0x8e2f000,
  0x81e0849000:0x8e38000,0x81e0861000:0x8e50000,0x81e08c9000:0x8eb8000,0x81e08d2000:0x8ec1000,
  0x81e0a00000:0x8ed9000,CDM_EXECUTION_ADDRESS:0x92d9000,ARGS_VA:0x96d9000,GRID_VA:0x98d9000,
  RESUME_VA:0x9ad9000,
  **{PDS_MAIN_VA+i*0x1000:0x9ed9000+i*0x1000 for i in range(7)},
  TEXT_VA:0xa0df000,0xe000400000:0xa4df000,0xe000402000:0xa4e1000,0xe000408000:0xa4e7000,
  0xf000000000:0xa4ed000,0xf000400000:0xa8ed000,TASK_VA:0xb0ed000,0x100805000:0xb0ee000,
  **{0x100820000+i*0x1000:0xb0ef000+i*0x1000 for i in range(9)},
}

class MTGeneratedLaunch(MTLaunch):
  _text_span = 0

  def generate_task_page(self) -> None:
    # Generation-1 CDM task/context layout. Kernel-specific resources and PDS
    # programs are overlaid from ELF metadata by set_generated()/generate_pds().
    page = bytearray(0x1000)
    qwords = {0x10:0x8000404500,0x18:0x6800000000,0x78:0x81e0e00000,0x80:8,
              0x90:0,0x98:0,0xa0:0x1800000000,0xa8:0,0xb0:0x2000000000,0xb8:1,0xc0:0,0xc8:0x30d4000000178,
              0x1a0:0x10000001f0,0x1f8:0x81e0812000,0x200:0x8000404300,0x208:0x8000404380,0x218:0x8000404400,
              0x220:0x2078200800,0x300:0xe140000040010081,0x308:0x3606c1400002,0x310:0x3601,
              0x320:0x2000000,0x328:0x80000000,0x380:0xb0}
    dwords = {0x24:0xffffffff,0x28:0x58,0x2c:2,0x38:1,0x40:1,0x58:2,0x60:0xa0,0x68:0x198,0x70:0xa,
              0xd0:0x14,0x184:1,0x1a8:2,0x210:0x2c,0x228:0x200,0x500:4}
    for field,value in qwords.items(): struct.pack_into('<Q',page,field,value)
    for field,value in dwords.items(): struct.pack_into('<I',page,field,value)
    self._write_backing(self.backing_offsets[TASK_VA],page)

  def generate_static_pds(self) -> None:
    # Firmware resume and main PDS programs. The flags/payloads not defined by Mesa are generation-1 resume ABI bits.
    resume = bytearray(0x4000)
    def words(off:int, *values:int) -> None: struct.pack_into(f'<{len(values)}I',resume,off,*values)
    words(0x20,_doutu())
    words(0x30,_release())
    words(0x40,_doutw(0,0x10),_doutu(flags=0x10))
    words(0x50,_doutw(0,0x10),_doutu(flags=0x10))
    def doutw(off:int) -> None: words(off,_doutd(0x20),_doutu(src0=0x18,flags=0x8))
    def doutd(off:int) -> None:
      words(off,_vendor_sp9(1,0x340003),_doutw(0,0x13),_release(),_doutw(0,0x1b),_doutu(cc=1,flags=0x10))
    for base in range(0x80,0x1800,0x100):
      doutw(base)
      doutd(base+0x80)
    words(0x1880,_vendor_sp9(1,0x340002),_release(),_doutw(0,0x10),_doutu(flags=0x10))
    words(0x1900,_doutw(0,0x10),_doutu(flags=0x10))
    words(0x1980,_doutc(0x10),_release())
    for off,address,size in [(0x1a00,0x81e08a1180,0x1000000600),(0x1a20,0x81e08a1180,0x2000000800),
                             (0x1a40,0x81e08a5180,0x1000000600),(0x1a60,0x81e08a5180,0x2000000800)]:
      struct.pack_into('<QI4xQ',resume,off,address,0x80000043,size)
    struct.pack_into('<QII',resume,0x1a80,0x81e08a9180,0,0x21000000)
    words(0x1aa0,_wdf(0x340002),_wdf(0x300040),_sftlp64(0,7,0,0x19,0x17,0xfc,0xd),_cmp(0,0,1,1,0xb,0x3001),
          _ddmad(0,0,0x19,8,4,7),_st(0x67),_ld(2,flags=0x200000),_st(),_release())
    struct.pack_into('<QII',resume,0x1b00,0x81e08a9180,0,0x30400000)
    words(0x1b10,_wdf(0x340002),_wdf(0x300040),_sftlp64(0,7,0,0x19,0x17,0xfc,0xd),_cmp(0,0,1,1,0xb,0x3001),
          _ddmad(0,0,0x19,8,4,7),_ld(0x67),_st(),_release())
    words(0x1b80,_vendor_sp9(1,0x340002),_release(),_doutw(0,0x10),_doutu(flags=0x10))
    words(0x1c00,_doutw(0,0x10),_doutu(flags=0x10))
    words(0x1c80,_doutc(0x10),_release())
    for base in range(0x1d00,0x3500,0x100):
      doutw(base)
      doutd(base+0x80)
    words(0x3500,_doutd(0x10,end=1))
    args = self.generated_image.metadata.kernel['.args']
    has_block = any(a['.kind'] in (5,6,7) for a in args)
    has_local = any(a['.kind'] in (1,2,3) for a in args)
    if self.generated_image.metadata.kernel['.has_barrier']:
      words(0x3580,_vendor_sp9(1,0x340004),_doutw(0,0x13),_doutw(0,0x18),_release(),_doutw(0,0x23),_doutu(cc=1,flags=0x18))
    elif has_block: doutd(0x3580)
    elif has_local: words(0x3580,_vendor_sp9(1,0x340002),_release(),_doutw(0,0x13),_doutu(cc=1,flags=0x10))
    else: words(0x3580,_vendor_sp9(1,0x340002),_release(),_doutu(cc=1,flags=0x8))
    words(0x3600,_doutc(0x10))
    self._write_backing(self.backing_offsets[RESUME_VA],resume)
    main = bytearray(0x1000)
    for off,word in {0x00:_doutd(0x10,end=1),0x10:_doutu(),0x20:_doutd(0x10),0x24:_doutu(flags=0x10)}.items():
      struct.pack_into('<I',main,off,word)
    self._write_backing(self.backing_offsets[PDS_MAIN_VA],main)

  def _map_generated(self) -> None:
    self.backing_offsets = GENERATED_OFFSETS.copy()
    self.write_map = {}
    for va in (RESOURCE_VA,ARGS_VA,GRID_VA,CDM_EXECUTION_ADDRESS):
      self._write_backing(self.backing_offsets[va],bytes(0x1000))
    self.text_va = TEXT_VA
    self.text_offset = 0x2000
    self.pds_execution_address = CDM_EXECUTION_ADDRESS
    self.pds_code_off = self.backing_offsets[PDS_MAIN_VA]
    self.pds_data_offsets = [self.backing_offsets[TASK_VA]]
    self.generate_task_page()
    items = list(GENERATED_OFFSETS.items())
    for i,(va,off) in enumerate(items):
      size = items[i+1][1]-off if i+1 < len(items) else 0x1000
      pa = self.backing_pa+off if off+size <= len(self.backing) else self.zero_pa
      if self.mmu.pages.get(va) != pa: self.mmu.map_range(va,pa,size,MT_PTE_READ_WRITE)
    for va in (0x2078200000,0x207820c000):
      if va not in self.mmu.pages: self.mmu.map(va,self.zero_pa,MT_PTE_READ_WRITE)
    error_off = self.backing_offsets[0x81e0e00000]
    self._write_backing(error_off+0x1f0,struct.pack('<Q',0x81e0e00218))
    self._write_backing(error_off+0x1f8,struct.pack('<I',1))
    self._write_backing(error_off+0x200,struct.pack('<Q',0x81e0e00288))
    self._write_backing(error_off+0x208,struct.pack('<I',0x11d))
    self._write_backing(error_off+0x218,struct.pack('<Q',0x81e0e00228))
    self._write_backing(error_off+0x220,struct.pack('<I',6))
    self._write_backing(error_off+0x228,struct.pack('<Q',0x208e150))
    self._write_backing(error_off+0x230,struct.pack('<I',0x20001))
    self._write_backing(error_off+0x238,struct.pack('<Q',0x8000020004084178))
    self._write_backing(error_off+0x240,struct.pack('<I',0x100004))
    self._write_backing(error_off+0x248,struct.pack('<Q',0x8000020040084800))
    self._write_backing(error_off+0x250,struct.pack('<I',0x1000004))
    self._write_backing(error_off+0x258,struct.pack('<Q',0x8000020000000000))
    self._write_backing(error_off+0x260,struct.pack('<I',4))
    self._write_backing(error_off+0x268,struct.pack('<Q',0x8000020000000000))
    self._write_backing(error_off+0x270,struct.pack('<I',4))
    self._write_backing(error_off+0x278,struct.pack('<Q',0x80000200010841b8))
    self._write_backing(error_off+0x280,struct.pack('<I',0x40004))
  def set_generated(self, image:MTKernelImage) -> None:
    self.generated_image = image
    k = image.metadata.kernel
    has_ids = any(a['.kind'] in (1,2,3,5,6,7) for a in k['.args'])
    has_barrier = k['.has_barrier']
    resource_off = self.backing_offsets[RESOURCE_VA]
    spill_bytes = max(k['.private_memory_size'],k['.local_mem_used_in_cluster'])
    # Spill text uses a 32 KiB region when possible and a 192 KiB region for larger images.
    self.text_offset = (0x8000 if len(image.text) <= 0x8000 else 0x30000) if spill_bytes else \
      0x1000 if len(image.text) <= 0x1000 else 0x2000 if len(image.text) <= 0x20000 else 0x30000
    self.resource_temp_offset = 0x2c if has_barrier else 0x24 if has_ids else 0x1c
    self.resource_text_field = 0x28 if has_barrier else 0x20 if has_ids else 0x18
    self.resource_temp_flags = 1 if has_barrier else 0
    self._write_backing(resource_off,struct.pack('<Q',ARGS_VA))
    self._write_backing(resource_off+8,struct.pack('<I',0x90000000|k['.shared_reg_count']))
    self._write_backing(self.backing_offsets[ARGS_VA]+8,struct.pack('<I',0xff))
    controls:tuple[int,...]
    if has_barrier:
      controls = (0x30000347,0xb0002003,0x80000043)
    elif any(a['.kind'] in (5,6,7) for a in k['.args']):
      controls = (0xb0000347,0x80000043)
    elif has_ids:
      controls = (0x80000043,)
    else:
      controls = (0x80,)
    for i,value in enumerate(controls): self._write_backing(resource_off+0x18+i*4,struct.pack('<I',value))
    self._write_backing(resource_off+self.resource_text_field,struct.pack('<I',self.text_offset))
    temp = ((k['.temp_reg_count']*4+7)&~7) | self.resource_temp_flags
    self._write_backing(resource_off+self.resource_temp_offset,struct.pack('<I',temp))
    MTGeneratedLaunch._text_span = max(MTGeneratedLaunch._text_span,len(image.text))
    text_off = self.backing_offsets[self.text_va]+self.text_offset
    self._write_backing(text_off,bytes((MTGeneratedLaunch._text_span+0xfff)&~0xfff))
    self.load_kernel(image)

# **** compiler ****

DEFAULT_MUSA_PATH = '/opt/musa'

class MTCompiler:
  def __init__(self, musa_path:str|None=None):
    self.musa_path = musa_path or os.environ.get('MUSA_PATH', DEFAULT_MUSA_PATH)
    self.bin = pathlib.Path(self.musa_path) / 'bin'

  def _run(self, argv:list[str], cwd:str) -> None:
    result = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f"MT tool failed ({' '.join(argv)}):\n{result.stdout}{result.stderr}")

  def extract_raw(self, obj:pathlib.Path, cwd:str) -> bytes:
    fatbin = pathlib.Path(cwd) / 'fatbin.bin'
    self._run([str(self.bin/'llvm-objcopy'), f'--dump-section=.mt_fatbin={fatbin}', str(obj)], cwd)
    device = pathlib.Path(cwd) / 'device.out'
    self._run([str(self.bin/'clang-offload-bundler'), '-unbundle', '-type=o',
               '-targets=host-x86_64-unknown-linux,musa-mtgpu-mt-musa--mp_21',
               f'-inputs={fatbin}', f'-outputs={cwd}/host.o,{device}'], cwd)
    return device.read_bytes()

  def compile_raw(self, source:str) -> bytes:
    with tempfile.TemporaryDirectory(prefix='mt-mcc-') as td:
      src, obj = pathlib.Path(td) / 'kernel.mu', pathlib.Path(td) / 'kernel.o'
      src.write_text(source)
      include = pathlib.Path(td)/'include'
      include.mkdir()
      (include/'cuda_fp16.h').write_text('#pragma once\n#include <musa_fp16.h>\n')
      self._run([str(self.bin/'mcc'), '-O3', f'--musa-path={self.musa_path}', '--offload-arch=mp_21', f'-I{include}',
                 '-c', str(src), '-o', str(obj)], td)
      return self.extract_raw(obj,td)

  def compile(self, source:str) -> MTKernelImage:
    return MTKernelImage.from_elf(self.compile_raw(source))

# **** runtime ****

# KERNEL8 value used by multi-thread spill workgroups.
MT_SPILL_WORKGROUP_KERNEL8 = 0x108085
MT_SPILL_MIN_STRIDE = 0x2000000

class MUSACompiler(Compiler):
  def __init__(self):
    super().__init__('mt_O3')
    self.mt = MTCompiler()
  def compile(self, src:str) -> bytes: return self.mt.compile_raw(src)

class MUSARenderer(CUDARenderer):
  kernel_typedef = 'extern "C" __global__ void'
  def __init__(self, target):
    CStyleLanguage.__init__(self, target)
    self.compiler, self.tensor_cores = MUSACompiler(), []
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    return super().render_kernel(function_name, kernel, bufs, uops, ['#include <musa.h>'] + (prefix or []))
  def supported_dtypes(self): return set(dtypes.all) - {dtypes.double}

@dataclasses.dataclass(frozen=True)
class MTBufferMeta:
  va: int
  pa: int
  size: int

class MTAllocator(Allocator['MTDevice']):
  def __init__(self, dev):
    super().__init__(dev)
    self.next_va, self.next_pa = 0x8001000000, 0x80000000
    self.dev.views = {}
    self.host_buffers:dict[int,Any] = {}
    self.free_blocks:list[tuple[int,int,int]] = []

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    size = round_up(size, 0x1000)
    if options.host:
      host = (ctypes.c_ubyte * size)()
      va = ctypes.addressof(host)
      view = MMIOInterface(va,size,fmt='B')
      meta = MTBufferMeta(va,0,size)
      self.host_buffers[va] = host
      self.dev.views[va] = (view,meta)
      return BufferStorage(va,meta,view)
    if self.free_blocks and self.free_blocks[-1][2] == size:
      va, pa, size = self.free_blocks.pop()
    else:
      va, pa = self.next_va, self.next_pa
      self.next_va += size
      self.next_pa += size
    view = self.dev.mt.pci_dev.map_bar(2, off=pa, size=size, fmt='B')
    meta = MTBufferMeta(va, pa, size)
    self.dev.map_buffer(va, pa, size)
    self.dev.views[va] = (view, meta)
    return BufferStorage(va, meta, view)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    meta = storage.meta
    self.dev.views.pop(meta.va, None)
    self.dev.buffer_maps.pop(meta.va, None)
    if meta.va in self.host_buffers:
      self.host_buffers.pop(meta.va)
    else: self.free_blocks.append((meta.va, meta.pa, meta.size))

  def _copyin(self, dest:int, src:memoryview):
    view, meta = self.dev._view(dest)
    off = dest-meta.va
    if off % 4 == 0 and src.nbytes % 4 == 0: view[off:off+src.nbytes] = src.cast('B')
    else:
      start, end = off & ~3, round_up(off+src.nbytes,4)
      data = bytearray(view[start:end])
      data[off-start:off-start+src.nbytes] = src.cast('B')
      view[start:end] = data

  def _copyout(self, dest:memoryview, src:int):
    view, meta = self.dev._view(src)
    off = src-meta.va
    start, end = off & ~3, round_up(off+dest.nbytes,4)
    data = view[start:end]
    dest[:] = data[off-start:off-start+dest.nbytes]

  def _map(self, buf:Buffer) -> BufferStorage:
    if buf.device == self.dev.device: return buf.get_storage()
    storage = self._alloc(buf.nbytes,buf.options)
    self._copyin(storage.buf,buf.as_memoryview(allow_zero_copy=True))
    return storage

  def _unmap(self, storage:BufferStorage): self._free(storage,BufferSpec())
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def mt_poll(link:UOp, addr:UOp, target:UOp) -> UOp:
  loop = UOp.range(UOp(Ops.NOOP).after(link,start:=get_time_ms(link)),next(UOp.unique_num),dtype=dtypes.void)
  slot = usb_stack(dtypes.u64)
  header = usb_ctrl(link.after(loop),0x40,0xF0,0x20|0x0F00,2,usb_stack(dtypes.uint64,addr,2).index(0),12,5000)
  read = usb_bulk(link.after(header),0x81,slot.index(0),8)
  response = slot.after(read,ccall(libc.usleep,100).sink()).index(0).load()
  done = (response >> 32).cast(dtypes.u64)
  poll = done.backedge(loop,(done < target) & (get_time_ms(done) - start < int(float(os.environ.get('MT_TIMEOUT','5'))*1000))).sink()
  status = response & 0xffffffff
  bad = ((done.after(poll) != target).cast(dtypes.u64) << 63) | \
    (((status&0xffffffff) != mtfw.MTFW_CCB_RESP_DONE).cast(dtypes.u64)*target << 32) | (status & 0xffffffff)
  return usb_fail(link.after(poll),bad)


@dataclasses.dataclass
class MTSubmission:
  """One encoded launch and the state needed to publish it.

  args is the runtime-patched argument page. write_map contains sparse generated
  backing pages. mmu_pages is captured before shared launch state is replaced.
  grid_deps is an optional symbolic grid/local-size patch.
  """
  launch: MTGeneratedLaunch
  args: UOp
  packet: bytes
  write_map: dict[int,bytes]
  generation: int
  mmu_pages: dict[int,bytes]|None
  grid_deps: UOp|None

class MTComputeQueue(HWQueue):
  dev:MTDevice
  def __init__(self, submit):
    super().__init__(submit)
    self.submissions:list[MTSubmission] = []
    self.signals:list[tuple[UOp,UOp]] = []

  def encode(self) -> UOp:
    for u in self.lin.src:
      if u.op is Ops.END: raise NotImplementedError('MT queue loops are not supported')
      if u.op is not Ops.CALL: continue
      body = u.src[0]
      if body.op is Ops.PROGRAM: self.exec(u,body)
      elif body.op is Ops.NATIVE:
        if body.arg.code == 'store': self.signal(u.src[1],u.src[2])
        elif body.arg.code == 'timestamp': self.timestamp(u.src[1])
        elif body.arg.code not in ('memory_barrier','exec_barrier','wait','wait_eq'):
          raise NotImplementedError(f'MT queue operation {body.arg.code}')
      elif body.op is not Ops.STORE: raise NotImplementedError(f'MT queue operation {body.op}')
    return self.submit(UOp.sink())

  def memory_barrier(self): pass  # every MT submission is drained before the next
  def exec_barrier(self): pass
  def wait(self, signal:UOp, value:UOp): pass  # synchronous submissions make cross-queue waits implicit
  def signal(self, signal:UOp, value:UOp): self.signals.append((signal,value))
  def timestamp(self, signal:UOp): pass  # timestamps are read synchronously by MTDevice

  def snapshot_mmu(self, launch:MTLaunch) -> dict[int,bytes]:
    pages:dict[int,bytes] = {}
    for va in launch.mmu.pages:
      if 0x8001000000 <= va < cast(MTAllocator,self.dev.allocator).next_va: continue
      pci,pdi,pti = (va>>30)&0x3ff,(va>>21)&0x1ff,(va>>12)&0x1ff
      if (pci,pdi) in launch.mmu.pt:
        offs = {(pci*4)&~0xfff,(launch.mmu.pd[pci]-launch.mmu.root_pa+pdi*8)&~0xfff,
                (launch.mmu.pt[pci,pdi]-launch.mmu.root_pa+pti*8)&~0xfff}
        for off in offs: pages[off] = bytes(launch.mmu.root[off:off+0x1000])
    return pages

  def exec(self, call:UOp, prg:UOp):
    if self.submissions and self.submissions[-1].mmu_pages is None:
      self.submissions[-1].mmu_pages = self.snapshot_mmu(self.submissions[-1].launch)
    obj = TinyELF.from_program(prg)
    global_size = cast(tuple[int,int,int],tuple(x if isinstance(x,int) else 1 for x in prg.arg.global_size))
    local_size = cast(tuple[int,int,int],tuple(x if isinstance(x,int) else 1 for x in prg.arg.local_size))
    launch = self.dev.get_launch(obj,global_size,local_size)
    self.dev.activate_launch(launch,obj,global_size,local_size)
    if obj.profile_key is not None: self.dev.cached_launches[obj.profile_key] = (launch,obj,global_size,local_size)
    grid_deps = None
    if any(not isinstance(x,int) for x in (*prg.arg.global_size,*prg.arg.local_size)):
      grid_rows = [(off,(prg.arg.global_size[i]-1).cast(dtypes.u32)) for i,off in enumerate([0x2c,0x30,0x34])
                   if not isinstance(prg.arg.global_size[i],int)]
      local_rows = [(0x3c,prg.arg.local_size[2]-1,dtypes.u8),(0x3d,(prg.arg.local_size[1]-1)<<2,dtypes.u8),
                    (0x3e,(prg.arg.local_size[0]-1)<<4,dtypes.u16)]
      grid_rows += [(off,val.cast(dt)) for off,val,dt in local_rows if not isinstance(val,int)]
      grid_off = launch.backing_offsets[GRID_VA]
      grid = UOp.alloc((0x1000,),dtypes.u8,device='CPU')
      grid_deps = patch(grid,grid_rows,launch.write_map.get(grid_off,bytes(0x1000)))
    const_off = launch.backing_offsets[ARGS_VA]
    args = UOp.alloc((0x1000,),dtypes.u8,device='CPU')
    rows = [(0x38+i*8,arg.cast(dtypes.u64)) for i,arg in enumerate(get_call_kernel_args(call,prg,self.devs))]
    self.submissions.append(MTSubmission(launch=launch,args=patch(args,rows,launch.write_map.get(const_off,bytes(0x1000))),
                                         packet=bytes(launch.packet),write_map=dict(launch.write_map),generation=launch.generation,
                                         mmu_pages=None,grid_deps=grid_deps))

  def submit(self, cmdbuf:UOp) -> UOp:
    if len(self.submissions) > 1 and self.submissions[-1].mmu_pages is None:
      self.submissions[-1].mmu_pages = self.snapshot_mmu(self.submissions[-1].launch)
    ops: list[Any] = []
    link = usb_link(self.dev.device)
    def add(op):
      nonlocal link
      ops.append(op)
      link = link.after(op)
    root_addr = self.dev.mt.pci_dev.bar_info(2)[0]+ROOT_BASE
    for i,submission in enumerate(self.submissions):
      launch,deps,packet,write_map,generation,mmu_pages,grid_deps = (submission.launch,submission.args,submission.packet,
                                                                      submission.write_map,submission.generation,submission.mmu_pages,
                                                                      submission.grid_deps)
      ccb = launch.ccb
      assert ccb is not None
      def write(addr:int|UOp, blob:bytes, rows=()):
        data = UOp.alloc((len(blob),),dtypes.u8,device='CPU')
        addr = addr if isinstance(addr,UOp) else UOp.const(addr,dtypes.u64)
        data = patch(data,list(rows),blob)
        if len(blob) > 0x200000: op = usb_stream(link,addr,data,UOp.const(len(blob),dtypes.int),True)
        else:
          header = usb_ctrl(link,0x40,0xF0,0x60|0x0F00,1,usb_stack(dtypes.uint64,addr,len(blob)//4).index(0),12,5000)
          op = usb_bulk(link.after(header),0x02,data.index(0),len(blob))
        add(op)
      previous = self.submissions[i-1].write_map if i else self.dev.uploaded_write_map
      changed = sorted((off,data) for off,data in write_map.items() if off != launch.backing_offsets[ARGS_VA] and
                       (previous.get(off) != data))
      chunks:list[tuple[int,bytearray]] = []
      for off,data in changed:
        if chunks and off == chunks[-1][0]+len(chunks[-1][1]): chunks[-1][1].extend(data)
        else: chunks.append((off,bytearray(data)))
      for off,chunk in chunks: write(launch.backing.addr+off,bytes(chunk))
      if len(self.submissions) > 1:
        if i == 0: write(launch.sv.addr,packet)
        else:
          previous_packet = self.submissions[i-1].packet
          for off in range(0,len(packet),0x1000):
            packet_chunk = packet[off:off+0x1000]
            if previous_packet[off:off+len(packet_chunk)] != packet_chunk: write(launch.sv.addr+off,packet_chunk)
        previous_mmu = self.submissions[i-1].mmu_pages if i else self.submissions[-1].mmu_pages
        assert mmu_pages is not None and previous_mmu is not None
        for off,data in mmu_pages.items():
          if previous_mmu.get(off) != data: write(root_addr+off,data)
      idx = link.index(2).load()
      seq = link.index(3).load()
      token = (MT_TOKEN_BASE+seq).cast(dtypes.u64)
      item_size = MT_CCB_ITEM_SIZE
      item = UOp.alloc((item_size,),dtypes.u8,device='CPU')
      item = patch(item,[(mt_off(mtfw.MTFW_CCB_ITEM,'ccbItemToken'),token)],build_ccb_item(generation,len(packet)))
      add(usb_stream(link.after(deps),UOp.const(launch.v.addr+ccb.item_off,dtypes.u64),item,UOp.const(item_size,dtypes.int),True))
      args_off = launch.backing_offsets[ARGS_VA]
      add(usb_stream(link,UOp.const(launch.backing.addr+args_off,dtypes.u64),deps,UOp.const(0x1000,dtypes.int),True))
      if grid_deps is not None:
        grid_off = launch.backing_offsets[GRID_VA]
        add(usb_stream(link,UOp.const(launch.backing.addr+grid_off,dtypes.u64),grid_deps,UOp.const(0x1000,dtypes.int),True))
      next_wr = ((idx + 1) & 0xff).cast(dtypes.u32)
      write(launch.v.addr+ccb.queue_off+idx*8,struct.pack('<Q',META_BASE+ccb.item_off))
      add(usb_poke(link,launch.v.addr+ccb.queue_off+mt_off(mtfw.MTFW_CCB_QUEUE,'reqCcbCtrl'),next_wr))
      add(usb_poke(link,self.dev.mt.mmio.addr+0xc00b00,UOp.const(mtfw.MTFW_CCB_TYPE_CDM,dtypes.u32)))
      response_off = mt_off(mtfw.MTFW_CCB_QUEUE,'respCcbQ') + mt_off(mtfw.MTFW_CCB_RESP,'respValue') + idx*mtfw.MTFW_CCB_RESP.SIZE
      add(mt_poll(link,launch.v.addr+ccb.queue_off+response_off,token))
      add(usb_poke(link,launch.v.addr+ccb.queue_off+mt_off(mtfw.MTFW_CCB_QUEUE,'respCcbCtrl')+mt_off(mtfw.MTFW_RING_CTRL,'rdIdx'),next_wr))
      add(link.index(2).store(next_wr.cast(dtypes.u64)))
      add(link.index(3).store(seq+1))
      self.dev.uploaded_write_map = write_map
    for signal,value in self.signals: add(signal.index(0).store(value))  # type: ignore[assignment]
    return UOp.sink(UOp.barrier(*self.deps,*ops))


Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("mt_compute_submit")).call(UPat()).after(allow_any_len=True,name="s"),lambda s:MTComputeQueue(s).encode()),
])

class MTDevice(Compiled):
  renderers = [MUSARenderer]
  timestamp_divider = 25
  serialize_hcq = True
  enqueue_copies = False
  def __init__(self, device:str=''):
    dev_id = int(device.split(':')[1]) if ':' in device else 0
    self.mt = MTDev.open_usb(dev_id)
    self.launch_order:list[MTGeneratedLaunch] = []
    self.launch_cache:dict[tuple[bytes,tuple[int,int,int],tuple[int,int,int]],MTGeneratedLaunch] = {}
    self.buffer_maps:dict[int,tuple[int,int]] = {}
    self.spill_buffer:tuple[int,BufferStorage]|None = None
    self.current_launch:MTGeneratedLaunch|None = None
    self.cached_launches:dict[bytes,tuple[MTGeneratedLaunch,TinyELF,tuple[int,int,int],tuple[int,int,int]]] = {}
    self.uploaded_write_map:dict[int,bytes] = {}
    self.started = False
    self.views:dict[int,tuple[Any,MTBufferMeta]] = {}
    super().__init__(device, MTAllocator(self), None, arch='mp_21')
    usb_dev = cast(USBPCIDevice,self.mt.pci_dev).usb
    words = [ctypes.addressof(x.contents) for x in (usb_dev.usb.handle,USB3.ctx())] + [0,0,0]
    host = Buffer('CPU',HOST_SIZE,options=BufferSpec(nolru=True),initial_value=struct.pack('5Q',*words).ljust(HOST_SIZE,b'\0'))
    Compiled.link_bufs[to_name(self.device,'usb_host')] = host

  @property
  def has_copy_queue(self) -> bool: return False

  def get_launch(self, obj:TinyELF, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1)) -> MTGeneratedLaunch:
    key = (obj.lib, global_size, local_size)
    if key not in self.launch_cache:
      launch = MTGeneratedLaunch(self.mt)
      launch.kernel_lib = b''
      self.launch_cache[key] = launch
      self.launch_order.append(launch)
      if len(self.launch_order) > 512:
        old = self.launch_order.pop(0)
        for launch_key,cached_launch in list(self.launch_cache.items()):
          if cached_launch is old: del self.launch_cache[launch_key]
        for profile_key,cached in list(self.cached_launches.items()):
          if cached[0] is old: del self.cached_launches[profile_key]
      for va,(pa,size) in self.buffer_maps.items(): launch.map_buffer(va,pa,size)
    return self.launch_cache[key]

  def map_buffer(self, va:int, pa:int, size:int):
    self.buffer_maps[va] = (pa,size)
    if self.current_launch is not None: self.current_launch.map_buffer(va,pa,size)
    if self.started and self.current_launch is not None: self.current_launch.upload_mmu()

  def _fault(self) -> tuple[int, ...]:
    if self.current_launch is None: return ()
    launch = self.current_launch
    return (launch._read(mtregs.RGX_CR_MMU_FAULT_STATUS1),launch._read(mtregs.RGX_CR_MMU_FAULT_STATUS2),
            launch._read(mtregs.RGX_CR_EVENT_STATUS,4),launch._read(mtregs.RGX_CR_CDM_CONTEXT_STORE_STATUS))

  def synchronize(self, timeout:int|None=None):
    if self.prof_ents and self.current_launch is not None:
      timestamp = struct.unpack('<Q',self.current_launch.v[0x40a000:0x40a008])[0]
      run_time = struct.unpack('<I',self.current_launch.v[0x409008:0x40900c])[0]
      for (buf,_), ent in self.prof_ents.items():
        sigs = buf.host.view(fmt='Q')
        sigs[ent.st_id], sigs[ent.en_id] = timestamp, timestamp + run_time
    super().synchronize(timeout)
    host_buf = Compiled.link_bufs[to_name(self.device,'usb_host')]
    assert isinstance(host_buf,Buffer)
    host = host_buf.host.view(fmt='q')
    if host[5]: raise RuntimeError(f'MT USB error: {host[5]}, fault={self._fault()}')

  def activate_cached(self, profile_key:bytes|None) -> None:
    if profile_key is not None and (cached:=self.cached_launches.get(profile_key)) is not None:
      launch,obj,global_size,local_size = cached
      if self.current_launch is not launch or MTLaunch._loaded is not launch:
        self.activate_launch(launch,obj,global_size,local_size)
        for off,data in launch.write_map.items(): launch.backing[off:off+len(data)] = data
        launch.sv[:0x1000] = launch.packet[:0x1000]

  def hcq_cacheable(self, linear:UOp) -> bool:
    for call in linear.toposort():
      if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and call.src[0].arg.target.arch == 'mp_21':
        if MTKernelImage.from_elf(TinyELF.from_program(call.src[0]).lib).metadata.kernel['.has_barrier']: return False
    return True

  def ensure_started(self, launch:MTGeneratedLaunch):
    if self.current_launch is not launch:
      launch.start(timeout=float(os.environ.get('MT_BOOT_TIMEOUT','10')))
      self.current_launch = launch
      self.started = True

  def activate_launch(self, launch:MTGeneratedLaunch, obj:TinyELF, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> None:
    image = MTKernelImage.from_elf(obj.lib)
    unchanged = MTLaunch._loaded is launch and launch.kernel_lib == obj.lib
    self.ensure_started(launch)
    if not unchanged:
      launch.set_generated(image)
      launch.kernel_lib = obj.lib
      launch.generate_static_pds()
      launch.generate_pds(image)
    if (spill_bytes:=max(image.metadata.kernel['.private_memory_size'],image.metadata.kernel['.local_mem_used_in_cluster'])):
      cores = launch.core_count
      threads = global_size[0]*global_size[1]*global_size[2]*local_size[0]*local_size[1]*local_size[2]
      # Private frames use per-core power-of-two strides (minimum 32 MiB).
      per_core_threads = (threads+cores-1)//cores
      spill_stride = max(MT_SPILL_MIN_STRIDE,1 << (spill_bytes*per_core_threads).bit_length())
      needed = spill_stride*cores
      if needed > 0x40000000: raise MemoryError(f'MT spill allocation {needed:#x} exceeds the validated 1 GiB limit')
      if self.spill_buffer is None or self.spill_buffer[0] < needed:
        self.spill_buffer = (needed,self.allocator.alloc(needed,BufferSpec()))
      spill = self.spill_buffer[1]
      launch.map_buffer(spill.buf,spill.meta.pa,needed)
      launch.set_spill(spill.buf)
      launch._write_backing(launch.backing_offsets[ARGS_VA]+8,struct.pack('<I',cores))
    if launch.mmu_dirty:
      launch.upload_mmu()
      launch.mmu_dirty = False
    launch.set_grid(global_size,local_size)
    if spill_bytes and local_size[0]*local_size[1]*local_size[2] > 1:
      launch._write_backing(launch.backing_offsets[GRID_VA]+0x20,struct.pack('<I',MT_SPILL_WORKGROUP_KERNEL8))

  def _view(self, va:int):
    if va in self.views: return self.views[va]
    for base, (view, meta) in self.views.items():
      if base <= va < base + meta.size: return view, meta
    raise RuntimeError(f'Unknown MT buffer address {va:#x}')

  def finalize(self):
    super().finalize()
    self.mt.close()
