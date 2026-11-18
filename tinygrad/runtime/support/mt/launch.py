from __future__ import annotations
import dataclasses, os, pathlib, pickle, struct, zlib
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.compute import MTFirmware, SUB_PA, FW_DEVVA, ROOT_BASE
from tinygrad.runtime.support.mt.ccb import MTCCB, MTResponse
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.elf import MTKernelImage

@dataclasses.dataclass(frozen=True)
class MTLaunchResult:
  response: MTResponse
  timestamp: int
  run_time: int
  output: int

@dataclasses.dataclass
class MTBundle:
  writes:list[tuple[int,bytes]]
  maps:list[tuple[int,int]]
  offsets:dict[int,int]
  cursor:int
  text_offset:int


def compress_mt_bundle(state:MTBundle) -> bytes: return zlib.compress(pickle.dumps(state,protocol=5),1)
def decompress_mt_bundle(data:bytes) -> MTBundle: return pickle.loads(zlib.decompress(data))


def read_mt_map(path:pathlib.Path) -> tuple[int,list[tuple[int,bytes]]]:
  if path.suffix == '.sparse':
    raw = path.read_bytes()
    file_size, count = struct.unpack_from('<QQ',raw)
    chunks, off = [], 16
    for _ in range(count):
      page, chunk_size = struct.unpack_from('<II',raw,off)
      off += 8
      chunks.append((page,raw[off:off+chunk_size]))
      off += chunk_size
    return file_size,chunks
  data, zero_page = path.read_bytes(), bytes(0x1000)
  return len(data),[(off,chunk) for off in range(0,len(data),0x1000)
                    if (chunk:=data[off:off+0x1000]) != zero_page[:len(chunk)]]


def parse_mt_bundle(bundle:pathlib.Path) -> MTBundle:
  writes:list[tuple[int,bytes]] = []
  maps:list[tuple[int,int]] = []
  offsets:dict[int,int] = {}
  cursor, resource_byte = 0, 0
  common = bundle.parent if bundle.name.endswith('fresh') else pathlib.Path('/tmp/mt-research') if bundle.name.startswith('mt-launch-') else bundle
  paths:dict[int,pathlib.Path] = {}
  for root in [common,bundle]:
    for path in root.glob('real-map-*'):
      if path.suffix not in ['.bin','.sparse']: continue
      va = int(path.name[len('real-map-'):].split('.')[0],16)
      if va not in paths or path.suffix == '.sparse': paths[va] = path
  for va,path in sorted(paths.items()):
    size,chunks = read_mt_map(path)
    offsets[va] = cursor
    pa = 0x40000000 + cursor
    chunk_map = dict(chunks)
    for off in range(0,size,0x1000):
      if (chunk:=chunk_map.get(off)) is not None:
        writes.append((cursor+off,chunk))
        maps.append((va+off,pa+off))
      else: maps.append((va+off,SUB_PA+0x1000))
    if va == 0xda00406000:
      for off,chunk in chunks:
        if off <= 8 < off+len(chunk): resource_byte = chunk[8-off]
    cursor += (size + 0xfff) & ~0xfff
  return MTBundle(writes,maps,offsets,cursor,0x100 if resource_byte >= 21 else 0x80)


class MTLaunch(MTFirmware):
  write_map:dict[int,bytes]
  packet:bytearray
  _active:MTLaunch|None = None
  _loaded:MTLaunch|None = None

  def __init__(self, dev:MTDev, bundle:str, packet:str, state:MTBundle|None=None):
    reuse = MTLaunch._active is not None and MTLaunch._active.dev is dev
    super().__init__(dev, prepare=not reuse)
    if reuse:
      assert MTLaunch._active is not None
      self.mmu = MTLaunch._active.mmu
    self.bundle, self.packet_path, self.bundle_state = pathlib.Path(bundle), pathlib.Path(packet), state
    self.sv = self.dev.pci_dev.map_bar(2, off=SUB_PA, size=0x2000000, fmt='B')
    if not reuse: self.sv[:0x20000] = bytes(0x20000)
    self.zero_pa, self.out_pa = SUB_PA + 0x1000, SUB_PA + 0x3000
    self.mmu_dirty = True
    self.backing_pa = 0x40000000
    self.backing = self.dev.pci_dev.map_bar(2, off=self.backing_pa, size=0x10000000, fmt='B')
    self._map_bundle()
    self._prepare_packet()
    self.ccb:MTCCB|None = None
    self.launch_id = 1
    self.generation = 1
    MTLaunch._loaded = self

  def _map_bundle(self) -> None:
    if self.bundle_state is None: self.bundle_state = parse_mt_bundle(self.bundle)
    state = self.bundle_state
    if state.cursor > len(self.backing): raise MemoryError("MT launch bundle exceeds backing store")
    self.backing_offsets, self.backing_cursor, self.text_offset = state.offsets, state.cursor, state.text_offset
    previous = MTLaunch._loaded
    previous_writes = previous.write_map if previous is not None and previous.dev is self.dev else {}
    self.write_map = dict(state.writes)
    for off,data in state.writes:
      if previous_writes.get(off) != data: self.backing[off:off+len(data)] = data
    for base in [0xda00400000,0xe000400000]:
      for off in range(0,0x1000000,0x1000):
        if base+off not in self.mmu.pages: self.mmu.map(base+off,self.zero_pa,MT_PTE_READ_WRITE)
    for va,pa in state.maps:
      if self.mmu.pages.get(va) != pa: self.mmu.map(va,pa,MT_PTE_READ_WRITE)
    for off in range(0x400000, 0x410000, 0x1000): self.mmu.map(FW_DEVVA + off, 0x20000000 + off, MT_PTE_READ_WRITE)
    for i in range(18): self.mmu.map(0xd9bfff0000 + i * 0x1000, self.zero_pa, MT_PTE_READ_WRITE)
    self.mmu.map(0xe1c2003000, self.out_pa, MT_PTE_READ_WRITE)
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
    packet = bytearray(self.packet_path.read_bytes())
    struct.pack_into('<Q', packet, 0x38, 1)
    struct.pack_into('<I', packet, 0x40, 1)
    struct.pack_into('<Q', packet, 0x08, FW_DEVVA + 0x40b000)
    struct.pack_into('<Q', packet, 0xc0 + 0x88, FW_DEVVA + 0x409000)
    struct.pack_into('<Q', packet, 0xc0 + 0x90, FW_DEVVA + 0x40a000)
    previous = MTLaunch._loaded
    for off in range(0,len(packet),0x1000):
      chunk = bytes(packet[off:off+0x1000])
      if previous is None or previous.packet[off:off+len(chunk)] != chunk: self.sv[off:off+len(chunk)] = chunk
      if self.mmu.pages.get(0x8000420000+off) != SUB_PA+off: self.mmu.map(0x8000420000+off,SUB_PA+off,MT_PTE_READ_WRITE)
    self.packet = packet
    self.v[0x405000:0x405010] = struct.pack('<QII', ROOT_BASE, 0xffffffff, 0)

  def load_kernel(self, image:MTKernelImage, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1)) -> None:
    text_off = self.backing_offsets[0xe000000000] + self.text_offset
    self.backing[text_off:text_off+0x80] = image.text.ljust(0x80, b'\0')
    if os.environ.get('MT_LOAD_TEXT_ONLY'): return
    const_off = self.backing_offsets[0x81e1200000]
    args = image.metadata.kernel['.args']
    user_args = [a for a in args if a['.argBase']['.name'] not in ['Global Spill Base','Core Number']
                 and not a['.argBase']['.name'].startswith('threadIdx')]
    clear_start = 0x48 if len(user_args) <= 1 else 0x50
    if not os.environ.get('MT_LOAD_NO_IMMS'):
      self.backing[const_off+clear_start:const_off+0x80] = bytes(0x80-clear_start)
      for imm in image.metadata.immediates:
        if imm.offset >= 18 and imm.value not in [0x8000,0x20000]:
          self.backing[const_off+imm.offset*4:const_off+imm.offset*4+4] = struct.pack('<I', imm.value)
    if os.environ.get('MT_LOAD_NO_RESOURCES'): return
    resource_off = self.backing_offsets[0xda00406000]
    shared = bytearray(self.backing[resource_off+8:resource_off+12])
    shared[0] = image.metadata.kernel['.shared_reg_count']
    self.backing[resource_off+8:resource_off+12] = shared
    temp = bytearray(self.backing[resource_off+0x1c:resource_off+0x20])
    temp[0] = 8 if image.metadata.kernel['.shared_reg_count'] % 2 == 0 else 0
    self.backing[resource_off+0x1c:resource_off+0x20] = temp

  def map_buffer(self, va:int, pa:int, size:int, flags:int=MT_PTE_READ_WRITE) -> None:
    self.mmu.map_range(va, pa, size, flags)
    self.mmu_dirty = True

  def set_args(self, args:tuple[int, ...]) -> None:
    off = self.backing_offsets[0x81e1200000] + 0x38
    for i, arg in enumerate(args): self.backing[off+i*8:off+i*8+8] = struct.pack('<Q', arg)

  def set_grid(self, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> None:
    off = self.backing_offsets[0x81e1400000]
    for i, field in enumerate([0x2c,0x30,0x34]): self.backing[off+field:off+field+4] = struct.pack('<I', global_size[i] - 1)
    block = bytearray(self.backing[off+0x3c:off+0x40])
    block[0] = local_size[2] - 1
    block[1] = (local_size[1] - 1) << 2
    block[2:4] = struct.pack('<H', (local_size[0] - 1) << 4)
    self.backing[off+0x3c:off+0x40] = block

  def start(self, timeout:float=10.0) -> None:
    if MTLaunch._active is not None and MTLaunch._active.dev is self.dev:
      active = MTLaunch._active
      assert active.ccb is not None
      self.v, self.layout, self.ccb = active.v, active.layout, active.ccb
      self.launch_id = active.launch_id + 1
      active.launch_id = self.launch_id
      if MTLaunch._loaded is not self:
        self._map_bundle()
        self._prepare_packet()
        MTLaunch._loaded = self
      struct.pack_into('<Q', self.packet, 0x38, self.launch_id)
      self.generation = active.generation + 1
      active.generation = self.generation
      self.upload_mmu()
      self.v[0x405000:0x405010] = struct.pack('<QII', ROOT_BASE, 0xffffffff, 0)
    else:
      super().start(timeout)
      MTLaunch._active = self
    self._write(0xe150, 0)
    self._write(0xe158, 0)
    self._write(0x138, 0xffffffff, 4)
    self._write(0xe160, 0)
    self._write(0xe198, 0)
    self.mmu_dirty = False
    self.sv[0x3000:0x4000] = bytes(0x1000)
    if self.ccb is None:
      ci = mtfw.MTFW_CCB_ITEM()
      self.ccb = MTCCB(self.dev, self.v, self.layout.item + 3 * 4096 * ci.SIZE, self.layout.ccb + 3 * 0x4030)

  def execute(self) -> MTLaunchResult:
    if self.ccb is None: raise RuntimeError("MT firmware was not started")
    if self.mmu_dirty:
      self.upload_mmu()
      self.mmu_dirty = False
    self.ccb.submit(0x8000420000, len(self.packet), ROOT_BASE, generation=self.generation)
    response = self.ccb.wait(float(os.environ.get('MT_TIMEOUT','5')))
    if response.value:
      fault = (self._read(0xe150), self._read(0xe158), self._read(0x130, 4), self._read(0x4a0))
      raise RuntimeError(f"MT launch failed: {response}, fault={fault}")
    return MTLaunchResult(response, struct.unpack('<Q', self.v[0x40a000:0x40a008])[0], struct.unpack('<I', self.v[0x409008:0x40900c])[0],
                          struct.unpack('<I', self.sv[0x3000:0x3004])[0])
