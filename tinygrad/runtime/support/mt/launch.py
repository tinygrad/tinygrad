from __future__ import annotations
import struct
from typing import Any
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.compute import MTFirmware, SUB_PA, FW_BASE, FW_DEVVA, ROOT_BASE, INIT_SIZE
from tinygrad.runtime.support.mt.ccb import MTCCB
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.elf import MTKernelImage
from tinygrad.runtime.support.mt.pds import compute_pds, PDS_UNUSED
from tinygrad.runtime.support.mt.submission import build_submission, SUBMISSION_VA, ARGS_VA, GRID_VA

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
  cache_activations:int
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
    self.cache_activations = 0
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
      self._write(0xe150, 0)
      self._write(0xe158, 0)
      self._write(0x138, 0xffffffff, 4)
      self._write(0xe160, 0)
      self._write(0xe198, 0)
      ci = mtfw.MTFW_CCB_ITEM()
      self.ccb = MTCCB(self.dev, self.v, self.layout.item + 3 * 4096 * ci.SIZE, self.layout.ccb + 3 * 0x4030)
