from __future__ import annotations
import struct
from tinygrad.runtime.support.mt.launch import MTLaunch
from tinygrad.runtime.support.mt.elf import MTKernelImage
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.submission import CDM_EXECUTION_ADDRESS, ARGS_VA, GRID_VA, RESOURCE_VA, TEXT_VA, RESUME_VA, PDS_MAIN_VA, TASK_VA

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
    # Firmware resume and main PDS programs. These instruction words are fixed
    # by the generation-1 PDS ABI, not per-kernel captured launch data.
    resume = bytearray(0x4000)
    def words(off:int, *values:int) -> None: struct.pack_into(f'<{len(values)}I',resume,off,*values)
    words(0x20,0xf4000002)
    words(0x30,0xd2800000)
    words(0x40,0xf0001001,0xf4000012)
    words(0x50,0xf0001001,0xf4000012)
    def doutw(off:int) -> None: words(off,0xf0002000,0xf400180a)
    def doutd(off:int) -> None: words(off,0xdcb40003,0xf0001301,0xd2800000,0xf0001b01,0xfc000012)
    for base in range(0x80,0x1800,0x100):
      doutw(base)
      doutd(base+0x80)
    words(0x1880,0xdcb40002,0xd2800000,0xf0001001,0xf4000012)
    words(0x1900,0xf0001001,0xf4000012)
    words(0x1980,0xf4001005,0xd2800000)
    for off,address,size in [(0x1a00,0x81e08a1180,0x1000000600),(0x1a20,0x81e08a1180,0x2000000800),
                             (0x1a40,0x81e08a5180,0x1000000600),(0x1a60,0x81e08a5180,0x2000000800)]:
      struct.pack_into('<QI4xQ',resume,off,address,0x80000043,size)
    struct.pack_into('<QII',resume,0x1a80,0x81e08a9180,0,0x21000000)
    words(0x1aa0,0xd1340002,0xd1300040,0xa766ff8d,0xb1acc005,0xe0650107,0xd0080067,0xd0200002,0xd0800000,0xd2800000)
    struct.pack_into('<QII',resume,0x1b00,0x81e08a9180,0,0x30400000)
    words(0x1b10,0xd1340002,0xd1300040,0xa766ff8d,0xb1acc005,0xe0650107,0xd0000067,0xd0800000,0xd2800000)
    words(0x1b80,0xdcb40002,0xd2800000,0xf0001001,0xf4000012)
    words(0x1c00,0xf0001001,0xf4000012)
    words(0x1c80,0xf4001005,0xd2800000)
    for base in range(0x1d00,0x3500,0x100):
      doutw(base)
      doutd(base+0x80)
    words(0x3500,0xf4001000)
    args = self.generated_image.metadata.kernel['.args']
    has_block = any(a['.kind'] in (5,6,7) for a in args)
    has_local = any(a['.kind'] in (1,2,3) for a in args)
    if self.generated_image.metadata.kernel['.has_barrier']:
      words(0x3580,0xdcb40004,0xf0001301,0xf0001801,0xd2800000,0xf0002301,0xfc00001a)
    elif has_block: doutd(0x3580)
    elif has_local: words(0x3580,0xdcb40002,0xd2800000,0xf0001301,0xfc000012)
    else: words(0x3580,0xdcb40002,0xd2800000,0xfc00000a)
    words(0x3600,0xf4001005)
    self._write_backing(self.backing_offsets[RESUME_VA],resume)
    main = bytearray(0x1000)
    for off,word in {0x00:0xf4001000,0x10:0xf4000002,0x20:0xf0001000,0x24:0xf4000012}.items():
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
