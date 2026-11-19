from __future__ import annotations
import struct
from tinygrad.runtime.support.mt.launch import MTLaunch
from tinygrad.runtime.support.mt.elf import MTKernelImage
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE

# Device-VA to backing-PA layout used by the generated compute context. This is
# an allocation map, not captured launch state.
GENERATED_OFFSETS = {
  0x100000000:0x0,0x100400000:0x200000,0x100800000:0x400000,0x100a00000:0x600000,
  0x101a00000:0x1600000,0x102a00000:0x2600000,0x102b00000:0x2700000,0x102c00000:0x2800000,
  0x102d00000:0x2900000,0x102e00000:0x2a00000,0x102f00000:0x2b00000,0x103000000:0x2c00000,
  0x103100000:0x2d00000,0x103200000:0x2e00000,0x103300000:0x2f00000,0x103400000:0x3000000,
  0x103500000:0x3100000,0x103600000:0x3200000,0x103700000:0x3300000,0x103800000:0x3400000,
  0x103900000:0x3500000,0x103a00000:0x3600000,0x103b00000:0x3700000,0x103c00000:0x3800000,
  0x103d00000:0x3900000,0x103e00000:0x3a00000,0x103f00000:0x3b00000,0x104000000:0x3c00000,
  0x104100000:0x3d00000,0x104200000:0x3e00000,0x104300000:0x3f00000,0x104400000:0x4000000,
  0x104500000:0x4100000,0x104600000:0x4200000,0x104700000:0x4300000,0x104800000:0x4400000,
  0x104900000:0x4500000,0x104a00000:0x4600000,0x105a00000:0x5600000,0x105b00000:0x5700000,
  0x105c00000:0x5800000,0x105d00000:0x5900000,0x105e00000:0x5a00000,0x105f00000:0x5b00000,
  0x106000000:0x5c00000,0x106100000:0x5d00000,0x106200000:0x5e00000,0x106300000:0x5f00000,
  0x106400000:0x6000000,0x106500000:0x6100000,0x106600000:0x6200000,0x106700000:0x6300000,
  0x106800000:0x6400000,0x106900000:0x6500000,0x106a00000:0x6600000,0x106b00000:0x6700000,
  0x106c00000:0x6800000,0x106d00000:0x6900000,0x106e00000:0x6a00000,0x106f00000:0x6b00000,
  0x107000000:0x6c00000,0x107100000:0x6d00000,0x107200000:0x6e00000,0x107300000:0x6f00000,
  0x107400000:0x7000000,0x107500000:0x7100000,0x107600000:0x7200000,0x107700000:0x7300000,
  0x107800000:0x7400000,0x107900000:0x7500000,0x8000000000:0x8600000,0x8000400000:0x8a00000,
  0x81e0802000:0x8e00000,0x81e0812000:0x8e10000,0x81e0814000:0x8e12000,0x81e0815000:0x8e13000,
  0x81e081e000:0x8e1c000,0x81e0820000:0x8e1e000,0x81e0830000:0x8e1f000,0x81e0840000:0x8e2f000,
  0x81e0849000:0x8e38000,0x81e0861000:0x8e50000,0x81e08c9000:0x8eb8000,0x81e08d2000:0x8ec1000,
  0x81e0a00000:0x8ed9000,0x81e0e00000:0x92d9000,0x81e1200000:0x96d9000,0x81e1400000:0x98d9000,
  0xda00000000:0x9ad9000,0xda00400000:0x9ed9000,0xda00401000:0x9eda000,0xda00402000:0x9edb000,
  0xda00403000:0x9edc000,0xda00404000:0x9edd000,0xda00405000:0x9ede000,0xda00406000:0x9edf000,
  0xe000000000:0xa0df000,0xe000400000:0xa4df000,0xe000402000:0xa4e1000,0xe000408000:0xa4e7000,
  0xf000000000:0xa4ed000,0xf000400000:0xa8ed000,0x100804000:0xb0ed000,0x100805000:0xb0ee000,
  0x100820000:0xb0ef000,0x100821000:0xb0f0000,0x100822000:0xb0f1000,0x100823000:0xb0f2000,
  0x100824000:0xb0f3000,0x100825000:0xb0f4000,0x100826000:0xb0f5000,0x100827000:0xb0f6000,0x100828000:0xb0f7000,
}

class MTGeneratedLaunch(MTLaunch):
  _text_span = 0

  def generate_task_page(self) -> None:
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
    self._write_backing(self.backing_offsets[0x100804000],page)

  def generate_static_pds(self) -> None:
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
    self._write_backing(self.backing_offsets[0xda00000000],resume)
    main = bytearray(0x1000)
    for off,word in {0x00:0xf4001000,0x10:0xf4000002,0x20:0xf0001000,0x24:0xf4000012}.items():
      struct.pack_into('<I',main,off,word)
    self._write_backing(self.backing_offsets[0xda00400000],main)

  def _map_generated(self) -> None:
    self.backing_offsets = GENERATED_OFFSETS.copy()
    self.backing_cursor = 0xb0f8000
    self.write_map = {}
    for va in (0xda00406000,0x81e1200000,0x81e1400000,0x81e0e00000):
      self._write_backing(self.backing_offsets[va],bytes(0x1000))
    self.text_va = 0xe000000000
    self.text_offset = 0x2000
    self.free_text_offset = 0x30000
    self.pds_execution_address = 0x81e0e00000
    self.pds_code_off = self.backing_offsets[0xda00400000]
    self.pds_data_offsets = [self.backing_offsets[0x100804000]]
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
    resource_off = self.backing_offsets[0xda00406000]
    self.template_text = image.text
    self.text_offset = 0x1000 if len(image.text) <= 0x1000 else 0x2000 if len(image.text) <= 0x20000 else 0x30000
    self.template_temp_count = k['.temp_reg_count']
    self.template_shared_count = k['.shared_reg_count']
    self.resource_temp_offset = 0x2c if has_barrier else 0x24 if has_ids else 0x1c
    self.resource_text_field = 0x28 if has_barrier else 0x20 if has_ids else 0x18
    self.resource_temp_delta = 0
    self.resource_temp_flags = 1 if has_barrier else 0
    self.free_text_offset = 0x30000
    self._write_backing(resource_off,struct.pack('<Q',0x81e1200000))
    self._write_backing(resource_off+8,struct.pack('<I',0x90000000|k['.shared_reg_count']))
    self._write_backing(self.backing_offsets[0x81e1200000]+8,struct.pack('<I',0xff))
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
