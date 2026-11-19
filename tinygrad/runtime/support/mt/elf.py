from __future__ import annotations
import dataclasses, struct
from tinygrad.runtime.support.elf import elf_loader

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

  def template_signature(self) -> str:
    k = self.metadata.kernel
    fields = ['.shared_reg_count','.attr_reg_count','.shared_memory_size',
              '.local_mem_used_in_cluster','.private_memory_size','.has_barrier','.has_texture','.global_atomic_count',
              '.shared_atomic_lock_idx','.barrier_coef','.wave_mode','.coef_reg_count','.slot_reg_count']
    params = tuple((a['.argBase']['.name'],a['.kind'],a['.argBase']['.type'],a['.argBase']['.size']) for a in k['.args'])
    return repr((tuple(k.get(f) for f in fields),params))

  def superset_signature(self) -> str:
    k = self.metadata.kernel
    fields = ['.attr_reg_count','.shared_memory_size','.has_barrier','.has_texture','.global_atomic_count','.shared_atomic_lock_idx',
              '.barrier_coef','.wave_mode','.coef_reg_count']
    return repr(tuple(k.get(f) for f in fields))

  def universal_signature(self) -> str:
    k = self.metadata.kernel
    fields = ['.shared_memory_size','.has_barrier','.has_texture','.global_atomic_count','.shared_atomic_lock_idx',
              '.barrier_coef','.wave_mode','.coef_reg_count']
    return repr(tuple(k.get(f) for f in fields))

  def launch_signature(self) -> str:
    k = self.metadata.kernel
    fields = ['.shared_reg_count','.attr_reg_count','.shared_memory_size',
              '.local_mem_used_in_cluster','.private_memory_size','.has_barrier','.has_texture','.global_atomic_count',
              '.shared_atomic_lock_idx','.barrier_coef','.wave_mode','.coef_reg_count','.slot_reg_count']
    params = tuple((a['.argBase']['.name'],a['.kind'],a['.argBase']['.type'],a['.argBase']['.size']) for a in k['.args'])
    return repr(((self.entry,len(self.text),k['.temp_reg_count'],tuple(k.get(f) for f in fields)),params))
