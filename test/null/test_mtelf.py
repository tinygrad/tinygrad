import struct, unittest
from tinygrad.runtime.support.mt.elf import MTLaunchMetadata, MTImmediate

def pack(x):
  if isinstance(x, int):
    if 0 <= x < 0x80: return bytes([x])
    if x < 0x100: return b'\xcc' + x.to_bytes(1, 'big')
    return b'\xcd' + x.to_bytes(2, 'big')
  if isinstance(x, str):
    raw = x.encode()
    if len(raw) < 0x20: return bytes([0xa0 | len(raw)]) + raw
    return b'\xd9' + bytes([len(raw)]) + raw
  if isinstance(x, list): return bytes([0x90 | len(x)]) + b''.join(pack(v) for v in x)
  if isinstance(x, dict): return bytes([0x80 | len(x)]) + b''.join(pack(k) + pack(v) for k,v in x.items())
  raise TypeError(type(x))

class TestMTElf(unittest.TestCase):
  def test_launch_metadata(self):
    payload = pack({'mtgpu.kernels':[{'.imms':[{'.immBase':{'.offset':14,'.value':42}}, {'.immBase':{'.offset':15,'.value':43}}],
                                      '.temp_reg_count':4}], 'mtgpu.version':1})
    note = struct.pack('<III', 6, len(payload), 0x40) + b'MTGPU\0\0\0' + payload
    md = MTLaunchMetadata.from_note(note)
    self.assertEqual(md.immediates, (MTImmediate(0,14,42), MTImmediate(0,15,43)))
    self.assertEqual(md.kernel['.temp_reg_count'], 4)

  def test_invalid_note(self):
    with self.assertRaises(ValueError): MTLaunchMetadata.from_note(bytes(20))

if __name__ == '__main__': unittest.main()
