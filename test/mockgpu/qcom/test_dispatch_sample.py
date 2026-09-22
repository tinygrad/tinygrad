import hashlib, json, struct, unittest
from pathlib import Path
from test.mockgpu.qcom.compute import QCOMCompute
from test.mockgpu.qcom.test_compute import packet, register, scalar_state

class TestDispatchSample(unittest.TestCase):
  def test_fixture_manifest(self):
    folder = Path(__file__).parent / 'fixtures/compute'
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, metadata in manifest.items():
      if name == 'scalar_execution': continue
      with self.subTest(profile=name):
        program = (folder / metadata['program']).read_bytes()
        self.assertEqual(hashlib.sha256(program).hexdigest(), metadata['sha256'])
        self.assertEqual(len(program), metadata.get('bytes', metadata.get('code_bytes')))
    spec, metadata = manifest['scalar_execution'], manifest['ir3-compiled-add/1']
    self.assertEqual(spec['program_sha256'], metadata['sha256'])
    self.assertEqual(spec['global_size'], metadata['global_size'])
    self.assertEqual(spec['local_size'], metadata['local_size'])
    self.assertLessEqual(len(spec['constant_dwords']), metadata['constlen_vec4'] * 4)
    for index, name in ((0, 'output'), (2, 'input_a'), (4, 'input_b')):
      self.assertEqual(spec['constant_dwords'][index] | (spec['constant_dwords'][index+1] << 32), spec['regions'][name])

  def test_compiled_groups_and_address_carry(self):
    for profile, size, local in (('', 256, 32), ('', 1024, 32), ('', 4096, 32), ('production', 1024, 128), ('production', 4096, 128)):
      groups = size // (local*4)
      folder = Path(__file__).parent / 'fixtures/compute'
      key = f'ir3-dispatch-add/{profile + "/" if profile else ""}{size}'
      metadata = json.loads((folder / 'manifest.json').read_text())[key]
      program = (folder / metadata['program']).read_bytes()
      self.assertEqual(hashlib.sha256(program).hexdigest(), metadata['sha256'])
      self.assertEqual((metadata['bytes'], metadata['global'], metadata['local'], metadata['wgid'], metadata['lid']),
                       (384, [groups, 1, 1], [local, 1, 1], 192, 0))
      for address in (0x100000100, 0x1ffffff00):
        with self.subTest(size=size, address=address):
          addresses = (address, address + 0x10000, address + 0x20000)
          left = [(i * 1234567 + 0xffffffff) & 0xffffffff for i in range(size)]
          right = [i + 1 for i in range(size)]
          def pack(values): return bytearray(struct.pack(f'<{len(values)}I', *values))
          memory = {addresses[0]: bytearray(b'\xaa' * (size*4)), addresses[1]: pack(left), addresses[2]: pack(right),
                    0x4000: bytearray(program)}
          raw = scalar_state() + register(0xbb08, 0x60) + register(0xbb08, 0)
          raw += register(0xa9bc, 3) + packet(0x34, 0xf60000, 0x4000, 0)
          constants = [part for addr in addresses for part in (addr & 0xffffffff, addr >> 32)] + [0, 0]
          raw += packet(0x34, 0xb44000, 0, 0, *constants)
          raw += register(0xb990, 3 | ((local-1) << 2), size//4, 0, 1, 0, 1, 0, 0x00fcfcc0, 0xfc, groups, 1, 1)
          raw += packet(0x33, 0, groups, 1, 1)
          gpu = QCOMCompute()
          gpu.run(raw, memory)
          self.assertEqual(memory[addresses[0]], pack([(a+b) % (1 << 32) for a, b in zip(left, right)]))
          self.assertEqual((memory[addresses[1]], memory[addresses[2]]), (pack(left), pack(right)))
          self.assertEqual(gpu.dispatches, 1)
          memory[addresses[1]][:4] = bytes(4)
          before = {base: data.copy() for base, data in memory.items()}
          with self.assertRaisesRegex(ValueError, 'unsupported packet'): gpu.run(raw + packet(0x10), memory)
          self.assertEqual(memory, before)
          self.assertEqual(gpu.dispatches, 1)
          memory[addresses[0]] = bytearray(b'\xaa' * (size*4-4))
          before = {base: data.copy() for base, data in memory.items()}
          with self.assertRaisesRegex(ValueError, 'unmapped range'): gpu.run(raw, memory)
          self.assertEqual(memory, before)
          self.assertEqual(gpu.dispatches, 1)

if __name__ == '__main__': unittest.main()
