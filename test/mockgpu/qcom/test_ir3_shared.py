import struct, unittest
from test.mockgpu.qcom.ir3 import execute_group

def code(*words): return struct.pack(f'<{len(words)}Q', *words)
STORE = 0xc106010001800004  # stl.u32 l[r0.x], r0.z, 1
LOAD = 0xc046000301804001   # ldl.u32 r0.w, l[r0.y], 1
BARRIER, END = 0xe042000000000000, 0x0300000000000000
ADD_SS = 0x4210100400030003

class TestIR3Shared(unittest.TestCase):
  def test_shift_and_mad_ss(self):
    for instruction, expected in ((0x4710100420010003, 0x1fc00000), (0x6381100400020003, 0x40000000)):
      with self.subTest(instruction=instruction):
        program = code(STORE, BARRIER, LOAD, instruction, END)
        self.assertEqual(execute_group(program, [{0: 0, 1: 0, 2: 0x3f800000}], {}, {}, 4)[0][4], expected)
        with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
          execute_group(code(STORE, BARRIER, LOAD, instruction & ~(1 << 44), END),
                        [{0: 0, 1: 0, 2: 0x3f800000}], {}, {}, 4)

  def test_barrier_exchanges_lane_values(self):
    initial = [{0: 0, 1: 4, 2: 10}, {0: 4, 1: 0, 2: 20}]
    results = execute_group(code(STORE, BARRIER, LOAD, ADD_SS, END), initial, {}, {}, 8)
    self.assertEqual([r[4] for r in results], [40, 20])
    self.assertEqual(initial, [{0: 0, 1: 4, 2: 10}, {0: 4, 1: 0, 2: 20}])

  def test_shared_load_needs_ss(self):
    for sync in (0, 1 << 60):
      with self.subTest(sync=sync):
        with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
          execute_group(code(STORE, BARRIER, LOAD, 0x4210000400030003 | sync, END), [{0: 0, 1: 0, 2: 7}], {}, {}, 4)
    with self.assertRaisesRegex(ValueError, 'need ss'):
      execute_group(code(STORE, BARRIER, LOAD, END), [{0: 0, 1: 0, 2: 7}], {}, {}, 4)

  def test_repeated_barriers_and_group_isolation(self):
    program = code(STORE, BARRIER, LOAD, ADD_SS, 0x200cc00200000004, STORE, BARRIER, LOAD, ADD_SS, END)
    initial = [{0: 0, 1: 4, 2: 10}, {0: 4, 1: 0, 2: 20}]
    self.assertEqual([r[4] for r in execute_group(program, initial, {}, {}, 8)], [40, 80])
    with self.assertRaisesRegex(ValueError, 'uninitialized shared'):
      execute_group(code(LOAD, ADD_SS, END), [{1: 0}], {}, {}, 8)

  def test_shared_address_overwrite_needs_ss(self):
    move = 0x204cc00100000000
    for sync in (0, 1 << 44):
      with self.subTest(sync=sync):
        program = code(STORE, BARRIER, LOAD, move | sync, ADD_SS, END)
        if sync: self.assertEqual(execute_group(program, [{0: 0, 1: 0, 2: 7}], {}, {}, 4)[0][4], 14)
        else:
          with self.assertRaisesRegex(ValueError, 'before overwrite'):
            execute_group(program, [{0: 0, 1: 0, 2: 7}], {}, {}, 4)

  def test_compare_ss_publishes_shared_load(self):
    raw = code(STORE, BARRIER, LOAD, 0x4290500200050003, 0x2009400600000002, END)
    self.assertEqual(execute_group(raw, [{0: 0, 1: 0, 2: 7, 5: 8}], {}, {}, 4)[0][6], 1)

  def test_conversion_ss_publishes_shared_load(self):
    for sync in (0, 1 << 44):
      with self.subTest(sync=sync):
        raw = code(STORE, BARRIER, LOAD, 0x200c400400000003 | sync, END)
        if sync: self.assertEqual(execute_group(raw, [{0: 0, 1: 0, 2: 7}], {}, {}, 4)[0][4], 0x40e00000)
        else:
          with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
            execute_group(raw, [{0: 0, 1: 0, 2: 7}], {}, {}, 4)

  def test_uninitialized_racing_and_out_of_bounds(self):
    with self.assertRaisesRegex(ValueError, 'uninitialized shared'):
      execute_group(code(STORE, LOAD, ADD_SS, END), [{0: 0, 1: 4, 2: 10}, {0: 4, 1: 0, 2: 20}], {}, {}, 8)
    with self.assertRaisesRegex(ValueError, 'racing shared writes'):
      execute_group(code(STORE, BARRIER, END), [{0: 0, 2: 10}, {0: 0, 2: 20}], {}, {}, 4)
    for addr in (1, 4, 0xfffffffc):
      with self.subTest(addr=addr):
        with self.assertRaisesRegex(ValueError, 'invalid shared range'):
          execute_group(code(STORE, END), [{0: addr, 2: 10}], {}, {}, 4)
    with self.assertRaisesRegex(ValueError, 'uninitialized shared'):
      execute_group(code(LOAD, ADD_SS, END), [{1: 0}], {}, {}, 4)

  def test_predicated_store_and_unknown_instruction(self):
    program = code(0x42b400f820000000, 0x0682000000000000, 0xc0c6050001800008, 0x0782000000000000, END)
    memory = {0x1000: bytearray(4)}
    execute_group(program, [{0: 0, 2: 0x1000, 3: 0, 4: 7}, {0: 1}], {}, memory, 0)
    self.assertEqual(memory[0x1000], b'\x07\0\0\0')
    with self.assertRaisesRegex(ValueError, 'unsupported predicated'):
      execute_group(code(0x42b400f820000000, 0x0682000000000000, BARRIER, 0x0782000000000000, END), [{0: 1}], {}, {}, 0)

if __name__ == '__main__': unittest.main()
