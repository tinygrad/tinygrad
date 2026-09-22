import struct, unittest
from pathlib import Path
from test.mockgpu.qcom.ir3 import execute, execute_group

END = bytes.fromhex('0000000000000003')
LDG = bytes.fromhex('01008001020006c0')  # ldg.u32 r0.z, g[r0.x], 1
STG = bytes.fromhex('04008001000bc6c0')  # stg.u32 g[r1.y], r0.z, 1
SYNC = bytes.fromhex('0200020002001052')  # (sy)add.u r0.z, r0.z, r0.z

class TestIR3Memory(unittest.TestCase):
  def test_group_failure_only_changes_submission_snapshot(self):
    store = bytes.fromhex('04008001000bc6c0')  # stg.u32 g[r1.y], r0.z, 1
    invalid = bytes.fromhex('00000000000000e0')
    original = {0x1000: bytearray.fromhex('00000000')}
    snapshot = {base: data.copy() for base, data in original.items()}
    with self.assertRaisesRegex(ValueError, 'unsupported'):
      execute_group(store + invalid + bytes.fromhex('0000000000000003'), [{2: 7, 5: 0x1000, 6: 0}], {}, snapshot, 0)
    self.assertEqual(original[0x1000], bytes.fromhex('00000000'))
    self.assertEqual(snapshot[0x1000], bytes.fromhex('07000000'))

  def test_load_sync_store(self):
    memory = {0x1000: bytearray.fromhex('01000080'), 0x2000: bytearray.fromhex('aaaaaaaa')}
    regs = {0: 0x1000, 1: 0, 5: 0x2000, 6: 0}
    result = execute(LDG + SYNC + STG + END, regs, memory=memory)
    self.assertEqual(result[2], 2)
    self.assertEqual(memory[0x2000], bytes.fromhex('02000000'))
    self.assertEqual(memory[0x1000], bytes.fromhex('01000080'))
    self.assertNotIn(2, regs)

  def test_high_address_and_guards(self):
    memory = {0x100001000: bytearray.fromhex('aaaaaaaa 78563412 bbbbbbbb')}
    execute(LDG + SYNC + STG + END, {0: 0x1004, 1: 1, 5: 0x1004, 6: 1}, memory=memory)
    self.assertEqual(memory[0x100001000], bytes.fromhex('aaaaaaaa f0ac6824 bbbbbbbb'))

  def test_missing_sync_and_overwrite(self):
    for tail in (END, STG + END, bytes.fromhex('0200020002001042') + END,
                 bytes.fromhex('0000000002c02c20') + END,
                 bytes.fromhex('0000000000c02c20') + END, LDG + END):
      with self.subTest(tail=tail.hex()):
        with self.assertRaisesRegex(ValueError, 'pc=0x8.*(needs sy|need sy)'):
          execute(LDG + tail, {0: 0x1000, 1: 0, 5: 0x1000, 6: 0}, {0: 0}, {0x1000: bytearray(4)})

  def test_missing_address_halves(self):
    for regs, missing in (({}, 0), ({0: 0x1000}, 1)):
      with self.subTest(regs=regs):
        with self.assertRaisesRegex(ValueError, f'pc=0x0.*uninitialized register {missing}'):
          execute(LDG + END, regs)

  def test_address_high_overwrite_requires_sync(self):
    move = bytes.fromhex('0000000001c02c20')  # mov.u32u32 r0.y, c0.x
    memory = {0x1000: bytearray.fromhex('03000000')}
    with self.assertRaisesRegex(ValueError, 'pc=0x8.*register 1 needs sy/ss before overwrite'):
      execute(LDG + move + END, {0: 0x1000, 1: 0}, {0: 9}, memory)
    result = execute(LDG + SYNC + move + END, {0: 0x1000, 1: 0}, {0: 9}, memory)
    self.assertEqual((result[1], result[2]), (9, 6))

  def test_multiple_region_rollback_after_sync(self):
    second_store = bytes.fromhex('04008001000fc6c0')  # stg.u32 g[r1.w], r0.z, 1
    memory = {0x1000: bytearray.fromhex('03000000'), 0x2000: bytearray.fromhex('aaaaaaaa'),
              0x3000: bytearray.fromhex('bbbbbbbb')}
    before = {base: bytes(data) for base, data in memory.items()}
    regs = {0: 0x1000, 1: 0, 5: 0x2000, 6: 0, 7: 0x3000, 8: 0}
    prefix = LDG + SYNC + STG + second_store
    with self.assertRaisesRegex(ValueError, 'pc=0x20.*unsupported'):
      execute(prefix + bytes.fromhex('00000000000000e0') + END, regs, memory=memory)
    self.assertEqual(memory, before)
    execute(prefix + END, regs, memory=memory)
    self.assertEqual(memory[0x2000], bytes.fromhex('06000000'))
    self.assertEqual(memory[0x3000], bytes.fromhex('06000000'))

  def test_memory_data_register_boundaries(self):
    for reg in range(244, 256):
      for load in (True, False):
        with self.subTest(reg=reg, load=load):
          word = (0xc006000001800001 | (reg << 32)) if load else (0xc0c60b0001800000 | (reg << 1))
          memory = {0x1000: bytearray.fromhex('03000000')}
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*special/reserved'):
            execute(struct.pack('<Q', word) + END, {0: 0x1000, 1: 0, 5: 0x1000, 6: 0}, memory=memory)
          self.assertEqual(memory[0x1000], bytes.fromhex('03000000'))

  def test_invalid_address_registers(self):
    for addr_reg in (243, 244, 255):
      for load in (True, False):
        with self.subTest(addr_reg=addr_reg, load=load):
          word = (0xc006000201800001 | (addr_reg << 14)) if load else (0xc0c6010001800004 | (addr_reg << 41))
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*special/reserved'):
            execute(struct.pack('<Q', word) + END, {243: 0})

  def test_address_bounds(self):
    for addr in (0, 0xffc, 0x1001, 0x1004, 0xfffffffffffffffc):
      for instr in (LDG, STG):
        with self.subTest(addr=addr, instr=instr.hex()):
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*(unmapped range|invalid address)'):
            execute(instr + END, {0: addr & 0xffffffff, 1: addr >> 32, 2: 1, 5: addr & 0xffffffff, 6: addr >> 32},
                    memory={0x1000: bytearray(4)})
    with self.assertRaisesRegex(ValueError, 'unmapped range'):
      execute(LDG + END, {0: 0x1000, 1: 0}, memory={0x1000: bytearray(2), 0x1002: bytearray(2)})

  def test_invalid_regions(self):
    shared = bytearray(4)
    for memory in ({-1: bytearray(4)}, {(1 << 64)-2: bytearray(4)}, {0: bytearray()}, {0: bytes(4)},
                   {0: bytearray(8), 4: bytearray(4)}, {0: shared, 4: shared}, {False: bytearray(4)}):
      with self.subTest(memory=memory):
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*(invalid memory|overlapping|aliased)'):
          execute(END, {}, memory=memory)  # type: ignore[arg-type]

  def test_encoding_allowlist(self):
    for raw, variable in ((LDG, set(range(14, 22)) | set(range(32, 40)) | set(range(24, 27)) | {49, 50, 51, 54, 60}),
                          (STG, set(range(1, 9)) | set(range(41, 49)) | set(range(24, 27)) | set(range(49, 52)) | {60})):
      for bit in set(range(64)) - variable:
        with self.subTest(raw=raw.hex(), bit=bit):
          changed = struct.pack('<Q', int.from_bytes(raw, 'little') ^ (1 << bit))
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*unsupported'):
            execute(changed + END, {0: 0x1000, 1: 0, 2: 1, 5: 0x1000, 6: 0}, memory={0x1000: bytearray(4)})

  def test_pending_end_and_failure_rollback(self):
    memory = {0x1000: bytearray.fromhex('78563412')}
    for tail in (LDG + END, bytes.fromhex('00000000000000e0') + END, b''):
      with self.subTest(tail=tail.hex()):
        with self.assertRaises(ValueError):
          execute(STG + tail, {0: 0x1000, 1: 0, 2: 42, 5: 0x1000, 6: 0}, memory=memory)
        self.assertEqual(memory[0x1000], bytes.fromhex('78563412'))

  def test_store_then_load(self):
    memory = {0x1000: bytearray(4)}
    execute(STG + LDG + SYNC + END, {0: 0x1000, 1: 0, 2: 21, 5: 0x1000, 6: 0}, memory=memory)
    self.assertEqual(memory[0x1000], bytes.fromhex('15000000'))
    result = execute(LDG + SYNC + END, {0: 0x1000, 1: 0}, memory=memory)
    self.assertEqual(result[2], 42)

  def test_last_address_and_register(self):
    memory = {(1 << 64)-4: bytearray.fromhex('01000000')}
    result = execute(LDG + SYNC + END, {0: 0xfffffffc, 1: 0xffffffff}, memory=memory)
    self.assertEqual(result[2], 2)
    load = struct.pack('<Q', 0xc00600f301800001 | (242 << 14))
    sync = struct.pack('<Q', 0x521000f300f300f3)
    self.assertEqual(execute(load + sync + END, {242: 0x1000, 243: 0}, memory={0x1000: bytearray.fromhex('03000000')})[243], 6)

  def test_uninitialized_store_value(self):
    with self.assertRaisesRegex(ValueError, 'pc=0x0.*uninitialized register 2'):
      execute(STG + END, {5: 0x1000, 6: 0}, memory={0x1000: bytearray(4)})

  def test_two_loads_one_sync(self):
    second = bytes.fromhex('01408001070006c0')  # address at r0.y/r0.z, result r1.w
    sync = bytes.fromhex('0300070004001052')  # (sy)add.u r1.x, r0.w, r1.w
    first = bytes.fromhex('01008001030006c0')
    memory = {0: bytearray.fromhex('05000000'), 0x1000: bytearray.fromhex('07000000')}
    self.assertEqual(execute(first + second + sync + END, {0: 0x1000, 1: 0, 2: 0}, memory=memory)[4], 12)

  def test_nop_and_modifiers(self):
    for repeat in range(6):
      for delay in range(4):
        with self.subTest(repeat=repeat, delay=delay):
          nop = struct.pack('<Q', repeat << 40)
          sync = struct.pack('<Q', int.from_bytes(SYNC, 'little') | ((delay & 1) << 43) | ((delay >> 1) << 51))
          result = execute(LDG + nop + sync + END, {0: 0x1000, 1: 0}, memory={0x1000: bytearray.fromhex('03000000')})
          self.assertEqual(result[2], 6)
    for repeat in (6, 7):
      with self.subTest(repeat=repeat):
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*unsupported'): execute(struct.pack('<Q', repeat << 40) + END, {})
    with self.assertRaisesRegex(ValueError, 'pc=0x48.*need sy'):
      execute(LDG + bytes(64) + END, {0: 0x1000, 1: 0}, memory={0x1000: bytearray(4)})
    for bit in set(range(64)) - {40, 41, 42, 44, 55, 56, 61}:  # br/jump have independent flow tests.
      with self.subTest(bit=bit):
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*unsupported'): execute(struct.pack('<Q', 1 << bit) + END, {})

  def test_padding_policy(self):
    self.assertEqual(execute(END + bytes(248), {}, padded=True), {})
    for program in (END + bytes(8), END + bytes(256), END + bytes(240) + END):
      with self.subTest(size=len(program)):
        with self.assertRaisesRegex(ValueError, 'trailing'): execute(program, {}, padded=True)
    with self.assertRaisesRegex(ValueError, 'missing end'): execute(bytes(256), {}, padded=True)

  def test_compiled_program(self):
    code = (Path(__file__).parent / 'fixtures/compute/scalar.bin').read_bytes()
    for a, b, expected in ((0, 0, 0), (1, 2, 3), (0xffffffff, 1, 0), (0x7fffffff, 1, 0x80000000), (0xffffffff, 0xffffffff, 0xfffffffe)):
      with self.subTest(a=a, b=b):
        memory = {0x100001000: bytearray(a.to_bytes(4, 'little')), 0x200002000: bytearray(b.to_bytes(4, 'little')),
                  0x300003000: bytearray.fromhex('aaaaaaaa')}
        constants = {0: 0x3000, 1: 3, 2: 0x1000, 3: 1, 4: 0x2000, 5: 2}
        execute(code, {}, constants, memory, padded=True)
        self.assertEqual(memory[0x300003000], expected.to_bytes(4, 'little'))
        self.assertEqual(memory[0x100001000], a.to_bytes(4, 'little'))
        self.assertEqual(memory[0x200002000], b.to_bytes(4, 'little'))

if __name__ == '__main__': unittest.main()
