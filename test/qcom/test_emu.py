import ctypes, struct, pytest
import test.mockgpu.qcom.emu as emu
from test.mockgpu.qcom.emu import decode_ir3, execute_dispatch, local_id_regs, workgroup_id_regs

def test_decode_add_f():
  inst = decode_ir3(bytes.fromhex('0b0010000b081850'), gpu_id=630)[0]
  assert inst.name == 'add.f'
  assert inst.dst == ('r', 2, 3)
  assert inst.srcs == (('r', 2, 3), ('r', 4, 0))
  assert (inst.sy, inst.nop) == (True, 3)

def _bits(values):
  return [struct.unpack('<I', struct.pack('<f', x))[0] for x in values]

def test_execute_add_f():
  regs = {('r', 2, 3): _bits([1, 2, 3]), ('r', 4, 0): _bits([4, 5, 6])}
  emu.execute_ir3(bytes.fromhex('0b0010000b081850'), regs)
  assert regs[('r', 2, 3)] == _bits([5, 7, 9])

def test_repeated_add_f_is_rejected():
  with pytest.raises(NotImplementedError, match='repeated add.f is unsupported'):
    decode_ir3(bytes.fromhex('0f001400180b1850'))

def test_saturated_add_f_is_rejected():
  with pytest.raises(NotImplementedError, match='unsupported IR3 modifier SAT'):
    decode_ir3(bytes.fromhex('0b0010000b0c1850'))

def test_invalid_instruction_is_rejected():
  with pytest.raises(ValueError, match='invalid IR3 instruction encoding'):
    decode_ir3(bytes.fromhex('ffffffffffffffff'))

def test_decode_nop():
  inst = decode_ir3(bytes(8), gpu_id=630)[0]
  assert inst.name == 'nop'

def test_execute_nop():
  regs: dict[tuple[str, int, int], list[int]] = {}
  emu.execute_ir3(bytes(8), regs)
  assert regs == {}

def test_local_id_regs():
  assert local_id_regs((3, 1, 1), 7) == {
    ('r', 7, 0): [0, 1, 2],
    ('r', 7, 1): [0, 0, 0],
    ('r', 7, 2): [0, 0, 0],
  }

def test_workgroup_id_regs():
  assert workgroup_id_regs((2, 3, 4), 2, 48) == {
    ('r', 48, 0): [2, 2],
    ('r', 48, 1): [3, 3],
    ('r', 48, 2): [4, 4],
  }

def test_execute_dispatch_nop():
  regs = execute_dispatch(bytes(8), (1, 1, 1), (3, 1, 1), 7)
  assert regs == {
    ('r', 7, 0): [0, 1, 2],
    ('r', 7, 1): [0, 0, 0],
    ('r', 7, 2): [0, 0, 0],
  }

def test_execute_dispatch_workgroups(monkeypatch):
  seen = []
  monkeypatch.setattr(emu, 'execute_ir3', lambda _code, regs, **_kwargs:
                      seen.append((regs[('r', 48, 0)], regs[('r', 7, 0)])))
  execute_dispatch(bytes(8), (2, 1, 1), (2, 1, 1), 7, workgroup_id_register=48)
  assert seen == [([0, 0], [0, 1]), ([1, 1], [0, 1])]

def test_decode_ashr_b():
  inst = decode_ir3(bytes.fromhex('00001f2003081847'))[0]
  assert inst.name == 'ashr.b'
  assert inst.dst == ('r', 0, 3)
  assert inst.srcs == (('r', 0, 0), 31)

def test_execute_ashr_b():
  regs = {('r', 0, 0): [0, 1, 0x80000000, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('00001f2003081847'), regs)
  assert regs[('r', 0, 3)] == [0, 0, 0xffffffff, 0xffffffff]

def test_decode_shl_b():
  inst = decode_ir3(bytes.fromhex('030002200308d846'))[0]
  assert inst.name == 'shl.b'
  assert inst.dst == ('r', 0, 3)
  assert inst.srcs == (('r', 0, 3), 2)

def test_execute_shl_b():
  regs = {('r', 0, 3): [0, 1, 0x40000000, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('030002200308d846'), regs)
  assert regs[('r', 0, 3)] == [0, 4, 0, 0xfffffffc]

def test_execute_shift_count_is_masked():
  regs = {('r', 0, 0): [1, 0x80000000, 0xffffffff], ('r', 0, 3): [1, 0x80000000, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('0000202003081847'), regs)
  emu.execute_ir3(bytes.fromhex('030020200308d846'), regs)
  assert regs[('r', 0, 3)] == [1, 0x80000000, 0xffffffff]

def test_decode_shrg():
  inst = decode_ir3(bytes.fromhex('1e30030003040065'))[0]
  assert inst.name == 'shrg'
  assert inst.dst == ('r', 0, 3)
  assert inst.srcs == (30, ('r', 0, 0), ('r', 0, 3))

def test_execute_shrg():
  regs = {('r', 0, 0): [0, 0x40000000, 0x80000000, 0xffffffff],
          ('r', 0, 3): [1, 2, 4, 8]}
  emu.execute_ir3(bytes.fromhex('1e30030003040065'), regs)
  assert regs[('r', 0, 3)] == [1, 3, 6, 11]

def test_decode_mull_u():
  inst = decode_ir3(bytes.fromhex('c0000c2005005046'))[0]
  assert inst.name == 'mull.u'
  assert inst.dst == ('r', 1, 1)
  assert inst.srcs == (('r', 48, 0), 12)

def test_execute_mull_u():
  regs = {('r', 48, 0): [0, 1, 0x40000000, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('c0000c2005005046'), regs)
  assert regs[('r', 1, 1)] == [0, 12, 0, 0xfffffff4]

def test_decode_madsh_m16():
  inst = decode_ir3(bytes.fromhex('1010050004008261'))[0]
  assert inst.name == 'madsh.m16'
  assert inst.dst == ('r', 1, 0)
  assert inst.srcs == (('c', 4, 0), ('r', 1, 0), ('r', 1, 1))

def test_execute_madsh_m16():
  regs = {('c', 4, 0): [0, 1, 0xffff, 0x12345678],
          ('r', 1, 0): [0xffffffff, 0x00010000, 0xffff0000, 0x9abc0000],
          ('r', 1, 1): [7, 11, 13, 17]}
  emu.execute_ir3(bytes.fromhex('1010050004008261'), regs)
  assert regs[('r', 1, 0)] == [7, 0x1000b, 0x1000d, 0xb0200011]

def test_decode_add_u():
  inst = decode_ir3(bytes.fromhex('0310030008001042'))[0]
  assert inst.name == 'add.u'
  assert inst.dst == ('r', 2, 0)
  assert inst.srcs == (('c', 0, 3), ('r', 0, 3))

def test_execute_add_u():
  regs = {('c', 0, 3): [0, 1, 0xffffffff, 0x80000000],
          ('r', 0, 3): [0, 2, 1, 0x80000000]}
  emu.execute_ir3(bytes.fromhex('0310030008001042'), regs)
  assert regs[('r', 2, 0)] == [0, 3, 0, 0]

def test_decode_immediate_add_u():
  inst = decode_ir3(bytes.fromhex('0400012006001042'))[0]
  assert inst.name == 'add.u'
  assert inst.dst == ('r', 1, 2)
  assert inst.srcs == (('r', 1, 0), 1)

def test_execute_immediate_add_u():
  regs = {('r', 1, 0): [0, 1, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('0400012006001042'), regs)
  assert regs[('r', 1, 2)] == [1, 2, 0]

def test_decode_repeated_add_u():
  inst = decode_ir3(bytes.fromhex('03100b0010011842'))[0]
  assert inst.name == 'add.u'
  assert inst.dst == ('r', 4, 0)
  assert inst.srcs == (('c', 0, 3), ('r', 2, 3))
  assert (inst.repeat, inst.repeat_srcs) == (1, (False, True))

def test_execute_repeated_add_u():
  regs = {('c', 0, 3): [10, 20], ('r', 2, 3): [1, 2], ('r', 3, 0): [3, 4]}
  emu.execute_ir3(bytes.fromhex('03100b0010011842'), regs)
  assert regs[('r', 4, 0)] == [11, 22]
  assert regs[('r', 4, 1)] == [13, 24]

def test_decode_cmps_u_lt():
  inst = decode_ir3(bytes.fromhex('0400021000409042'))[0]
  assert inst.name == 'cmps.u.lt'
  assert inst.dst == ('hr', 0, 0)
  assert inst.srcs == (('r', 1, 0), ('c', 0, 2))

def test_execute_cmps_u_lt():
  regs = {('r', 1, 0): [0, 1, 2, 0xffffffff], ('c', 0, 2): [1, 1, 0xffffffff, 0]}
  emu.execute_ir3(bytes.fromhex('0400021000409042'), regs)
  assert regs[('hr', 0, 0)] == [1, 0, 1, 0]

def test_decode_cov_u16s32():
  inst = decode_ir3(bytes.fromhex('000000000a400920'))[0]
  assert inst.name == 'cov.u16s32'
  assert inst.dst == ('r', 2, 2)
  assert inst.srcs == (('hr', 0, 0),)

def test_execute_cov_u16s32():
  regs = {('hr', 0, 0): [0, 1, 0xffff, 0x10000]}
  emu.execute_ir3(bytes.fromhex('000000000a400920'), regs)
  assert regs[('r', 2, 2)] == [0, 1, 0xffff, 0]

def test_decode_mov_u32u32():
  inst = decode_ir3(bytes.fromhex('c000000004c00c20'))[0]
  assert inst.name == 'mov.u32u32'
  assert inst.dst == ('r', 1, 0)
  assert inst.srcs == (('r', 48, 0),)

def test_execute_mov_u32u32():
  regs = {('r', 48, 0): [0, 1, 0xffffffff]}
  emu.execute_ir3(bytes.fromhex('c000000004c00c20'), regs)
  assert regs[('r', 1, 0)] == [0, 1, 0xffffffff]

def test_decode_ldg_u32():
  inst = decode_ir3(bytes.fromhex('010081010b0006c0'))[0]
  assert inst.name == 'ldg.u32'
  assert inst.dst == ('r', 2, 3)
  assert inst.srcs == (('r', 1, 0), 0, 1)

def test_execute_ldg_u32():
  values = (ctypes.c_uint32 * 3)(5, 7, 9)
  addresses = [ctypes.addressof(values) + index * 4 for index in range(3)]
  checked = []
  regs = {('r', 1, 0): [address & 0xffffffff for address in addresses],
          ('r', 1, 1): [address >> 32 for address in addresses]}
  emu.execute_ir3(bytes.fromhex('010081010b0006c0'), regs, check_range=lambda address, size: checked.append((address, size)))
  assert regs[('r', 2, 3)] == [5, 7, 9]
  assert checked == [(address, 4) for address in addresses]

def test_decode_stg_u32():
  inst = decode_ir3(bytes.fromhex('16008001000dc6c0'))[0]
  assert inst.name == 'stg.u32'
  assert inst.srcs == (('r', 1, 2), ('r', 2, 3), 0, 1)

def test_execute_stg_u32():
  values = (ctypes.c_uint32 * 3)()
  addresses = [ctypes.addressof(values) + index * 4 for index in range(3)]
  checked = []
  regs = {('r', 1, 2): [address & 0xffffffff for address in addresses],
          ('r', 1, 3): [address >> 32 for address in addresses],
          ('r', 2, 3): [5, 7, 9]}
  emu.execute_ir3(bytes.fromhex('16008001000dc6c0'), regs, check_range=lambda address, size: checked.append((address, size)))
  assert list(values) == [5, 7, 9]
  assert checked == [(address, 4) for address in addresses]

def test_execute_vector_memory():
  values = (ctypes.c_uint32 * 8)()
  addresses = [ctypes.addressof(values), ctypes.addressof(values) + 16]
  regs = {('r', 2, 1): [address & 0xffffffff for address in addresses],
          ('r', 2, 2): [address >> 32 for address in addresses],
          **{('r', 1, component): [component + 1, component + 5] for component in range(4)}}
  emu.execute_ir3(bytes.fromhex('080080040013c6c0'), regs)
  assert list(values) == list(range(1, 9))
  regs = {('r', 1, 1): [address & 0xffffffff for address in addresses],
          ('r', 1, 2): [address >> 32 for address in addresses]}
  emu.execute_ir3(bytes.fromhex('014081040f0006c0'), regs)
  assert [regs[('r', 3, 3)], regs[('r', 4, 0)], regs[('r', 4, 1)], regs[('r', 4, 2)]] == \
    [[1, 5], [2, 6], [3, 7], [4, 8]]

def test_decode_end():
  inst = decode_ir3(bytes.fromhex('0000000000000003'))[0]
  assert inst.name == 'end'

def test_execute_end():
  regs = {('r', 0, 0): [1, 2, 3]}
  emu.execute_ir3(bytes.fromhex('0000000000000003'), regs)
  assert regs == {('r', 0, 0): [1, 2, 3]}
