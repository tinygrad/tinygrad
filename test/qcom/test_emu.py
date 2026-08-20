import struct
from test.mockgpu.qcom.emu import decode_ir3

def test_decode_add_f():
  inst = decode_ir3(bytes.fromhex('0b0010000b081850'), gpu_id=630)[0]
  assert inst.name == 'add.f'
  assert inst.dst == ('r', 2, 3)
  assert inst.srcs == (('r', 2, 3), ('r', 4, 0))
  assert (inst.sy, inst.nop) == (True, 3)
  
def _bits(values):
  return [struct.unpack('<I', struct.pack('<f', x))[0] for x in values]

def test_execute_add_f():
  import test.mockgpu.qcom.emu as emu
  regs = {('r', 2, 3): _bits([1, 2, 3]), ('r', 4, 0): _bits([4, 5, 6])}
  emu.execute_ir3(bytes.fromhex('0b0010000b081850'), regs)
  assert regs[('r', 2, 3)] == _bits([5, 7, 9])
  
def test_decode_nop():
  inst = decode_ir3(bytes(8), gpu_id=630)[0]
  assert inst.name == 'nop'
  
def test_execute_nop():
  import test.mockgpu.qcom.emu as emu
  regs = {}
  emu.execute_ir3(bytes(8), regs)
  assert regs == {}
