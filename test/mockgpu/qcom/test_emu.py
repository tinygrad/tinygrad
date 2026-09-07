import struct
import pytest
from test.mockgpu.qcom.emu import execute,decode

# Mesa 25.2.7 ir3-cat1/cat2/cat6 encodings: load pointers, add, store, end.
# Deliberately varied memory values verify arithmetic rather than image recognition.
ADD = (0x202cc00000000002, 0x202cc00100000003, 0x202cc00300000004, 0x202cc00400000005,
       0x202cc00500000000, 0x202cc00600000001, 0xc006000201800001, 0xc00600070180c001,
       0x5218080200070002, 0xc0c60b0001800004, 0x0300000000000000)

class Memory:
  def __init__(self, regions): self.regions = {a:bytearray(v) for a,v in regions.items()}
  def region(self, address, size):
    for base, data in self.regions.items():
      if base <= address and address + size <= base + len(data): return base, data
    raise RuntimeError(f"unmapped memory: {address:#x}+{size}")
  def read(self, address, size):
    base,data = self.region(address,size)
    return bytes(data[address-base:address-base+size])
  def write(self, address, value):
    base,data = self.region(address,len(value))
    data[address-base:address-base+len(value)] = value
  def validate(self,address,size,write=False): self.region(address,size)

@pytest.mark.parametrize("a,b", [(7,19),(0xffffffff,2),(0x80000000,0x80000000)])
def test_real_machine_code_integer_add(a,b):
  memory = Memory({0x1000:bytes(4),0x2000:struct.pack('<I',a),0x3000:struct.pack('<I',b)})
  execute(struct.pack('<11Q',*ADD),struct.pack('<3Q',0x1000,0x2000,0x3000),(1,1,1),(1,1,1),0xfc,0xfc,memory)
  assert memory.read(0x1000,4) == struct.pack('<I',(a+b)&0xffffffff)

def test_invalid_encoding_does_not_write():
  memory = Memory({0x1000:b'keep'})
  with pytest.raises(RuntimeError,match='IR3|ir3'):
    execute(bytes.fromhex('ffffffffffffffff'),b'',(1,1,1),(1,1,1),0xfc,0xfc,memory)
  assert memory.read(0x1000,4) == b'keep'

def test_valid_but_unsupported_opcode_is_rejected_before_the_first_store():
  # cat0 emit is a valid graphics opcode, unsupported by this compute emulator.
  image = struct.pack('<12Q',*ADD[:-1],0x0380000000000000,ADD[-1])
  memory = Memory({0x1000:b'keep',0x2000:struct.pack('<I',3),0x3000:struct.pack('<I',9)})
  with pytest.raises(RuntimeError,match='unsupported'):
    execute(image,struct.pack('<3Q',0x1000,0x2000,0x3000),(1,1,1),(1,1,1),0xfc,0xfc,memory)
  assert memory.read(0x1000,4) == b'keep'

def test_nondefault_rounding_is_rejected_at_decode():
  # ir3-cat1.xml ROUND[56:55]. The default conversion contract uses mode 0.
  with pytest.raises(RuntimeError,match='rounding'):
    decode(struct.pack('<Q',ADD[0] | (1<<55)))

def test_store_checks_permission_before_mutating_a_mapped_region():
  class ReadOnlyMemory(Memory):
    def validate(self,address,size,write=False):
      if write: raise RuntimeError('read-only allocation')
      return super().validate(address,size,write)
  memory = ReadOnlyMemory({0x1000:b'keep',0x2000:struct.pack('<I',3),0x3000:struct.pack('<I',9)})
  with pytest.raises(RuntimeError,match='read-only'):
    execute(struct.pack('<11Q',*ADD),struct.pack('<3Q',0x1000,0x2000,0x3000),(1,1,1),(1,1,1),0xfc,0xfc,memory)
  assert memory.read(0x1000,4) == b'keep'

@pytest.mark.parametrize('condition',[6,7])
def test_unknown_comparison_condition_is_rejected_from_machine_code(condition):
  # cat2 compare COND is three bits, but Mesa defines only LT..NE (0..5).
  word = (2<<61)|(20<<53)|(1<<52)|(condition<<48)|(248<<32)|1<<16
  with pytest.raises(RuntimeError,match='condition'):
    decode(struct.pack('<2Q',word,6<<55))

def test_repeated_destination_must_fit_the_register_file_at_decode():
  # A repeat on destination r63.w would run past the final scalar register.
  word = (2<<61)|(16<<53)|(1<<52)|(1<<40)|(255<<32)|1<<16
  with pytest.raises(RuntimeError,match='destination'):
    decode(struct.pack('<2Q',word,6<<55))

@pytest.mark.parametrize('rounding',[2,3])
def test_directed_float_rounding_remains_unsupported(rounding):
  word=(1<<61)|(rounding<<55)|(1<<50)|(1<<32)
  with pytest.raises(RuntimeError,match='rounding'):
    decode(struct.pack('<2Q',word,6<<55))
