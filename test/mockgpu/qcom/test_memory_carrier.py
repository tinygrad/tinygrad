import gc, weakref
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Instruction, Operand, Workgroup
from test.mockgpu.qcom.test_emu import Memory


def group(memory=None, count=3, private_bytes=8, shared_bytes=16):
  return Workgroup(bytes(64),(count,1,1),(0,0,0),0xfc,0xfc,0xfc,memory or Memory({}),shared_bytes,private_bytes)


@pytest.mark.parametrize('width',[1,2,4])
def test_global_unaligned_gather_and_scatter_across_allocations(width):
  memory=Memory({0x1000:bytes(range(16)),0x2000:bytes(range(16,32))})
  machine=group(memory)
  addresses=np.array([0x1001,0x2003,0x1007],np.uint64)
  expected=[int.from_bytes(memory.read(int(address),width),'little') for address in addresses]
  np.testing.assert_array_equal(machine.transfer(addresses,width),expected)
  values=np.array([0xfedcba98,0x01234567,0],np.uint32)
  machine.transfer(addresses,width,values)
  for address,value in zip(addresses,values):
    assert memory.read(int(address),width) == int(value).to_bytes(4,'little')[:width]


@pytest.mark.parametrize('space', ['l','p'])
def test_unaligned_half_storage_keeps_address_space_identity(space):
  machine=group()
  machine.registers[0][0]=[1,3,5]
  machine.registers[1][1]=[0xaabb,0xccdd,0xeeff]
  store=Instruction('st'+space,{'TYPE':2,'SIZE':1},dict(DST=Operand('register',0),SRC=Operand('register',1)),6<<61)
  load=Instruction('ld'+space,{'TYPE':2,'SIZE':1},dict(DST=Operand('register',2),SRC=Operand('register',0)),6<<61)
  machine.run((store,load,Instruction('end',{}, {},0)))
  np.testing.assert_array_equal(machine.registers[1][2],[0xaabb,0xccdd,0xeeff])


def test_private_span_cannot_spill_into_the_following_lane():
  machine=group(count=2,private_bytes=8)
  machine.private[:] = np.arange(16,dtype=np.uint8).reshape(2,8)
  machine.registers[0][0]=[7,0]
  machine.registers[1][1]=[0xaaaa,0xbbbb]
  before=machine.private.copy()
  store=Instruction('stp',{'TYPE':2,'SIZE':1},dict(DST=Operand('register',0),SRC=Operand('register',1)),6<<61)
  with pytest.raises(RuntimeError,match='out of bounds'):
    machine.run((store,Instruction('end',{}, {},0)))
  np.testing.assert_array_equal(machine.private,before)


def test_global_view_retains_its_buffer_owner_through_permission_check():
  class OwnedBytes(bytearray): pass
  class EphemeralMemory:
    def region(self,address,size):
      data=OwnedBytes(bytes(8))
      self.owner=weakref.ref(data)
      return 0x1000,data
    def validate(self,address,size,write=False):
      gc.collect()
      assert self.owner() is not None
      assert (address,size,write)==(0x1001,4,True)
  memory=EphemeralMemory()
  group(memory,count=1).transfer(np.array([0x1001],np.uint64),4,np.array([0x12345678],np.uint32))


def test_overlapping_global_stores_preserve_lane_order():
  memory=Memory({0x1000:bytes(8)})
  group(memory,count=2).transfer(np.array([0x1000,0x1001],np.uint64),4,np.array([0x11223344,0xaabbccdd],np.uint32))
  assert memory.read(0x1000,5) == bytes.fromhex('44ddccbbaa')

def test_global_store_does_not_admit_floating_values_as_integer_bits():
  memory=Memory({0x1000:bytes(8)})
  with pytest.raises(TypeError):
    group(memory,count=1).transfer(np.array([0x1000],np.uint64),4,np.array([1.5],np.float32))
  assert memory.read(0x1000,8) == bytes(8)
