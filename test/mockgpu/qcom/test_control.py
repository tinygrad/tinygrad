import struct
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Operand,Instruction,Workgroup,decode
from test.mockgpu.qcom.test_emu import Memory

def R(index): return Operand('register',index)
def I(value): return Operand('immediate',value)
END=Instruction('end',{}, {},0)


def state(lanes=4,shared_bytes=0,private_bytes=0,memory=None):
  return Workgroup(bytes(64),(lanes,1,1),(0,0,0),0xfc,0xfc,0xfc,memory or Memory({}),shared_bytes,private_bytes)


def move(dst,value):
  return Instruction('mov',{'SRC_TYPE':3,'DST_TYPE':3},{'DST':R(dst),'SRC':I(value)},1<<61)


def add(dst,source,value):
  return Instruction('add.u',{},dict(DST=R(dst),SRC1=R(source),SRC2=I(value)),2<<61)


def branch(offset):
  return Instruction('br',{'INV1':0,'COMP1':0,'IMMED':offset}, {},0)


def jump(offset):
  return Instruction('jump',{'IMMED':offset&0xffffffff},{},0)


def test_divergent_lanes_reconverge_with_their_own_values():
  group=state()
  group.registers[0][0]=[0,1,2,3]
  compare=Instruction('cmps.u',{'COND':0},dict(DST=R(248),SRC1=R(0),SRC2=I(2)),2<<61)
  group.run((compare,branch(3),move(1,20),jump(2),move(1,10),add(1,1,1),END))
  np.testing.assert_array_equal(group.registers[0][1],[11,11,21,21])


def test_backward_branch_has_independent_lane_trip_counts():
  group=state()
  group.registers[0][0]=[0,1,2,3]
  compare=Instruction('cmps.u',{'COND':0},dict(DST=R(248),SRC1=R(1),SRC2=R(0)),2<<61)
  group.run((compare,branch(2),jump(3),add(1,1,1),jump(-4),END))
  np.testing.assert_array_equal(group.registers[0][1],[0,1,2,3])


def test_predicated_else_and_end_restore_lane_execution():
  group=state()
  group.registers[0][248]=[0,1,0,1]
  group.run((Instruction('predt',{}, {},0),move(0,10),Instruction('predf',{}, {},0),move(0,20),
             Instruction('prede',{}, {},0),add(0,0,1),END))
  np.testing.assert_array_equal(group.registers[0][0],[21,11,21,11])


@pytest.mark.parametrize('repeat', [0, 3])
def test_end_of_quad_nop_preserves_compute_lanes_and_predication(repeat):
  # ir3_legalize.c:helper_sched uses EQ to end fragment helper invocations.
  # A compute workgroup has no helper lanes; EQ must not end ordinary lanes or
  # change the predicate mask while the compiler's image program continues.
  nop = decode(struct.pack('<Q', (1 << 48) | (repeat << 40)))[0]
  assert nop.op == 'nop' and nop.fields['EQ'] == 1
  group = state()
  group.registers[0][0] = [1, 2, 3, 4]
  group.registers[0][248] = [0, 1, 0, 1]
  group.run((Instruction('predt', {}, {}, 0), add(0, 0, 10), nop,
             Instruction('predf', {}, {}, 0), add(0, 0, 20), Instruction('prede', {}, {}, 0), add(0, 0, 1), END))
  np.testing.assert_array_equal(group.registers[0][0], [22, 13, 24, 15])


@pytest.mark.parametrize('opcode', [2, 4, 6])
def test_end_of_quad_on_other_control_instructions_remains_unsupported(opcode):
  with pytest.raises(RuntimeError, match='unsupported execution modifier'):
    decode(struct.pack('<Q', (opcode << 55) | (1 << 48)))


def test_shared_barrier_exposes_all_participating_lane_stores():
  group=state(shared_bytes=16)
  group.registers[0][0]=[0,4,8,12]
  group.registers[0][1]=[1,2,3,4]
  group.registers[0][2]=[4,8,12,0]
  store=Instruction('stl',{'TYPE':3,'SIZE':1},dict(DST=R(0),SRC=R(1)),6<<61)
  load=Instruction('ldl',{'TYPE':3,'SIZE':1},dict(DST=R(3),SRC=R(2)),6<<61)
  group.run((store,Instruction('bar',{}, {},7<<61),load,END))
  np.testing.assert_array_equal(group.registers[0][3],[2,3,4,1])


def test_nonparticipating_lane_rejects_divergent_barrier():
  group=state()
  group.registers[0][248]=[1,0,0,0]
  with pytest.raises(RuntimeError,match='divergent.*barrier'):
    group.run((branch(2),Instruction('bar',{}, {},7<<61),END))


def test_signed_memory_offset_and_private_lane_isolation():
  memory=Memory({0x1000:struct.pack('<4I',11,22,33,44)})
  group=state(memory=memory,private_bytes=8)
  group.registers[0][0]=0x1008
  load=Instruction('ldg',{'TYPE':3,'SIZE':2,'OFF':(1<<64)-8},dict(DST=R(2),SRC1=R(0)),6<<61)
  group.run((load,END))
  np.testing.assert_array_equal(group.registers[0][2:4],[[11]*4,[22]*4])
  group=state(private_bytes=8)
  group.registers[0][1]=[11,22,33,44]
  store=Instruction('stp',{'TYPE':3,'SIZE':1},dict(DST=R(0),SRC=R(1)),6<<61)
  load=Instruction('ldp',{'TYPE':3,'SIZE':1},dict(DST=R(2),SRC=R(0)),6<<61)
  group.run((store,load,END))
  np.testing.assert_array_equal(group.registers[0][2],[11,22,33,44])


def test_memory_vector_must_fit_every_lane():
  group=state(private_bytes=4)
  load=Instruction('ldp',{'TYPE':3,'SIZE':2},dict(DST=R(2),SRC=R(0)),6<<61)
  with pytest.raises(RuntimeError,match='out of bounds'): group.run((load,END))

@pytest.mark.parametrize('initial_predicate,predication_opcode',[(1,13),(0,14)])
def test_jump_point_refreshes_predication_from_machine_code(initial_predicate,predication_opcode):
  # ir3-cat0.xml predt docs: JP refreshes the predicate mask and retains its mode.
  def mov_word(dst,value,jp=False): return 0x204cc00000000000 | dst<<32 | value | int(jp)<<59
  words = (mov_word(248,initial_predicate), (predication_opcode<<55)|(1<<49), mov_word(0,11),
           mov_word(248,1-initial_predicate), mov_word(0,22,jp=True), (15<<55)|(1<<49), 6<<55)
  group=state(lanes=1)
  group.run(decode(struct.pack('<7Q',*words)))
  np.testing.assert_array_equal(group.registers[0][0],[11])

@pytest.mark.parametrize('opcode',[4,5])
def test_encoded_half_store_uses_full_address_and_half_data_registers(opcode):
  group=state(lanes=1,shared_bytes=8,private_bytes=8)
  group.registers[0][0]=2
  group.registers[1][0]=6
  group.registers[1][1]=0xabcd
  word=(6<<61)|(opcode<<54)|(2<<49)|(1<<40)|(1<<24)|(1<<23)|(1<<1)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  storage=group.shared if opcode==4 else group.private[0]
  np.testing.assert_array_equal(storage,[0,0,0xcd,0xab,0,0,0,0])

def test_encoded_negative_global_offset():
  group=state(lanes=1,memory=Memory({0x1000:struct.pack('<2I',11,22)}))
  group.registers[0][0]=0x1004
  word=(6<<61)|(3<<49)|(2<<32)|(1<<24)|(1<<23)|((-4&8191)<<1)|1
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][2],[11])
