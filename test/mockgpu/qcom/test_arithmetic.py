import struct
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Operand,Instruction,Workgroup,decode
from test.mockgpu.qcom.test_emu import Memory


def evaluate(op,inputs,*,source_half=False,dest_half=False,fields=None,modifiers=None):
  lanes=np.arange(len(inputs[0]))
  state=Workgroup(bytes(64),(len(lanes),1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  operands={'DST':Operand('register',20,dest_half)}
  for i,values in enumerate(inputs):
    state.registers[source_half][i] = np.asarray(values,np.uint16 if source_half else np.uint32)
    operands['SRC'+str(i+1)]=Operand('register',i,source_half,modifier=(modifiers or [0]*len(inputs))[i])
  ins=Instruction(op,fields or {},operands,4<<61 if op in ('rcp','sqrt','exp2','log2','sin','cos') else 3<<61)
  state.alu(ins,lanes,0)
  return state.registers[dest_half][20].copy()


def test_multiply_split_halves_matches_full_width_product():
  # Adreno compiler expands imul as mull plus two high-half madsh terms.
  a=np.array([0x10001,0xffff0001,0xffffffff,0x80008000,123456789],np.uint32)
  b=np.array([0x20003,0x0002ffff,0xffffffff,0x8000ffff,987654321],np.uint32)
  low=evaluate('mull.u',[a,b])
  first=evaluate('madsh.m16',[a,b,low])
  result=evaluate('madsh.m16',[b,a,first])
  expected=np.array([(int(x)*int(y))&0xffffffff for x,y in zip(a,b)],np.uint32)
  np.testing.assert_array_equal(result,expected)

@pytest.mark.parametrize('condition,expected',[(0,[1,0,0]),(1,[1,1,0]),(2,[0,0,1]),(3,[0,1,1]),(4,[0,1,0]),(5,[1,0,1])])
def test_signed_comparisons_write_boolean_not_float(condition,expected):
  result=evaluate('cmps.s',[[0xffffffff,0,1],[0,0,0]],fields={'COND':condition},dest_half=True)
  np.testing.assert_array_equal(result,np.array(expected,np.uint16))


def test_half_float_arithmetic_and_absneg():
  a=np.array([-1.5,2.5,-3.25],np.float16).view(np.uint16)
  b=np.array([.5,-.5,4.25],np.float16).view(np.uint16)
  result=evaluate('add.f',[a,b],source_half=True,dest_half=True,modifiers=[2,1]).view(np.float16)
  np.testing.assert_array_equal(result,np.array([1.,3.,-1.],np.float16))


def test_swap_reads_both_sources_before_writing():
  state=Workgroup(bytes(64),(2,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  state.registers[0][0]=[10,20]
  state.registers[0][1]=[30,40]
  swap=Instruction('swz',{},dict(DST0=Operand('register',0),DST1=Operand('register',1),SRC0=Operand('register',1),SRC1=Operand('register',0)),1<<61)
  state.run((swap,Instruction('end',{}, {},0)))
  np.testing.assert_array_equal(state.registers[0][:2],[[30,40],[10,20]])


def test_repeated_destination_and_source_advance_independently():
  state=Workgroup(bytes(64),(2,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  state.registers[0][0:3]=[[10,20],[30,40],[50,60]]
  ins=Instruction('add.u',{'REPEAT':2},dict(DST=Operand('register',10),SRC1=Operand('register',0,repeat=True),SRC2=Operand('immediate',7)),2<<61)
  state.run((ins,Instruction('end',{}, {},0)))
  np.testing.assert_array_equal(state.registers[0][10:13],[[17,27],[37,47],[57,67]])

def test_conversion_rounds_toward_zero_instead_of_nearest():
  # Mesa round_t ROUND_ZERO=0; 2051 is midway between half 2050 and 2052.
  group=Workgroup(bytes(64),(2,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[1][0]=[2051,65535]
  convert=Instruction('mov',{'SRC_TYPE':2,'DST_TYPE':0},dict(DST=Operand('register',1,True),SRC=Operand('register',0,True)),1<<61)
  group.run((convert,Instruction('end',{}, {},0)))
  np.testing.assert_array_equal(group.registers[1][1],[0x6801,0x7bff])

def test_mad_rounds_the_product_before_adding():
  # Mesa commit 92d2671 explicitly identifies mad.f16/mad.f32 as unfused.
  a=np.array([1+2**-23],np.float32).view(np.uint32)
  b=np.array([1-2**-23],np.float32).view(np.uint32)
  c=np.array([-1],np.float32).view(np.uint32)
  result=evaluate('mad.f32',[a,b,c]).view(np.float32)
  np.testing.assert_array_equal(result,[0.])

def test_encoded_bitwise_source_negation_complements_bits():
  # ir3.h:ir3_cat2_absneg maps AND's NEG source flag to IR3_REG_BNOT.
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  a=np.array([0,1,0xffffffff,0x80000000],np.uint32)
  b=np.array([0xffffffff,7,0xaaaaaaaa,0xffffffff],np.uint32)
  group.registers[0][0],group.registers[0][1]=a,b
  word = (2<<61)|(28<<53)|(1<<52)|(2<<32)|(1<<16)|(1<<14)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][2],~a&b)

@pytest.mark.parametrize('opcode,a,b,expected',[
  (16,65535,1,0),                    # ADD_U wraps at 16 bits, then zero-extends.
  (17,32767,1,0xffff8000),           # ADD_S wraps at 16 bits, then sign-extends.
  (56,65535,0,65535),                # ASHR_B shifts signed, then zero-extends.
  (0,0x3c00,0x1000,0x3f800000),     # ADD_F rounds its half result before widening.
])
def test_encoded_half_alu_result_converts_after_source_width_arithmetic(opcode,a,b,expected):
  group=Workgroup(bytes(64),(1,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[1][0],group.registers[1][1]=a,b
  word=(2<<61)|(opcode<<53)|(1<<46)|(2<<32)|(1<<16)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][2],[expected])

def test_encoded_half_float_table_immediate_rounds_before_arithmetic():
  group=Workgroup(bytes(64),(1,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  # Add h(pi) to half zero, with a full-width destination.
  word=(2<<61)|(1<<46)|(2<<32)|(5<<11)|(1<<10)|5
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][2].view(np.float32),[3.140625])

def test_encoded_float_alu_output_narrowing_uses_round_toward_zero():
  # ir3_cf.c folds only ROUND_ZERO conversions into a supported ALU output.
  group=Workgroup(bytes(64),(1,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=np.array([1],np.float32).view(np.uint32)
  group.registers[0][1]=np.array([2**-11+2**-13],np.float32).view(np.uint32)
  word=(2<<61)|(1<<52)|(1<<46)|(2<<32)|(1<<16)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[1][2],[0x3c00])

@pytest.mark.parametrize('immediate',[-1,-1024])
def test_encoded_signed_integer_immediate(immediate):
  group=Workgroup(bytes(64),(1,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  source=(4<<11)|(immediate&2047)
  word=(2<<61)|(16<<53)|(1<<52)|(2<<32)|(source<<16)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][2],[immediate&0xffffffff])

def test_encoded_andg_combines_masks():
  # ir3-cat3.xml: ANDG computes (SRC2 & SRC1) | SRC3.
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  a=np.array([0,0xffffffff,0xff00ff00,0x12345678],np.uint32)
  b=np.array([0xffffffff,0xaaaaaaaa,0x0f0f0f0f,0xfedcba98],np.uint32)
  c=np.array([3,0x55555555,0x80000001,0],np.uint32)
  group.registers[0][0],group.registers[0][1],group.registers[0][2]=a,b,c
  word=(3<<61)|(12<<55)|(1<<47)|(1<<42)|(3<<32)|(2<<16)|(1<<13)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[0][3],(a&b)|c)

@pytest.mark.parametrize('opcode,left,merge',[(8,False,False),(9,True,False),(11,True,True)])
def test_encoded_shift_combines_mask_or_merge(opcode,left,merge):
  # The cat3 alternate SHRM/SHLM/SHLG forms combine a shift with AND/OR.
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  shifts=np.array([0,1,4,31],np.uint32)
  values=np.array([0x12345678,0x87654321,0xffffffff,1],np.uint32)
  masks=np.array([0x0f0f0f0f,0x80000000,0xff00ff00,0xffffffff],np.uint32)
  group.registers[0][0],group.registers[0][1],group.registers[0][2]=shifts,values,masks
  word=(3<<61)|(opcode<<55)|(1<<47)|(1<<42)|(3<<32)|(2<<16)|(1<<13)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  shifted=values<<shifts if left else values>>shifts
  np.testing.assert_array_equal(group.registers[0][3],shifted|masks if merge else shifted&masks)

@pytest.mark.parametrize('opcode,expected',[(4,[0,1,0,-1]),(5,[1,0,-1,0])])
def test_encoded_trigonometric_source_is_radians(opcode,expected):
  # ir3_nir_trig.py reduces to [-pi,pi]; ISA SIN/COS consume radians.
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=np.array([0,np.pi/2,np.pi,-np.pi/2],np.float32).view(np.uint32)
  word=(4<<61)|(opcode<<53)|(1<<52)|(1<<32)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_allclose(group.registers[0][1].view(np.float32),expected,atol=1e-6)

@pytest.mark.parametrize('rounding,expected',[(0,[0x3c00,0xbc00,0x6801]),(1,[0x3c01,0xbc01,0x6802])])
def test_encoded_half_conversion_rounding_modes(rounding,expected):
  group=Workgroup(bytes(64),(3,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=np.array([1.0006,-1.0006,2051],np.float32).view(np.uint32)
  # cat1 source typeF32=1, destinationF16=0, gpr source, explicitROUND[56:55].
  word=(1<<61)|(rounding<<55)|(1<<50)|(1<<32)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[1][1],expected)

def test_encoded_sign_and_count_leading_zero():
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=np.array([-3.,-0.,0.,2.],np.float32).view(np.uint32)
  sign=(2<<61)|(4<<53)|(1<<52)|(1<<32)
  group.run(decode(struct.pack('<2Q',sign,6<<55)))
  np.testing.assert_array_equal(group.registers[0][1],np.array([-1.,-0.,0.,1.],np.float32).view(np.uint32))
  group=Workgroup(bytes(64),(4,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=[0,1,8,0xffffffff]
  clz=(2<<61)|(53<<53)|(1<<52)|(1<<32)
  group.run(decode(struct.pack('<2Q',clz,6<<55)))
  np.testing.assert_array_equal(group.registers[0][1],[0xffffffff,31,28,0])

@pytest.mark.parametrize('half',[False,True])
def test_encoded_count_leading_zero_uses_minus_one_for_zero(half):
  # ir3_compiler_nir.c ufind_msb keeps CLZ's zero result unchanged; NIR requires -1.
  group=Workgroup(bytes(64),(1,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  word=(2<<61)|(53<<53)|(int(not half)<<52)|(1<<32)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  np.testing.assert_array_equal(group.registers[half][1],[0xffff if half else 0xffffffff])

def test_encoded_signed_24bit_multiply_sign_extends_and_wraps():
  group=Workgroup(bytes(64),(6,1,1),(0,0,0),0xfc,0xfc,0xfc,Memory({}),0,0)
  group.registers[0][0]=[0x7fffff,0x800000,0xffffff,0x01000001,0xffffffff,0x7fffff]
  group.registers[0][1]=[2,2,2,0xffffff,0x800000,0x7fffff]
  word=(2<<61)|(49<<53)|(1<<52)|(2<<32)|(1<<16)
  group.run(decode(struct.pack('<2Q',word,6<<55)))
  expected=[0xfffffe,0xff000000,0xfffffffe,0xffffffff,0x800000,0xff000001]
  np.testing.assert_array_equal(group.registers[0][2],expected)
