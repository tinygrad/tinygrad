"""QCOMCL compiler-generated programs exercise the real HCQ2 mock launch path."""
import math
import struct
import numpy as np
import pytest
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import DEV
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


requires_qcomcl = pytest.mark.skipif(
  DEV.device != 'QCOM' or DEV.renderer != 'CL' or not DEV.interface.startswith('MOCK'),
  reason='requires the MOCK+QCOM:CL compiler profile',
)


def source_kernel(source, name, output, groups, local, *, inputs=()):
  """Compile ordinary OpenCL and retain the normal PROGRAM/kernargs/HCQ2 route."""
  renderer = Device[output.device].renderer
  binary = renderer.compiler.compile_cached(source)

  def program(out, *args):
    sink = UOp.sink(out, *args, arg=KernelInfo(name=name))
    info = ProgramInfo(name=name, global_size=groups, local_size=local, globals=tuple(range(len(args) + 1)),
                       outs=(0,), ins=tuple(range(1, len(args) + 1)), target=renderer.target)
    return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                                         UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=binary)))

  return Tensor.custom_kernel(output, *inputs, fxn=program)[0]


@requires_qcomcl
def test_generated_add_uses_every_lane_and_workgroup():
  # Reject a missing CL dispatch, missing group offset, or incorrect address carry.
  left = (np.arange(407, dtype=np.float32).reshape(37, 11) - 93) / 8
  right = (np.arange(407, dtype=np.float32).reshape(37, 11) % 19) / 4
  np.testing.assert_array_equal((Tensor(left) + Tensor(right)).numpy(), left + right)


@requires_qcomcl
@pytest.mark.parametrize('groups,local', [((3, 1, 1), (7, 1, 1)), ((2, 3, 2), (3, 2, 2))])
def test_opencl_invocation_builtins(groups, local):
  # Each builtin occupies its own output field; swaps cannot cancel numerically.
  source = '''__kernel void invocation_ids(__global uint *out) {
    uint x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
    uint index = (x + get_global_size(0) * (y + get_global_size(1) * z)) * 19;
    for (uint axis = 0; axis < 3; axis++) {
      out[index + axis] = get_global_id(axis);
      out[index + 3 + axis] = get_local_id(axis);
      out[index + 6 + axis] = get_group_id(axis);
      out[index + 9 + axis] = get_global_size(axis);
      out[index + 12 + axis] = get_local_size(axis);
      out[index + 15 + axis] = get_num_groups(axis);
    }
    out[index + 18] = get_work_dim();
  }'''
  global_size = tuple(g * l for g, l in zip(groups, local))
  output = Tensor.empty(math.prod(global_size) * 19, dtype=dtypes.uint32)
  actual = source_kernel(source, 'invocation_ids', output, groups, local).numpy().reshape(-1, 19)
  expected = []
  for z in range(global_size[2]):
    for y in range(global_size[1]):
      for x in range(global_size[0]):
        position = (x, y, z)
        expected.append((*position, *(v % l for v, l in zip(position, local)),
                         *(v // l for v, l in zip(position, local)), *global_size, *local, *groups, 3))
  np.testing.assert_array_equal(actual, np.array(expected, dtype=np.uint32))


@requires_qcomcl
def test_jit_refreshes_arguments_without_reusing_previous_values():
  # Cached shader/command structure must still consume the current input buffers.
  @TinyJit
  def calculate(left, right):
    return (left * 3 + right).realize()

  for step in range(5):
    left = np.arange(65, dtype=np.int32) - step * 7
    right = np.arange(65, dtype=np.int32)[::-1].copy() + step * 11
    np.testing.assert_array_equal(calculate(Tensor(left), Tensor(right)).numpy(), left * 3 + right)


@requires_qcomcl
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_generated_integer_widths_and_selection(dtype):
  # Compiler-selected narrow loads, widened arithmetic and writeback keep values.
  left = np.array([0, 1, 7, 31, 63, 127], dtype=dtype)
  right = np.array([127, 63, 31, 7, 1, 0], dtype=dtype)
  expected = np.where(left > right, left * 3, right + 5).astype(dtype)
  a, b = Tensor(left), Tensor(right)
  np.testing.assert_array_equal((a > b).where(a * 3, b + 5).numpy(), expected)


@requires_qcomcl
def test_generated_signed_pointer_offsets_stay_within_the_input():
  # Reversing a buffer exercises negative index offsets without invalid addresses.
  values = np.arange(53, dtype=np.int32) * 17 - 421
  np.testing.assert_array_equal((Tensor(values).flip(0) + 3).numpy(), values[::-1] + 3)


@requires_qcomcl
def test_generated_signed_byte_extremes_are_extended_before_arithmetic():
  values = np.array([-128, -127, -17, -1, 0, 1, 17, 127], dtype=np.int8)
  actual = (Tensor(values).cast(dtypes.int16) + 200).numpy()
  np.testing.assert_array_equal(actual, values.astype(np.int16) + 200)


@requires_qcomcl
def test_generated_reduction_and_barrier():
  values = (np.arange(13 * 67, dtype=np.float32).reshape(13, 67) % 23 - 11) / 8
  np.testing.assert_array_equal(Tensor(values).sum(axis=1).numpy(), values.sum(axis=1))


@requires_qcomcl
def test_generated_float_arithmetic_and_conversion():
  values = np.array([-10.25, -3.5, -.125, 0, .5, 1.25, 7.75], dtype=np.float32)
  actual = (Tensor(values) * 1.234375 + 2.345703125).numpy()
  expected = values * np.float32(1.234375) + np.float32(2.345703125)
  np.testing.assert_array_equal(actual, expected)
  np.testing.assert_array_equal(Tensor(values).cast(dtypes.int32).numpy(), values.astype(np.int32))


@requires_qcomcl
def test_jit_refreshes_symbolic_scalar_arguments():
  # The compiled loop bound comes from a current scalar kernarg on every replay.
  @TinyJit
  def reduce_columns(value):
    return (value * 2 - 1).sum().realize()

  values = np.arange(30, dtype=np.float32).reshape(3, 10) / 8
  data = Tensor(values).realize()
  for columns in (1, 3, 5, 2, 10):
    bound = Variable('columns', 1, 10).bind(columns)
    np.testing.assert_array_equal(reduce_columns(data[:, :bound]).numpy(), (values[:, :columns] * 2 - 1).sum())


@requires_qcomcl
def test_compiler_packed_half_constants_are_not_float32_demotions():
  # The QCOMCL compiler stores two half constants per 32-bit constant word.
  source = '''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void packed_half(__global float *out) {
    uint i = get_global_id(0);
    half value = convert_half_rte((float)i / 8.0f - 4.0f);
    half product = value * (half)1.234375f;
    out[i] = convert_float(product + (half)2.345703125f);
  }'''
  values = (np.arange(65, dtype=np.float32) / 8 - 4).astype(np.float16)
  expected = (values * np.float16(1.234375) + np.float16(2.345703125)).astype(np.float32)
  actual = source_kernel(source, 'packed_half', Tensor.empty(65, dtype=dtypes.float32), (5, 1, 1), (13, 1, 1)).numpy()
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('index,expected', [(0, 0xbd00), (1, 0x40b1)])
def test_half_sources_select_both_halves_of_a_packed_constant_word(index, expected):
  from test.mockgpu.qcom.emu import Workgroup, decode, validate_constant_footprint
  from test.mockgpu.qcom.test_emu import Memory
  # Inert exact-value fixture: -1.25 and 2.345703125 occupy adjacent half slots.
  group = Workgroup(struct.pack('<HH', 0xbd00, 0x40b1), (1, 1, 1), (0, 0, 0),
                    0xfc, 0xfc, 0xfc, Memory({}), 0, 0, constant_demotion=False)
  word = (2 << 61) | (6 << 53) | (2 << 32) | 0x1000 | index
  instructions = decode(struct.pack('<2Q', word, 6 << 55))
  validate_constant_footprint(instructions, 1, constant_demotion=False)
  group.run(instructions)
  assert int(group.registers[1][2, 0]) == expected


@pytest.mark.parametrize('dispatch,shader_cl,sampler_cl', [(True, False, False), (False, True, True), (True, True, False)])
def test_inconsistent_compiler_modes_reject_before_a_prefix_store(dispatch, shader_cl, sampler_cl):
  from tinygrad.runtime.autogen import mesa
  from tinygrad.runtime.ops_qcom import pkt4_hdr, qreg
  from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
  from test.mockgpu.qcom.test_pm4 import Memory, launch_words, packet, signal
  memory = Memory(65536)
  words = launch_words(memory)[:-5]
  words += (pkt4_hdr(mesa.REG_A6XX_SP_MODE_CNTL, 1),
            qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_CL if shader_cl else mesa.ISAMMODE_GL, constant_demotion_enable=not shader_cl))
  words += (pkt4_hdr(mesa.REG_A6XX_TPL1_MODE_CNTL, 1),
            qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_CL if sampler_cl else mesa.ISAMMODE_GL))
  words += packet(mesa.CP_RUN_OPENCL, 0) if dispatch else packet(mesa.CP_EXEC_CS, 0, 1, 1, 1)
  with pytest.raises(PM4Fault):
    execute_command(signal(memory.base, 91) + words, memory, 0)
  assert memory.read(memory.base, 4) == bytes(4)


def test_short_opencl_builtin_bank_rejects_before_a_prefix_store():
  from tinygrad.runtime.autogen import mesa
  from tinygrad.runtime.ops_qcom import pkt4_hdr, qreg
  from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
  from test.mockgpu.qcom.test_pm4 import Memory, launch_words, packet, signal
  memory = Memory(65536)
  words = launch_words(memory)[:-5]
  words += (pkt4_hdr(mesa.REG_A6XX_SP_MODE_CNTL, 1), qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_CL))
  words += (pkt4_hdr(mesa.REG_A6XX_TPL1_MODE_CNTL, 1), qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_CL))
  words += packet(mesa.CP_LOAD_STATE6_FRAG,
                  qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                       state_block=mesa.SB6_CS_SHADER, num_unit=4),
                  (memory.base + 8192) & 0xffffffff, (memory.base + 8192) >> 32)
  words += packet(mesa.CP_RUN_OPENCL, 0)
  with pytest.raises(PM4Fault):
    execute_command(signal(memory.base, 91) + words, memory, 0)
  assert memory.read(memory.base, 4) == bytes(4)


@requires_qcomcl
def test_generated_nonzero_entry_point_and_helper_return():
  # The ordinary compiler places this helper before the kernel's entry point.
  # Starting at zero or returning to the wrong lane PC produces a wrong result.
  source = '''__attribute__((noinline)) float helper(float value) { return value * value + 1.0f; }
  __kernel void call_helper(__global float *out, __global const float *input) {
    uint i = get_global_id(0);
    out[i] = helper(input[i]);
  }'''
  values = (np.arange(21, dtype=np.float32) - 10) / 4
  actual = source_kernel(source, 'call_helper', Tensor.empty(21, dtype=dtypes.float32), (7, 1, 1), (3, 1, 1),
                         inputs=(Tensor(values),)).numpy()
  np.testing.assert_array_equal(actual, values * values + 1)


def test_callee_predicate_changes_are_included_in_constant_admission():
  from test.mockgpu.qcom.emu import validate_constant_footprint
  from test.mockgpu.qcom.test_constant_admission import END, constant_move, control
  # Callee PREDF/PREDT can make the caller's JUMP fall through after RET.
  for predicate in ('predf', 'predt'):
    program = (control('call', 4), control('jump', 2), constant_move(7), END,
               control(predicate), control('ret'))
    with pytest.raises(RuntimeError, match='constant source span'):
      validate_constant_footprint(program, 1)


def test_entry_point_ignores_uncalled_helper_constants_but_checks_called_helpers():
  from test.mockgpu.qcom.emu import validate_constant_footprint
  from test.mockgpu.qcom.test_constant_admission import END, constant_move, control
  program = (constant_move(7), control('ret'), END, control('call', -3), END)
  validate_constant_footprint(program, 1, entry_pc=2)
  with pytest.raises(RuntimeError, match='constant source span'):
    validate_constant_footprint(program, 1, entry_pc=3)
  with pytest.raises(RuntimeError, match='return.*without a call'):
    validate_constant_footprint(program, 8, entry_pc=1)


@pytest.mark.parametrize('entry', [-1, 1, True])
def test_entry_point_must_select_an_existing_instruction(entry):
  from test.mockgpu.qcom.emu import validate_constant_footprint
  from test.mockgpu.qcom.test_constant_admission import END
  with pytest.raises(RuntimeError, match='entry point'):
    validate_constant_footprint((END,), 0, entry_pc=entry)


@pytest.mark.parametrize('word,expected', [
  (0x50f20001105f0000, [0, 0, 0xffffffff, 0]),
  (0x40f5080500000000, [0, 0, 0, 0xffffffff]),
])
def test_generated_float_comparisons_keep_integer_mask_bits(word, expected):
  from test.mockgpu.qcom.emu import Workgroup, decode
  from test.mockgpu.qcom.test_emu import Memory
  # Exact CMPV.F words emitted by the ordinary float-to-half conversion kernel.
  # Numeric conversion of the result to float would corrupt the following masks.
  group = Workgroup(bytes(96 * 4), (4, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, Memory({}), 0, 0)
  group.registers[0][0] = np.array([-1, 0, .5, np.nan], dtype=np.float32).view(np.uint32)
  instructions = decode(struct.pack('<2Q', word, 6 << 55))
  group.run(instructions)
  destination = instructions[0].operands['DST'].index
  np.testing.assert_array_equal(group.registers[0][destination], np.array(expected, dtype=np.uint32))


def test_compiler_legacy_barrier_waits_for_later_lane_stores():
  from test.mockgpu.qcom.emu import Instruction, decode
  from test.mockgpu.qcom.test_control import END, R, branch, jump, state
  # One lane reaches the barrier before the remaining lanes' stores at PC10.
  # Replacing the barrier with a NOP makes lane0 observe stale neighbor bytes.
  group = state(shared_bytes=16)
  group.registers[0][248] = [0, 1, 1, 1]
  group.registers[0][0] = [0, 4, 8, 12]
  group.registers[0][1] = [1, 2, 3, 4]
  group.registers[0][2] = [4, 8, 12, 0]
  store = Instruction('stl', {'TYPE':3, 'SIZE':1}, {'DST':R(0), 'SRC':R(1)}, 6 << 61)
  load = Instruction('ldl', {'TYPE':3, 'SIZE':1}, {'DST':R(3), 'SRC':R(2)}, 6 << 61)
  # These exact legacy FENCE/BAR forms came from the QCOMCL GEMV compiler.
  fence, barrier = decode(struct.pack('<2Q', 0xe098000000000000, 0xf000000000000000))
  nop = Instruction('nop', {}, {}, 0)
  group.run((branch(10), store, jump(3), nop, nop, barrier, load, jump(5), nop, nop, store, jump(-6), fence, END))
  np.testing.assert_array_equal(group.registers[0][3], [2, 3, 4, 1])


@pytest.mark.parametrize('extra', [1, 1 << 32, 1 << 45, 1 << 50])
def test_legacy_barrier_does_not_admit_unrecognized_reserved_bits(extra):
  from test.mockgpu.qcom.emu import decode
  with pytest.raises(RuntimeError, match='unrecognized encoding'):
    decode(struct.pack('<Q', 0xf000000000000000 | extra))


def test_generated_unsigned_mad_uses_low_digits_and_preserves_its_wide_result():
  from test.mockgpu.qcom.emu import Workgroup, decode
  from test.mockgpu.qcom.test_emu import Memory
  # Exact MAD.U16 from QCOMCL range reduction. Real compiler kernels establish
  # full register slots despite the disassembler's HALF labels; multiplying
  # only low16 digits and retaining carries are separate arithmetic obligations.
  group = Workgroup(b'', (4, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, Memory({}), 0, 0)
  group.registers[False][5] = [0xdead03e8, 65535, 0x10000, 0xabc0007b]
  group.registers[False][9] = [0xdead03e8, 0x1234ffff, 0xffffffff, 0xface01c8]
  group.registers[False][6] = [2000, 65535, 7, 789]
  group.run(decode(struct.pack('<2Q', 0x6004c80900068005, 6 << 55)))
  np.testing.assert_array_equal(group.registers[False][9], np.array([1002000, 4294901760, 7, 56877], dtype=np.uint32))


def test_generated_unsigned_comparison_masks_do_not_sign_extend_inputs():
  from test.mockgpu.qcom.emu import Workgroup, decode
  from test.mockgpu.qcom.test_emu import Memory
  # Actual CMPV.U r0.x,r0.x,64 from QCOMCL's compiled software range reduction.
  group = Workgroup(b'', (4, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, Memory({}), 0, 0)
  group.registers[False][0] = [0, 63, 64, 0xffffffff]
  group.run(decode(struct.pack('<2Q', 0x4430000020400000, 6 << 55)))
  np.testing.assert_array_equal(group.registers[False][0], np.array([0xffffffff, 0xffffffff, 0, 0], dtype=np.uint32))


@requires_qcomcl
def test_compiler_masked_mad_uses_full_register_inputs_and_accumulator():
  # Current compiler loads these arguments into full registers and emits one
  # MAD.U16. Its two multiplicands use low16 bits; the accumulator stays32 bits.
  source = '''__kernel void masked_mad(__global uint *out, __global const uint *a,
                                     __global const uint *b, __global const uint *c) {
    uint i = get_global_id(0);
    out[i] = (a[i] & 65535u) * (b[i] & 65535u) + c[i];
  }'''
  left = np.array([3, 0x10002, 65535, 0x1234ffff], dtype=np.uint32)
  right = np.array([5, 0x20003, 65535, 0xf7654321], dtype=np.uint32)
  addend = np.array([7, 0x12345678, 0xffffffff, 0xabcd1234], dtype=np.uint32)
  expected = (((left & 65535).astype(np.uint64) * (right & 65535).astype(np.uint64) + addend) & 0xffffffff).astype(np.uint32)
  actual = source_kernel(source, 'masked_mad', Tensor.empty(4, dtype=dtypes.uint32), (2, 1, 1), (2, 1, 1),
                         inputs=(Tensor(left), Tensor(right), Tensor(addend))).numpy()
  np.testing.assert_array_equal(actual, expected)


@requires_qcomcl
def test_compiler_unsigned_halving_add_avoids_overflow(monkeypatch):
  from test.mockgpu.qcom.emu import Workgroup
  # Standard OpenCL hadd computes the full mathematical sum before halving.
  # The vendor compiler maps it directly to full-width ADD.U with EI set.
  source = '''__kernel void half_sum(__global uint *out, __global const uint *a, __global const uint *b) {
    uint i = get_global_id(0);
    out[i] = hadd(a[i], b[i]);
  }'''
  left = np.array([0, 1, 0xffffffff, 0xffffffff, 0x80000000, 0x12345678], dtype=np.uint32)
  right = np.array([1, 2, 0xffffffff, 1, 0x80000000, 0x9abcdef0], dtype=np.uint32)
  expected = np.array([0, 1, 0xffffffff, 0x80000000, 0x80000000, 0x56789ab4], dtype=np.uint32)
  observed = []
  original = Workgroup.alu

  def record(self, instruction, lanes, repeat):
    if instruction.op == 'add.u' and instruction.fields.get('EI'):
      observed.append(len(lanes))
    return original(self, instruction, lanes, repeat)

  monkeypatch.setattr(Workgroup, 'alu', record)
  actual = source_kernel(source, 'half_sum', Tensor.empty(6, dtype=dtypes.uint32), (2, 1, 1), (3, 1, 1),
                         inputs=(Tensor(left), Tensor(right))).numpy()
  np.testing.assert_array_equal(actual, expected)
  assert sum(observed) == 6


@pytest.mark.parametrize('word', [0x5230800000010000, 0x5200800000010000, 0x5210c00000010000])
def test_other_ei_arithmetic_forms_remain_unsupported(word):
  from test.mockgpu.qcom.emu import decode
  # The new exception covers only the compiler-proven full-width unsigned form.
  with pytest.raises(RuntimeError, match='unsupported execution modifier'):
    decode(struct.pack('<2Q', word, 6 << 55))


@requires_qcomcl
def test_compiler_vector_select_uses_integer_msb_not_float_condition(monkeypatch):
  from test.mockgpu.qcom.emu import Workgroup
  # OpenCL vector select uses the condition's MSB. NaN/-0-shaped integer words
  # distinguish that contract from interpreting condition bits as floating data.
  source = '''__kernel void vector_select(__global float4 *out, __global const float4 *left,
                                        __global const float4 *right, __global const int4 *condition) {
    uint i = get_global_id(0);
    out[i] = select(left[i], right[i], condition[i]);
  }'''
  left = np.arange(10, 18, dtype=np.float32)
  right = np.arange(30, 38, dtype=np.float32)
  condition = np.array([0, 0x80000000, 0xffffffff, 0x7fc00000, 0xffc00000, 1, 0xfffffffe, 0x7fffffff], dtype=np.uint32).view(np.int32)
  observed = []
  original = Workgroup.alu

  def record(self, instruction, lanes, repeat):
    if instruction.op == 'sel.s32':
      observed.append(len(lanes))
    return original(self, instruction, lanes, repeat)

  monkeypatch.setattr(Workgroup, 'alu', record)
  actual = source_kernel(source, 'vector_select', Tensor.empty(8, dtype=dtypes.float32), (2, 1, 1), (1, 1, 1),
                         inputs=(Tensor(left), Tensor(right), Tensor(condition))).numpy()
  np.testing.assert_array_equal(actual, np.array([10, 31, 32, 13, 34, 15, 36, 17], dtype=np.float32))
  assert sum(observed) == 8


@requires_qcomcl
def test_compiler_packed_predicate_bits_guard_transposed_convolution_loads(monkeypatch):
  from test.mockgpu.qcom.emu import Workgroup
  # Several independent edge conditions live across the reduction loop. The
  # compiler packs them into a half register and GETBIT restores each predicate.
  values = (np.arange(2 * 4 * 9 * 9, dtype=np.float32).reshape(2, 4, 9, 9) % 11) - 5
  weights = (np.arange(4 * 4 * 3 * 3, dtype=np.float32).reshape(4, 4, 3, 3) % 7) - 3
  bias = np.array([-2, -1, 1, 2], dtype=np.float32)
  expected = np.zeros((2, 4, 11, 11), dtype=np.float32)
  for incoming in range(4):
    for outgoing in range(4):
      for y in range(3):
        for x in range(3):
          expected[:, outgoing, y:y+9, x:x+9] += values[:, incoming] * weights[incoming, outgoing, y, x]
  expected += bias.reshape(1, 4, 1, 1)
  observed = []
  original = Workgroup.alu

  def record(self, instruction, lanes, repeat):
    if instruction.op == 'getbit.b':
      observed.append(len(lanes))
    return original(self, instruction, lanes, repeat)

  monkeypatch.setattr(Workgroup, 'alu', record)
  actual = Tensor(values).conv_transpose2d(Tensor(weights), Tensor(bias)).numpy()
  np.testing.assert_array_equal(actual, expected)
  assert sum(observed) > 0


@pytest.mark.parametrize('word', [
  0x4760000020000009,  # A normal GPR destination has no proven numeric result representation.
  0x477040f920000009,  # Full-register packed input is not emitted by this producer fixture.
  0x476000f900080009,  # A register-selected bit index is not qualified.
  0x476000f920100009,  # Bit 16 lies beyond the admitted half-register source.
])
def test_other_getbit_forms_remain_unsupported(word):
  from test.mockgpu.qcom.emu import decode
  with pytest.raises(RuntimeError, match='unsupported GETBIT predicate form'):
    decode(struct.pack('<2Q', word, 6 << 55))
