"""Bounded QCOMCL SEL.F32 sign selection; unproved conditions and payloads fail closed."""
import math
import struct
import numpy as np
import pytest
from tinygrad import Tensor
from test.mockgpu.qcom.emu import Workgroup, decode
from test.mockgpu.qcom.test_qcomcl import requires_qcomcl

SEL = 0x6680800200004000  # sel.f32 r0.z, -r0.x, r0.y, r0.x
END = 6 << 55
CONDITIONS = (0x3f7fffff, 0xffffffff)


def state(payloads, conditions):
  group = Workgroup(b'', (len(payloads), 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, object(), 0, 0)
  group.registers[0][0] = payloads
  group.registers[0][1] = conditions
  return group


def expected_sign_choice(payload, condition):
  # Scalar IEEE conversion, Python negation and bit packing are independent of
  # the emulator's raw-bit implementation. Normal F32 negation is exact.
  value = struct.unpack('<f', struct.pack('<I', payload))[0]
  selected = -value if condition == 0x3f7fffff else value
  return struct.unpack('<I', struct.pack('<f', selected))[0]


@pytest.mark.parametrize('word', [SEL, SEL & ~(255 << 32)])
def test_raw_payload_sign_and_source_order_with_destination_alias(word):
  payloads = [0x3fa00000, 0xbfa00000, 0x40000000, 0xc0600000, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff]
  random = np.random.default_rng(630).integers(0, 2**32, 256, dtype=np.uint32)
  payloads += [int(value) for value in random if 0 < (int(value) >> 23 & 255) < 255]
  pairs = [(payload, condition) for payload in payloads for condition in CONDITIONS]
  group = state([p for p, _ in pairs], [c for _, c in pairs])
  instructions = decode(struct.pack('<2Q', word, END))
  assert instructions[0].op == 'sel.f32'
  assert instructions[0].operands['SRC1'].modifier == 1
  assert instructions[0].operands['SRC2'].index == 1
  group.run(instructions)
  expected = [expected_sign_choice(p, c) for p, c in pairs]
  # Original oracle seed, then its negative payload, under both condition words.
  assert expected[:4] == [0xbfa00000, 0x3fa00000, 0x3fa00000, 0xbfa00000]
  assert expected == [p ^ (0x80000000 if c == 0x3f7fffff else 0) for p, c in pairs]
  np.testing.assert_array_equal(group.registers[0][instructions[0].operands['DST'].index], expected)


def test_every_other_condition_class_rejects_before_any_destination_write():
  conditions = [0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                0x7fc00001, 0xffc00001, 0x3f7ffffe, 0x3f800001, 0xfffffffe]
  conditions += [int(x) for x in np.random.default_rng(631).integers(0, 2**32, 1024, dtype=np.uint32) if int(x) not in CONDITIONS]
  instructions = decode(struct.pack('<2Q', SEL, END))
  for condition in conditions:
    group = state([0x3fa00000] * 3, [CONDITIONS[0], condition, CONDITIONS[1]])
    group.registers[0][2] = 0xdeadbeef
    before = group.registers[0].copy()
    with pytest.raises(RuntimeError, match='SEL.F32 condition outside admitted domain'):
      group.run(instructions)
    np.testing.assert_array_equal(group.registers[0], before)


@pytest.mark.parametrize('condition', CONDITIONS)
@pytest.mark.parametrize('payload', [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                                    0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345])
def test_unverified_zero_subnormal_infinite_and_nan_payloads_reject(payload, condition):
  # These raw words remain an oracle question, not extrapolated from sine.
  group = state([0x3fa00000, payload, 0xbfa00000], [condition] * 3)
  group.registers[0][2] = 0xdeadbeef
  before = group.registers[0].copy()
  with pytest.raises(RuntimeError, match='SEL.F32 payload outside admitted domain'):
    group.run(decode(struct.pack('<2Q', SEL, END)))
  np.testing.assert_array_equal(group.registers[0], before)


@pytest.mark.parametrize('word', [
  SEL ^ (1 << 14),   # No negation on SRC1 is a different form.
  SEL | (1 << 30),   # A modified condition is not the traced producer.
  SEL | (1 << 31),   # Negating both data choices is not established.
  SEL | (1 << 16),   # Different SRC3 storage is not the bounded sign choice.
  SEL | (1 << 12),   # A constant source is not this register-only form.
  SEL | (1 << 40),   # Repeated execution needs separate evidence.
  SEL | (1 << 42),   # Saturated output is not admitted.
  SEL | (1 << 43),   # At zero repeat this decodes as NOP=1, not source advancement.
  SEL | (1 << 15),   # At zero repeat this decodes as NOP=2.
  SEL | (1 << 29),   # Advancing SRC3 is not admitted.
  SEL | (1 << 46),   # Half-width output is not admitted.
])
def test_unverified_encoded_forms_reject_during_decode(word):
  with pytest.raises(RuntimeError, match='unsupported SEL.F32 form'):
    decode(struct.pack('<2Q', word, END))


def test_madsh_u16_stays_unsupported():
  with pytest.raises(RuntimeError, match='unsupported instruction madsh.u16'):
    decode(struct.pack('<2Q', 0x608a800b000b000d, END))


@requires_qcomcl
def test_real_compiler_sine_range_and_original_cosine_reproduction(monkeypatch):
  # Independent range/quarter-turn/large-argument data separated the six payload
  # models before admission. The original reproduction retains both assertions.
  from test.mockgpu.qcom.test_tensor_semantics import test_trigonometric_reduction_preserves_large_arguments
  grid = np.linspace(-128, 128, 1025, dtype=np.float32)
  quadrants = np.arange(-32, 33, dtype=np.float64) * (math.pi / 2)
  boundaries = (quadrants[:, None] + np.array([-1e-5, 0, 1e-5])).reshape(-1).astype(np.float32)
  large = np.array([-1e6, -1e5, -1e4, -1000, -10, -.5, 0, .5, 10, 1000, 1e4, 1e5, 1e6], np.float32)
  values = np.unique(np.concatenate((grid, boundaries, large)))
  expected = np.sin(values)
  scalar = np.array([math.sin(float(value)) for value in values], np.float32)
  np.testing.assert_allclose(expected, scalar, rtol=1e-6, atol=1e-7)
  observed = set()
  original = Workgroup.alu

  def record(self, instruction, lanes, repeat):
    if instruction.op == 'sel.f32':
      observed.update(map(int, self.read(instruction.operands['SRC2'], lanes, repeat)))
    return original(self, instruction, lanes, repeat)

  monkeypatch.setattr(Workgroup, 'alu', record)
  np.testing.assert_allclose(Tensor(values).sin().numpy(), expected, atol=3e-3, rtol=3e-3)
  assert observed == set(CONDITIONS)
  test_trigonometric_reduction_preserves_large_arguments()
