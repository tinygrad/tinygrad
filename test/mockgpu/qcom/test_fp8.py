import unittest

from test.mockgpu.qcom.decoder import decode_ir3
from test.mockgpu.qcom.executor import execute_ir3
from test.mockgpu.qcom.corpus import END, FP8_HALF_COMPARE, ir3_program


class TestA630FP8(unittest.TestCase):
  def test_half_float_compare_converts_full_constant_slot(self):
    program = ir3_program(FP8_HALF_COMPARE, END)
    compare, _ = decode_ir3(program)
    self.assertEqual((compare.name, compare.dst, compare.srcs, compare.source_half, compare.condition),
                     ('cmps.f', ('hr', 3, 0), (('hr', 0, 3), ('hc', 5, 0)), True, 0))

    # The fp8 encoder compares against the full-float -448.0 slot through an
    # hc source.  It must compare -1/64 against -448, not against raw 0x0000.
    regs = {('hr', 0, 3): [0xa400], ('c', 5, 0): [0xc3e00000], ('hc', 5, 0): [0]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('hr', 3, 0)], [0])
