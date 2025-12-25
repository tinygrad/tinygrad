# do not change these tests. we need to fix bugs to make them pass
# the Inst constructor should be looking at the types of the fields to correctly set the value

import unittest
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.asm import asm
from extra.assembly.rdna3.test.test_roundtrip import compile_asm

class TestIntegration(unittest.TestCase):
  def tearDown(self):
    b = self.inst.to_bytes()
    st = self.inst.disasm()
    reasm = asm(st)
    desc = f"{self.inst} {b} {st} {reasm}"
    self.assertEqual(b, compile_asm(st), desc)
    # TODO: this compare should work for valid things
    #self.assertEqual(self.inst, reasm)
    self.assertEqual(repr(self.inst), repr(reasm))

  def test_simple_stos(self):
    self.inst = s_mov_b32(s[0], s[1])

  def test_simple_wrong(self):
    # TODO: this should raise an exception on construction, s[1] is not a valid type
    with self.assertRaises(TypeError):
      self.inst = s_mov_b32(v[0], s[1])

  def test_simple_vtov(self):
    # TODO: this is broken, it's reconstructing with s[1] and not v[1]
    self.inst = v_mov_b32_e32(v[0], v[1])

  def test_simple_stov(self):
    self.inst = v_mov_b32_e32(v[0], s[2])

  def test_simple_float_to_v(self):
    # TODO: this should be the magic float value 1.0
    self.inst = v_mov_b32_e32(v[0], 1.0)

  def test_simple_int_to_v(self):
    # TODO: this should be the constant 1, not s[0]
    self.inst = v_mov_b32_e32(v[0], 1)

if __name__ == "__main__":
  unittest.main()
