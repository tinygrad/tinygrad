import struct, unittest, weakref
from dataclasses import replace
from tinygrad.device import TinyELF
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import UOp

class TestKernelArgs(unittest.TestCase):
  def test_signature_does_not_own_uops(self):
    p = UOp.param(997, dtypes.int32, 997, name='signature_lifetime')
    ref, signature = weakref.ref(p), (p.kernel_param,)
    del p
    self.assertIsNone(ref())
    self.assertEqual(signature[0].arg.slot, 997)

  def test_pack_repeated_slot(self):
    # IMAGE can have distinct ABI descriptors that share a CALL slot.
    p = UOp.param(2, dtypes.float32, 4).kernel_param
    image = p._replace(arg=replace(p.arg, image=(1, 1)))
    scalar = UOp.param(0, dtypes.int16, addrspace=AddrSpace.ALU).kernel_param
    signature = (scalar, image, p)
    self.assertEqual(TinyELF.pack(signature, (-3, 0x1000, 0x1000)),
                     struct.pack('<h6xQQ', -3, 0x1000, 0x1000))
    self.assertEqual(TinyELF.pack((p, scalar), (0x1000, -3), 12), bytearray(12) + struct.pack('<Qh', 0x1000, -3))
    self.assertEqual(TinyELF.pack((), (), 12), bytearray(12))
    for args in ((-3, 0x1000), (-3, 0x1000, 0x1000, 7)):
      with self.assertRaises(ValueError): TinyELF.pack(signature, args)
