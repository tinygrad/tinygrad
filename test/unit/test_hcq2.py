import unittest

from tinygrad.dtype import dtypes
from tinygrad.helpers import data64_le
from tinygrad.runtime.support.hcq2 import is_bare_addr, make_scatter_loops
from tinygrad.uop.ops import UOp


class TestHCQ2Patches(unittest.TestCase):
  @staticmethod
  def _address_words(offset:int|None):
    src = UOp.param(0, dtypes.uint8, (1,), device="AMD")
    addr = src.getaddr(("AMD",))
    if offset is not None: addr = addr + UOp.const(offset, dtypes.uint64)
    return [UOp.const(x, dtypes.uint32).simplify() for x in data64_le(addr)]

  def test_scatter_plain_input_address(self):
    self.assertTrue(all(map(is_bare_addr, self._address_words(None))))

  def test_scatter_input_address_offset(self):
    self.assertFalse(any(map(is_bare_addr, self._address_words(1 << 32))))

  def test_scatter_groups_by_destination(self):
    src = UOp.param(0, dtypes.uint8, (1,), device="AMD")
    vals = UOp.stack(*self._address_words(None))
    patches = [UOp.param(i, dtypes.uint32, (2,), device="AMD").index(UOp.stack(UOp.const(0), UOp.const(1))).store(vals) for i in (1, 2)]
    table = UOp.placeholder((1,), dtypes.uint64, next(UOp.unique_num), device="AMD")
    lt_patches:list[UOp] = []
    scatter = make_scatter_loops(patches, (table, {}, (), {src.getaddr(("AMD",)):0}), lt_patches)
    self.assertEqual(set(scatter), set(patches))
    self.assertEqual(len(lt_patches), 4)  # two plan tables for each destination


if __name__ == "__main__": unittest.main()
