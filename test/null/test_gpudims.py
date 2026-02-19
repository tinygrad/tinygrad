import unittest
from tinygrad.codegen.gpudims import get_grouped_dims
from tinygrad.uop.ops import Ops, PatternMatcher, graph_rewrite, UPat
from tinygrad.helpers import flatten, dedup

class TestGroupedDims(unittest.TestCase):
  def test_grouped_dims(self):
    def _assert_grouped_dims(prefix, dims, max_sizes, reverse_dims, expected_sizes, assert_same_length = True):
      idxs = get_grouped_dims(prefix, dims, max_sizes, reverse_dims)
      loop_idxs = dedup(flatten([[y for y in x.toposort() if y.op is Ops.SPECIAL] for x in idxs]))
      loop_idxs = sorted(loop_idxs, key=lambda uop: uop.arg)
      sizes = [x.src[0].arg for x in loop_idxs]
      assert len(idxs) == len(dims), f"expected idxs to have same length as dims {len(dims)}, got {len(idxs)}"
      if assert_same_length:
        assert len(loop_idxs) == min(len(sizes), len(dims)), f"expected idxs to have length {min(len(sizes), len(dims))}, got {len(loop_idxs)}"
      assert sizes == expected_sizes, f"expected sizes={expected_sizes}, got {sizes=}"

    # no-op
    _assert_grouped_dims("gidx", (2,), (16,16,16), False, [2])
    _assert_grouped_dims("gidx", (2,3), (16,16,16), False, [2,3])

    # check reverse dims
    _assert_grouped_dims("gidx", (2,3), (16,16,16), True, [3,2])
    _assert_grouped_dims("gidx", (2,3,4), (16,16,16), False, [2,3,4])

    # test splitting globals:    len(dims) == len(max)
    _assert_grouped_dims("gidx", (64,3,4), (16,16,16), False, [16,12,4])
    _assert_grouped_dims("gidx", (64,3,4), (16,4,16), False, [16,3,16])
    _assert_grouped_dims("gidx", (64,3,4), (16,16,16), True, [16,3,16])
    _assert_grouped_dims("gidx", (128,3,4), (16,4,256), False, [16,3,32])
    _assert_grouped_dims("gidx", (4,4,512), (16,4,256), False, [8,4,256])

    # prefer group_dim strategy when possible
    _assert_grouped_dims("gidx", (512,4,2), (8192,2,2), False, [2048,2])

    # test splitting globals:    len(dims) < len(max)
    #                            len(dim)        ->          len(limited)
    #                              1             ->             2
    _assert_grouped_dims("gidx", (128,), (16,16,256), False, [16,8], False)
    #                              1             ->             3
    _assert_grouped_dims("gidx", (65536,), (16,16,256), False, [16,16,256], False)
    #                              2             ->             3
    _assert_grouped_dims("gidx", (128,128), (16,16,256), False, [16,16,64], False)
    #                              2             ->             2
    _assert_grouped_dims("gidx", (65536,2), (65535,65535,65535), False, [32768,4], False)
    # test when the only divisor is the square root of dim
    _assert_grouped_dims("gidx", (121,), (12,12,12), False, [11,11], False)

    # collapse on onto the left most axis
    _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), False, [6,4,5])
    _assert_grouped_dims("gidx", (2,3,4,5), (32,16,16), True, [20,3,2])

    # collapse on left-most available axis (the left most is too small)
    _assert_grouped_dims("gidx", (2,3,4,5), (4,16,16), False, [2,12,5])
    _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), True, [5,12,2])

    # dim too large and not factorable
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (23,), (16,16,16), False,)
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (128,3,4), (16,2,2), False,)

    # too large for sizes
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (2,3,4,5,6), (16,16,16))

    # TODO: In the above cases we only test if the shape after reshape is correct, never the indices.
    # We should check if the returned indices are correct, for all cases.
    # (65536, 2) -> (32768, 4)
    dims, expected_limited_dims = (65536,2), (32768, 4)
    idxs = get_grouped_dims("gidx", dims, (65535,65535,65535))
    def match_div(): raise RuntimeError("match_div")
    def match_mod(): raise RuntimeError("match_mod")
    flat_idx_pattern = UPat(Ops.SPECIAL, arg='gidx0')*expected_limited_dims[1]+UPat(Ops.SPECIAL, arg='gidx1')
    pm = PatternMatcher([
      (flat_idx_pattern//dims[1], match_div),
      (flat_idx_pattern%dims[1], match_mod)
    ])

    with self.assertRaises(RuntimeError) as error:
      graph_rewrite(idxs[0], pm)
    self.assertIn("match_div", str(error.exception))

    with self.assertRaises(RuntimeError) as error:
      graph_rewrite(idxs[1], pm)
    self.assertIn("match_mod", str(error.exception))

if __name__ == '__main__':
  unittest.main()
