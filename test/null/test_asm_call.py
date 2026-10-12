import unittest
from test.mockgpu.amd import call
from test.mockgpu.amd.call import cfg_regions


class TestAsmCFG(unittest.TestCase):
  def test_diamond_with_unreachable_block(self):
    paths = {0:{1:0, 2:1}, 1:{3:0}, 2:{3:0}, 3:{}, 99:{1:0}}
    regions = cfg_regions(paths, 0)
    self.assertEqual(regions[0], (0, []))
    self.assertCountEqual(regions[1:3], [(1, []), (2, [])])
    self.assertEqual(regions[3:], [(3, [])])

  def test_nested_loops(self):
    paths = {0:{1:0}, 1:{2:0, 5:1}, 2:{3:0, 4:1}, 3:{2:0}, 4:{1:0}, 5:{}}
    self.assertEqual(cfg_regions(paths, 0), [(0, []), (1, [(1, []), (2, [(2, []), (3, [])]), (4, [])]), (5, [])])

  def test_multiple_backedges(self):
    paths = {0:{1:0}, 1:{2:0, 3:1}, 2:{1:0}, 3:{1:0, 4:1}, 4:{}}
    regions = cfg_regions(paths, 0)
    self.assertEqual([pc for pc,_ in regions], [0, 1, 4])
    self.assertEqual(regions[1][1][0], (1, []))
    self.assertCountEqual(regions[1][1][1:], [(2, []), (3, [])])

  def test_self_loop(self):
    self.assertEqual(cfg_regions({0:{0:0, 1:1}, 1:{}}, 0), [(0, [(0, [])]), (1, [])])

  def test_irreducible_loop(self):
    with self.assertRaisesRegex(AssertionError, "irreducible"):
      cfg_regions({0:{1:0, 2:1}, 1:{2:0}, 2:{1:0, 3:1}, 3:{}}, 0)

  def test_irreducible_inside_loop(self):
    # The outer SCC has one entry, but its nested cycle has two.
    with self.assertRaisesRegex(AssertionError, "irreducible"):
      cfg_regions({0:{1:0}, 1:{2:0, 3:1}, 2:{3:0}, 3:{2:0, 1:1}}, 0)

  def test_nested_loop_exits(self):
    # The inner loop can leave the outer loop directly, or continue through its tail.
    paths = {0:{1:0}, 1:{2:0, 5:1}, 2:{3:0, 4:1}, 3:{2:0, 5:1}, 4:{1:0}, 5:{}}
    self.assertEqual(cfg_regions(paths, 0), [(0, []), (1, [(1, []), (2, [(2, []), (3, [])]), (4, [])]), (5, [])])

  def test_long_chain(self):
    paths = {i:{i+1:0} for i in range(4999)} | {4999:{}}
    self.assertEqual(cfg_regions(paths, 0), [(i, []) for i in range(5000)])

  def test_long_chain_of_loops(self):
    paths = {i:{i:0, i+1:1} for i in range(4999)} | {4999:{4999:0}}
    self.assertEqual(cfg_regions(paths, 0), [(i, [(i, [])]) for i in range(5000)])

  def test_loop_lift_has_stable_cache_key(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import s_mov_b32, s_add_u32, s_cmp_lt_i32, s_cbranch_scc1, s_endpgm
    from tinygrad.renderer.amd.dsl import s
    code = b"".join((s_mov_b32(s[2], 0).to_bytes(), s_add_u32(s[2], s[2], 1).to_bytes(), s_cmp_lt_i32(s[2], 4).to_bytes(),
                     s_cbranch_scc1(simm16=-3).to_bytes(), s_endpgm().to_bytes()))
    dispatch = (1, 1, 1, 32, 1, 1, 0x19c, 0, None)
    first = call._lift(code, "rdna3", "PYTHON", dispatch)
    second = call._lift(code, "rdna3", "PYTHON", dispatch)
    self.assertEqual(first.src[0].key, second.src[0].key)


if __name__ == "__main__": unittest.main()
