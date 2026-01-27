"""Test elimination-based kernel optimization."""
import unittest
from tinygrad import Tensor, Device

class TestEliminate(unittest.TestCase):
  def test_solver_produces_single_survivor(self):
    from tinygrad.codegen.opt.eliminate import solve_matmul, MetalHardware
    survivors = solve_matmul(1024, 1024, 1024, MetalHardware(), verbose=False)
    self.assertEqual(len(survivors), 1)
    self.assertEqual(survivors[0].tile_m, 128)
    self.assertEqual(survivors[0].tile_n, 128)
    self.assertEqual(survivors[0].tile_k, 8)

  def test_config_to_opts(self):
    if Device.DEFAULT != "METAL": self.skipTest("Metal only")
    from tinygrad.codegen.opt import Opt, OptOps
    from tinygrad.codegen.opt.eliminate import solve_matmul, MetalHardware, config_to_opts
    survivors = solve_matmul(1024, 1024, 1024, MetalHardware(), verbose=False)
    opts = config_to_opts(survivors[0])
    expected = [
      Opt(op=OptOps.TC, axis=0, arg=(0, 1, 1)),
      Opt(op=OptOps.UPCAST, axis=0, arg=4),
      Opt(op=OptOps.UPCAST, axis=1, arg=4),
      Opt(op=OptOps.LOCAL, axis=1, arg=4),
    ]
    self.assertEqual(opts, expected)

  def test_opts_apply(self):
    if Device.DEFAULT != "METAL": self.skipTest("Metal only")
    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.codegen.opt.eliminate import solve_matmul, MetalHardware, config_to_opts
    a, b = Tensor.rand(1024, 1024), Tensor.rand(1024, 1024)
    sched = (a @ b).schedule()
    ast = None
    for item in sched:
      if hasattr(item, 'ast') and item.ast.op.name == 'SINK':
        k = Scheduler(item.ast, Device[Device.DEFAULT].renderer)
        k.convert_loop_to_global()
        if k.full_shape == [1024, 1024, 1024]: ast = item.ast; break
    self.assertIsNotNone(ast)
    k = Scheduler(ast, Device[Device.DEFAULT].renderer)
    k.convert_loop_to_global()
    for opt in config_to_opts(solve_matmul(1024, 1024, 1024, MetalHardware(), verbose=False)[0]):
      k.apply_opt(opt)
    self.assertEqual(k.full_shape, [32, 8, 32, 4, 2, 4, 4, 128])

class TestKernelDetection(unittest.TestCase):
  def test_conv2d_detection(self):
    from tinygrad.codegen.opt.eliminate import is_conv2d_kernel
    self.assertTrue(is_conv2d_kernel((64, 112, 112, 3, 7, 7)))
    self.assertTrue(is_conv2d_kernel((64, 56, 56, 64, 3, 3)))
    self.assertFalse(is_conv2d_kernel((1024, 1024, 1024)))

  def test_conv1x1_detection(self):
    from tinygrad.codegen.opt.eliminate import is_conv1x1_kernel
    self.assertTrue(is_conv1x1_kernel((64, 56, 56, 256)))
    self.assertFalse(is_conv1x1_kernel((1024, 1024, 1024)))

  def test_reduce_detection(self):
    from tinygrad.codegen.opt.eliminate import is_reduce_kernel
    from tinygrad.codegen.opt.postrange import AxisType
    self.assertTrue(is_reduce_kernel((1024,), [AxisType.REDUCE]))
    self.assertTrue(is_reduce_kernel((1024, 1024), [AxisType.GLOBAL, AxisType.REDUCE]))
    self.assertFalse(is_reduce_kernel((1024, 1024), [AxisType.GLOBAL, AxisType.GLOBAL]))

class TestEliminateOptimize(unittest.TestCase):
  def _find_kernel(self, sched, target_type):
    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.codegen.opt.eliminate import is_matmul_kernel, is_conv1x1_kernel, is_conv2d_kernel, is_reduce_kernel
    for item in sched:
      if hasattr(item, 'ast') and item.ast.op.name == 'SINK':
        k = Scheduler(item.ast, Device[Device.DEFAULT].renderer)
        k.convert_loop_to_global()
        shape, axis_types = tuple(k.full_shape), k.axis_types
        if not shape: continue
        if target_type == 'matmul' and is_matmul_kernel(shape): return item.ast, k
        if target_type == '1x1conv' and is_conv1x1_kernel(shape): return item.ast, k
        if target_type == 'conv2d' and is_conv2d_kernel(shape): return item.ast, k
        if target_type == 'reduce' and is_reduce_kernel(shape, axis_types): return item.ast, k
    return None, None

  def test_all_kernel_types(self):
    if Device.DEFAULT != "METAL": self.skipTest("Metal only")
    from tinygrad.codegen.opt.eliminate import eliminate_optimize
    tests = [
      ('matmul', Tensor.rand(1024, 1024) @ Tensor.rand(1024, 1024)),
      ('1x1conv', Tensor.rand(1, 256, 56, 56).conv2d(Tensor.rand(64, 256, 1, 1), stride=1)),
      ('conv2d', Tensor.rand(1, 64, 56, 56).conv2d(Tensor.rand(64, 64, 3, 3), stride=1, padding=1)),
      ('reduce', Tensor.rand(1024, 1024).sum(axis=1)),
    ]
    for ktype, tensor_expr in tests:
      ast, k = self._find_kernel(tensor_expr.schedule(), ktype)
      self.assertIsNotNone(ast, f"{ktype} kernel not found")
      opts = eliminate_optimize(k.copy())
      self.assertIsNotNone(opts, f"{ktype} opts is None")
      self.assertGreater(len(opts), 0, f"{ktype} opts is empty")
      for opt in opts: k.apply_opt(opt)

if __name__ == "__main__": unittest.main()
