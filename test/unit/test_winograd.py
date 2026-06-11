import unittest, sys
import numpy as np
from tinygrad import Tensor, GlobalCounters, Context, nn
from tinygrad.helpers import WINO

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinogradClose(unittest.TestCase):
  def test_close(self):
    inp = Tensor.rand(1, 16, 16, 16)
    conv = nn.Conv2d(16, 16, 3)
    conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("non winograd")
    with Context(WINO=0):
      cmp = conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("winograd")
    with Context(WINO=1):
      test = conv(inp).realize()
    np.testing.assert_allclose(cmp.numpy(), test.numpy(), atol=1e-5)

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_padded_conv2d(self):
    # tests padding order in winograd
    x,w = Tensor.rand(1,3,11,28).realize(), Tensor.rand(4,3,3,3).realize()
    with Context(WINO=0): expected = Tensor.conv2d(x,w,padding=1).realize()
    with Context(WINO=1): result = Tensor.conv2d(x,w,padding=1).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestWinogradGraphRewrite(unittest.TestCase):
  def _ref_and_count(self, x: Tensor, w: Tensor, bypass_cache=True, **kwargs) -> tuple[np.ndarray, int]:
    with Context(WINO=0, WINO_GRAPH=0):
      GlobalCounters.reset()
      if bypass_cache:
        got_ref = (x * 1.00001).conv2d(w, **kwargs).realize().numpy()
      else:
        got_ref = x.conv2d(w, **kwargs).realize().numpy()
      ops_ref = GlobalCounters.global_ops
    return got_ref, ops_ref

  def _wino_graph_and_count(self, x: Tensor, w: Tensor, **kwargs) -> tuple[np.ndarray, int]:
    with Context(WINO=0, WINO_GRAPH=1):
      GlobalCounters.reset()
      got_wino = x.conv2d(w, **kwargs).realize().numpy()
      ops_wino = GlobalCounters.global_ops
    return got_wino, ops_wino

  def test_basic_3x3_no_padding(self):
    x = Tensor.rand(1, 2, 10, 10).realize()
    w = Tensor.rand(8, 2, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_output_8x8(self):
    x = Tensor.rand(1, 2, 12, 12).realize()
    w = Tensor.rand(8, 2, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_multi_channel(self):
    x = Tensor.rand(1, 8, 14, 14).realize()
    w = Tensor.rand(16, 8, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_batch(self):
    x = Tensor.rand(2, 4, 10, 10).realize()
    w = Tensor.rand(8, 4, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_non_tile_output(self):
    x = Tensor.rand(1, 8, 15, 15).realize()
    w = Tensor.rand(16, 8, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_padding_1(self):
    x = Tensor.rand(1, 3, 10, 10).realize()
    w = Tensor.rand(8, 3, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w, padding=1)
    wino, ops_wino = self._wino_graph_and_count(x, w, padding=1)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_3x3_non_square_spatial(self):
    x = Tensor.rand(1, 8, 10, 16).realize()
    w = Tensor.rand(8, 8, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    np.testing.assert_allclose(wino, ref, atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_ref, ops_wino, "rewrite did not fire")

  def test_5x5_not_rewritten(self):
    x = Tensor.rand(1, 4, 10, 10).realize()
    w = Tensor.rand(8, 4, 5, 5).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    self.assertEqual(ops_ref, ops_wino, "5x5 rewrite should not fire")

  def test_1x1_not_rewritten(self):
    x = Tensor.rand(1, 4, 10, 10).realize()
    w = Tensor.rand(8, 4, 1, 1).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    self.assertEqual(ops_ref, ops_wino, "1x1 rewrite should not fire")

  def test_stride2_not_rewritten(self):
    x = Tensor.rand(1, 4, 10, 10).realize()
    w = Tensor.rand(8, 4, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False, stride=2)
    wino, ops_wino = self._wino_graph_and_count(x, w, stride=2)
    self.assertEqual(ops_ref, ops_wino, "stride=2 rewrite should not fire")

  def test_dilation2_not_rewritten(self):
    x = Tensor.rand(1, 4, 12, 12).realize()
    w = Tensor.rand(8, 4, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False, dilation=2)
    wino, ops_wino = self._wino_graph_and_count(x, w, dilation=2)
    self.assertEqual(ops_ref, ops_wino, "dilation=2 rewrite should not fire")

  def test_grouped_conv_rejects(self):
    x = Tensor.rand(1, 8, 10, 10).realize()
    w = Tensor.rand(8, 4, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False, groups=2)
    wino, ops_wino = self._wino_graph_and_count(x, w, groups=2)
    self.assertEqual(ops_ref, ops_wino, "grouped conv rewrite should not fire")

  def test_small_cout_heuristic_rejects(self):
    x = Tensor.rand(1, 4, 10, 10).realize()
    w = Tensor.rand(4, 4, 3, 3).realize()
    ref, ops_ref = self._ref_and_count(x, w, bypass_cache=False)
    wino, ops_wino = self._wino_graph_and_count(x, w)
    self.assertEqual(ops_ref, ops_wino, "cout=4 rewrite should not fire")

  def test_manual_sliding_window_triggers_rewrite(self):
    def _manual_sliding_window(x, w):
      bs, cin, cout = 1, 2, 8
      ky, kx = 3, 3
      out_H, out_W = 8, 8
      x_windows = x._pool((ky, kx))
      x_expanded = x_windows.reshape(bs, 1, cin, out_H, out_W, ky, kx).expand(bs, cout, cin, out_H, out_W, ky, kx)
      w_expanded = w.reshape(1, cout, cin, 1, 1, ky, kx).expand(bs, cout, cin, out_H, out_W, ky, kx)
      return (x_expanded * w_expanded).sum(axis=(2, 5, 6))

    x = Tensor.rand(1, 2, 10, 10).realize()
    w = Tensor.rand(8, 2, 3, 3).realize()

    with Context(WINO=0, WINO_GRAPH=0):
      GlobalCounters.reset()
      got_without = _manual_sliding_window((x * 1.00001), w).realize()
      ops_without = GlobalCounters.global_ops

    with Context(WINO=0, WINO_GRAPH=1):
      GlobalCounters.reset()
      got_with = _manual_sliding_window(x, w).realize()
      ops_with = GlobalCounters.global_ops

    np.testing.assert_allclose(got_without.numpy(), got_with.numpy(), atol=1e-3, rtol=1e-3)
    self.assertNotEqual(ops_without, ops_with, "rewrite likely did not fire")

if __name__ == '__main__':
  unittest.main()
