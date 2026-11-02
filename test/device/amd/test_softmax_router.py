import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, moe_softmax_router

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestSoftmaxRouter(unittest.TestCase):
  def test_routing(self):
    if not amd_custom_kernels_supported("AMD"): self.skipTest("requires AMD custom kernels")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(81)
    N, E, K, scale = len(devices), 512, 10, 1.25
    for T, width in ((1, 1), (5, 2), (6, 0), (2112, 0)):
      table = rng.integers(-1, 430, size=(N, E+4, width), dtype=np.int32) if width else None
      tt = Tensor(table).shard(devices, axis=0).realize() if width else None
      @TinyJit
      def run(x):
        ys = moe_softmax_router(x, K, scale, tt)
        Tensor.realize(*ys)
        return ys
      for iteration in range(3):
        logits = rng.normal(size=(1, T, E)).astype(np.float32)
        logits[0, 0] = 0  # ties must prefer lower expert indices, in pairwise_topk's output order
        logits[0, 0, 1::2] = -0.
        if T > 1: logits[0, 1] = np.linspace(-1000, 1000, E)
        if T > 2: logits[0, 2] = -1000 - np.arange(E)
        indices = np.argsort(-logits, axis=-1, kind="stable")[..., :K][..., ::-1]
        picked = np.take_along_axis(logits, indices, axis=-1)
        prob = np.exp(picked - picked.max(axis=-1, keepdims=True))
        prob = prob / prob.sum(axis=-1, keepdims=True) * scale
        if width:
          expected = table[:, indices, :].reshape(N, 1, T, K*width)
          prob = np.broadcast_to(np.repeat(prob, width, axis=-1), expected.shape)
        else: expected = indices
        x = Tensor(logits).shard(devices, axis=None if width else 1).realize()
        sel, probs = run(x)
        with self.subTest(tokens=T, width=width, iteration=iteration):
          np.testing.assert_array_equal(sel.numpy(), expected)
          np.testing.assert_allclose(probs.numpy(), prob, rtol=2e-6, atol=2e-7)

if __name__ == "__main__": unittest.main()
