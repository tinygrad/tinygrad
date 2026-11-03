import functools, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import qsa_prefill_mask, qsa_sparse_prefill, _qsa_tiles_kernel, pool_index_rows


@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestQSAPrefill(unittest.TestCase):
  def test_long_pool_lists(self):
    if Device[Device.DEFAULT].target[0] not in (11, 12): self.skipTest("requires RDNA3/4")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(232)
    T, P, top, kp = 5, 65536, 512, 4
    @TinyJit
    def run(scores, start): return pool_index_rows(scores, start, (start+T+kp-1)//kp, top, kp).realize()
    for iteration in range(3):
      start = 262120+iteration*4
      scores = rng.normal(size=(T, P)).astype(np.float32)
      scores[0] = 0
      result = run(Tensor(scores).shard(devices).realize(), UOp.variable("qsa_list_start", 0, 262139).bind(start))
      for device, buf in zip(devices, result.uop.buffer.bufs):
        indices = Tensor(UOp.from_buffer(buf, result.dtype, device)).reshape(result.shape).numpy()
        for row in range(T):
          complete = (start+row+1)//kp
          threshold = np.partition(scores[row, :complete], -top)[-top]
          pools = np.flatnonzero(scores[row, :complete] >= threshold)[:top]
          expected = np.concatenate(((pools[:, None]*kp+np.arange(kp)).ravel(), np.arange(complete*kp, start+row+1)))
          self.assertEqual(indices[row, -1], len(expected))
          np.testing.assert_array_equal(indices[row, :len(expected)], expected)

  def test_sparse_tiles(self): self._test_sparse_tiles(64)
  def test_model_head_dim(self): self._test_sparse_tiles(256)

  def _test_sparse_tiles(self, dim):
    if Device[Device.DEFAULT].target[0] not in (11, 12): self.skipTest("requires RDNA3/4")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(231)
    T, N, D, KP, top = len(devices)*32, 4096, dim, 4, 16
    H = len(devices)*2
    @TinyJit
    def run(scores, q, cache, start):
      mask = qsa_prefill_mask(scores, start, (start+T-7+KP-1)//KP, top, KP)
      out = qsa_sparse_prefill(q, cache, mask, start, start+T-7, KP)
      Tensor.realize(mask, out)
      return mask, out
    for iteration in range(3):
      start = N-T-23+iteration*4
      scores = rng.normal(size=(T, N//KP)).astype(np.float32)
      scores[0] = 0  # all ties: preserve the generic prefill mask's inclusive-threshold semantics
      scores[1] = -5
      scores[1, -100:-84] = 20  # some early union tiles are fully masked for this query
      q = rng.normal(0, .2, (1, H, T, D)).astype(np.float16)
      cache = rng.normal(size=(len(devices), 1, N, 2*D)).astype(np.float16)
      valid_end = start+T-7
      poolmask = np.zeros_like(scores, dtype=bool)
      for r in range(T):
        complete = min((start+r+1)//KP, (valid_end+KP-1)//KP)
        bits = scores[r].view(np.uint32)
        keys = np.where(bits >= 0x80000000, bits ^ np.uint32(0xffffffff), bits | np.uint32(0x80000000))
        threshold = np.partition(keys[:complete], -top)[-top] if complete >= top else 0
        poolmask[r, :complete] = keys[:complete] >= threshold
        if (tail:=(start+r+1)//KP) < N//KP: poolmask[r, tail] = True
      expected_mask = (poolmask.reshape(T//32, 32, N//KP).astype(np.uint32) *
                       (np.uint32(1) << np.arange(32, dtype=np.uint32))[None, :, None]).sum(1, dtype=np.uint32)
      mask, out = run(Tensor(scores).shard(devices, axis=0).realize(), Tensor(q).shard(devices, axis=1).realize(),
                      Tensor(cache).shard(devices, axis=0).realize(), UOp.variable("qsa_test_start", 0, N-T).bind(start))
      np.testing.assert_array_equal(mask.numpy(), expected_mask)
      tiles = Tensor.custom_kernel(Tensor.empty(T//32, N//32+1, dtype=dtypes.int32, device=devices), mask,
        fxn=functools.partial(_qsa_tiles_kernel, q_start=UOp.const(start, dtypes.weakint),
                              valid_end=UOp.const(valid_end, dtypes.weakint), kpool=KP))[0].realize()
      for device, buf in zip(devices, tiles.uop.buffer.bufs):
        indices = Tensor(UOp.from_buffer(buf, tiles.dtype, device)).reshape(tiles.shape).numpy()
        for row in range(T//32):
          expected = np.where(expected_mask[row].reshape(-1, 32//KP).any(1) &
                              (np.arange(N//32)*32 < min(start+(row+1)*32, valid_end)))[0]
          self.assertEqual(indices[row, -1], len(expected))
          np.testing.assert_array_equal(indices[row, :len(expected)], expected)
      actual = out.numpy()
      positions = np.arange(N)[None, :]
      visible = np.repeat(poolmask, KP, axis=1) & (positions <= start+np.arange(T)[:, None]) & (positions < valid_end)
      for h in range(H):
        k, v = cache[h//2, 0, :, :D].astype(np.float32), cache[h//2, 0, :, D:].astype(np.float32)
        logits = q[0, h].astype(np.float32) @ k.T / np.sqrt(D)
        logits = np.where(visible, logits, -np.inf)
        weights = np.exp(logits-logits.max(-1, keepdims=True))
        expected = weights / weights.sum(-1, keepdims=True) @ v
        with self.subTest(iteration=iteration, head=h):
          np.testing.assert_allclose(actual[0, h], expected, rtol=2e-3, atol=4e-4)


if __name__ == "__main__": unittest.main()
