import math, unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.qwen_vision import attn_pack

class TestVisionAttnPack(unittest.TestCase):
  def test_pack_is_masked_attention(self):
    # VISION_KERNELS: the fused kernel's inputs (head 72 -> 96, the spare column masking padded keys, q prescaled) give the masked float
    # attention when the kernel's own rule (scores / sqrt(96), softmax over all keys, not causal) runs on them: here on the CPU
    H, n, hd, real = 4, 64, 72, 50
    rng = np.random.default_rng(0)
    q, k, v = [Tensor(rng.standard_normal((H, n, hd)).astype(np.float32) * 2, device="CPU") for _ in range(3)]
    km = Tensor(np.concatenate([np.zeros(real), np.full(n - real, -1e30)]).astype(np.float32), device="CPU").reshape(1, 1, n)
    ref = q.scaled_dot_product_attention(k, v, attn_mask=km).numpy()
    qh, kv = attn_pack(q, k, v, km)
    D = qh.shape[-1]
    self.assertEqual((D, kv.shape), (96, (2, 1, H, n, 96)))
    o = ((qh.float() @ kv[0, 0].float().transpose(-1, -2)) / math.sqrt(D)).softmax(-1) @ kv[1, 0].float()
    o = o.numpy()
    self.assertEqual(np.abs(o[..., hd:]).max(), 0.0)
    a, b = o[..., :hd].reshape(-1, hd), ref.reshape(-1, hd)
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    self.assertGreater(cos.min(), 0.99999)  # (F16 rounding of q / k / v only)

if __name__ == "__main__":
  unittest.main()
