import unittest
from tinygrad import Tensor, dtypes, Context
from tinygrad.device import is_dtype_supported
from extra.llama_kernels.fused_ce import fused_ce_loss

def run_fused_ce(bs:int, seqlen:int, vocab:int, label_smoothing:float=0.0) -> None:
  Tensor.manual_seed(0)
  logits_rand = Tensor.randn(bs, seqlen, vocab).cast(dtypes.bfloat16)
  targets = Tensor.randint(bs, seqlen, high=vocab, dtype=dtypes.int32)
  logits, logits_ref = logits_rand.clone(), logits_rand.detach().float().contiguous()
  with Context(DEBUG=0):
    Tensor.realize(logits, logits_ref, targets)

  loss = fused_ce_loss(logits, targets, label_smoothing=label_smoothing)
  loss.backward()
  Tensor.realize(loss, logits.grad)

  ref = logits_ref.sparse_categorical_crossentropy(targets, label_smoothing=label_smoothing)
  ref.backward()
  Tensor.realize(ref, logits_ref.grad)

  assert logits.grad.shape == (bs, seqlen, vocab)
  with Context(DEBUG=0):
    assert loss.allclose(ref, atol=2e-3, rtol=2e-3).item(), "forward mismatch"
    assert logits.grad.allclose(logits_ref.grad, atol=2e-3, rtol=2e-3).item(), "grad mismatch"

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "need bfloat16")
class TestFusedCE(unittest.TestCase):
  def test_fused_ce_1_2_16(self): run_fused_ce(1, 2, 16, label_smoothing=0.2)
  def test_fused_ce_2_16_128(self): run_fused_ce(2, 16, 128)
  def test_fused_ce_4_128_1024(self): run_fused_ce(4, 128, 1024, label_smoothing=0.2)

  # note: this is the shape used in llama 8b
  #def test_fused_ce_smoothing_16_1024_128256(self): run_fused_ce(16, 1024, 128256, label_smoothing=0.2)

if __name__ == '__main__':
  unittest.main()
