# Q5_0 (ggml 6) in amd_qwen: _quant_word's four int8 weight bytes against the gguf loader's dequantization (DEV=CPU: the kernel's index math runs)
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd_qwen import Q5_0, SMALL_BLOCK_QUANTS, QUANT_SIZES, HALFWORD_QUANTS, GROUP_SCALE_QUANTS, _quant_word

N = 96  # blocks of 32 weights, 22 bytes

def _blocks(seed:int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(N, 22), dtype=np.uint8)
  d = rng.uniform(-2, 2, size=N).astype(np.float16)
  raw[:, :2] = d.view(np.uint8).reshape(N, 2)
  raw[0, 2:6], raw[1, 2:6], raw[2, 2:6], raw[3, 2:22] = 0, 255, 255, 255  # qh all clear / all set, qs all set
  return raw

def _kernel_int(out:UOp, raw:UOp) -> UOp:  # the word index as a Python int (unrolled callers)
  b = UOp.range(N, 0)
  stores = [out[b, k].store(_quant_word(raw, b*11, UOp.const(0, dtypes.weakint), k, Q5_0, None)) for k in range(8)]
  return UOp.group(*stores).end(b).sink(arg=KernelInfo(name="q5_0_words_int", opts_to_apply=()))

def _kernel(out:UOp, raw:UOp) -> UOp:
  b, i = UOp.range(N, 0), UOp.range(8, 1)
  word = _quant_word(raw, b*11, UOp.const(0, dtypes.weakint), i, Q5_0, None)
  return out[b, i].store(word).end(b, i).sink(arg=KernelInfo(name="q5_0_words", opts_to_apply=()))

class TestQ5_0(unittest.TestCase):
  def test_tables(self):
    self.assertEqual(SMALL_BLOCK_QUANTS[Q5_0], (11, 32))
    self.assertEqual(QUANT_SIZES[Q5_0], 176)
    self.assertIn(Q5_0, HALFWORD_QUANTS)
    self.assertIn(Q5_0, GROUP_SCALE_QUANTS)

  def test_words_against_loader(self):
    for kernel in (_kernel, _kernel_int): self._check(kernel)

  def _check(self, kernel):
    raw = _blocks(0)
    ref = ggml_data_to_tensor(Tensor(raw.reshape(-1)), N*32, 6).reshape(N, 32).numpy()  # (q - 16) * d, float
    d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(N, 1)
    halves = Tensor(raw.reshape(-1)).bitcast(dtypes.uint16).contiguous()
    words = Tensor.custom_kernel(Tensor.empty(N, 8, dtype=dtypes.uint32), halves, fxn=kernel)[0].numpy()
    q = words.reshape(N, 8, 1).view(np.int8).reshape(N, 32).astype(np.float32)
    np.testing.assert_array_equal(q * d, ref.astype(np.float32))
    self.assertEqual(q.min(), -16)
    self.assertEqual(q.max(), 15)

if __name__ == "__main__": unittest.main()
