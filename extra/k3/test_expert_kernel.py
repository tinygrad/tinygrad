#!/usr/bin/env python3
# DEV=AMD:HIP python3 extra/k3/test_expert_kernel.py: the grouped expert gemv over cache slots matches dequantize + matmul
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import DEV
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import expert_linear, expert_gemm, expert_combine, q8_act, q8_rms_norm, _q8_quantize, _q8_cache, QUANT_SIZES, IQ3_XXS

LAT, UNITS, NSLOTS = 3584, 512, 20
ROW_UP, ROW_DOWN = LAT // 256 * QUANT_SIZES[IQ3_XXS], UNITS // 256 * QUANT_SIZES[IQ3_XXS]
PART = 2 * UNITS * ROW_UP + LAT * ROW_DOWN  # gate+up rows, then down columns: the layout extra/k3/pack.py writes

def random_iq3xxs(nbytes:int, rng) -> np.ndarray:
  raw = rng.integers(0, 256, nbytes, dtype=np.uint8)
  # sane fp16 block scales (the first 2 bytes of every 98 byte block)
  d = raw.reshape(-1, QUANT_SIZES[IQ3_XXS])
  d[:, :2] = np.frombuffer(np.full(len(d), 0.01, np.float16).tobytes(), np.uint8).reshape(-1, 2)
  return raw

class TestExpertLinear(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(0)
    self.raw = random_iq3xxs(NSLOTS * PART, rng)
    self.pool = Tensor(self.raw).bitcast(dtypes.uint16).realize()
    self.sel = [5, 0, 19, 7, 3, 11, 2, 18, 9, 1, 14, 6, 17, 4, 12, 8]
    self.slots = Tensor(np.array(self.sel, np.int32)).realize()
    self.rng = rng

  def weight(self, slot:int, off:int, rows:int, cols:int) -> np.ndarray:
    blob = Tensor(self.raw[slot * PART + off:slot * PART + off + rows * cols // 256 * QUANT_SIZES[IQ3_XXS]])
    return ggml_data_to_tensor(blob, rows * cols, IQ3_XXS).reshape(rows, cols).numpy()

  def test_gate_up_shared_input(self):
    x = self.rng.standard_normal((1, LAT)).astype(np.float32)
    out = expert_linear(self.pool, self.slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[0]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_down_per_expert_input(self):
    h = self.rng.standard_normal((len(self.sel), UNITS)).astype(np.float32)
    off = 2 * UNITS * ROW_UP
    out = expert_linear(self.pool, self.slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, off, LAT, UNITS) @ h[k]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_down_weighted_sum(self):  # with weights: the weighted sum of the slots' outputs, plus a bias
    h, w = self.rng.standard_normal((len(self.sel), UNITS)).astype(np.float32), self.rng.uniform(0, 1, len(self.sel)).astype(np.float32)
    off = 2 * UNITS * ROW_UP
    b = self.rng.standard_normal(LAT).astype(np.float32)
    out = expert_linear(self.pool, self.slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, w=Tensor(w), bias=Tensor(b)).numpy()
    ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(self.sel)) + b
    np.testing.assert_allclose(out[0], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_tokens(self):  # every token its own slots (entry k of token k % T): gate/up on its token's input, each token's weighted sum plus its bias
    T, off, sel = 2, 2 * UNITS * ROW_UP, self.sel[:12]
    slots = Tensor(np.array(sel, np.int32)).realize()
    x = self.rng.standard_normal((T, LAT)).astype(np.float32)
    gu = expert_linear(self.pool, slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    h, w = self.rng.standard_normal((len(sel), UNITS)).astype(np.float32), self.rng.uniform(0, 1, len(sel)).astype(np.float32)
    b = self.rng.standard_normal((T, LAT)).astype(np.float32)
    out = expert_linear(self.pool, slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, tokens=T, w=Tensor(w), bias=Tensor(b)).numpy()
    for k, s in enumerate(sel):
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[k % T]
      np.testing.assert_allclose(gu[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())
    for t in range(T):
      ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(sel) if k % T == t) + b[t]
      np.testing.assert_allclose(out[t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_padding(self):  # entries of weight 0, each token's last: gate/up skips them (rows not written), the sums are of the others
    T, off, sel = 2, 2 * UNITS * ROW_UP, self.sel[:12]
    w = self.rng.uniform(0.1, 1, len(sel)).astype(np.float32)
    w[6::2], w[9::2] = 0, 0  # (token 0 keeps its first 3, token 1 its first 4 of 6)
    slots, x = Tensor(np.array(sel, np.int32)).realize(), self.rng.standard_normal((T, LAT)).astype(np.float32)
    gu = expert_linear(self.pool, slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2, w=Tensor(w), summed=False).numpy()
    h = self.rng.standard_normal((len(sel), UNITS)).astype(np.float32)
    out = expert_linear(self.pool, slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, tokens=T, w=Tensor(w)).numpy()
    for k, s in enumerate(sel):
      if w[k] == 0: continue
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[k % T]
      np.testing.assert_allclose(gu[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())
    for t in range(T):
      ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(sel) if k % T == t)
      np.testing.assert_allclose(out[t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_shared(self):  # a slot for both tokens (weights a token each, 0: not that token's): read once, the same as an entry a token
    T, off, sel = 2, 2 * UNITS * ROW_UP, self.sel[:6]
    w = self.rng.uniform(0.1, 1, (len(sel), T)).astype(np.float32)
    w[3, 1], w[4, 0], w[5] = 0, 0, 0  # (3: token 0's alone, 4: token 1's, 5: padding)
    slots, x = Tensor(np.array(sel, np.int32)).realize(), self.rng.standard_normal((T, LAT)).astype(np.float32)
    gu = expert_linear(self.pool, slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2, tokens=T, w=Tensor(w.reshape(-1)), summed=False,
                       shared=True).numpy()
    h = self.rng.standard_normal((len(sel) * T, UNITS)).astype(np.float32)
    out = expert_linear(self.pool, slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, tokens=T, w=Tensor(w.reshape(-1)), shared=True).numpy()
    for u, s in enumerate(sel[:5]):
      for t in range(T):
        ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[t]
        np.testing.assert_allclose(gu[u * T + t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())
    for t in range(T):
      ref = sum(w[u, t] * self.weight(s, off, LAT, UNITS) @ h[u * T + t] for u, s in enumerate(sel))
      np.testing.assert_allclose(out[t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_q8_act(self):  # the act of a gate/up output's halves in its q8 kernel: the q8 of the act
    gu = Tensor(self.rng.standard_normal((len(self.sel), 2 * UNITS)).astype(np.float32) * 3)
    def act(g, u): return (g / 2).tanh() * 2 * g.sigmoid() * u
    y = q8_act(act, gu, parts=2)
    got, want = _q8_cache[y.uop], _q8_quantize(Tensor(y.numpy()), tokens=len(self.sel), in_features=UNITS)
    np.testing.assert_equal(got[0].numpy(), want[0].numpy())
    for g, w in zip(got[1:], want[1:]): np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-6)

  def test_q8_act_inputs(self):  # gate and up apart (the shared expert's): the same q8
    g, u = (Tensor(self.rng.standard_normal((2, UNITS)).astype(np.float32) * 3) for _ in range(2))
    def act(g, u): return (g / 2).tanh() * 2 * g.sigmoid() * u
    y = q8_act(act, g, u)
    got, want = _q8_cache[y.uop], _q8_quantize(Tensor(y.numpy()), tokens=2, in_features=UNITS)
    np.testing.assert_equal(got[0].numpy(), want[0].numpy())
    for a, b in zip(got[1:], want[1:]): np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6)

  def test_q8_rms_norm(self):  # the q8 of a norm from its input: as the norm's within a step of the q8 (the sums in another order)
    norm, x = nn.RMSNorm(LAT, 1e-6), Tensor(self.rng.standard_normal((1, 2, LAT)).astype(np.float32) * 3)
    norm.weight = Tensor(self.rng.uniform(0.5, 1.5, LAT).astype(np.float32))
    y = q8_rms_norm(norm, x)
    q, scale, _ = _q8_cache[y.uop]
    np.testing.assert_allclose(y.numpy(), norm(x).numpy())
    deq = q.numpy().view(np.int8).reshape(2, -1, 32) * scale.numpy()[..., None]
    np.testing.assert_array_less(np.abs(deq.reshape(2, LAT) - y.numpy()[0]), scale.numpy().repeat(32, -1) * 0.51)

  def test_pool_past_2g_elements(self):  # a slot past 2**31 words: 64 bit offsets
    far = 2 ** 31 // (PART // 2) + 2
    pool = Tensor.empty((far + 1) * PART // 2, dtype=dtypes.uint16).contiguous().realize()
    pool[far * PART // 2:].assign(Tensor(self.raw[5 * PART:6 * PART]).bitcast(dtypes.uint16)).realize()
    x = self.rng.standard_normal((1, LAT)).astype(np.float32)
    out = expert_linear(pool, Tensor(np.full(16, far, np.int32)), Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    ref = self.weight(5, 0, 2 * UNITS, LAT) @ x[0]
    for k in range(16): np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

class TestExpertGemm(unittest.TestCase):  # the prefill's grouped gemm: 16 row tiles of (row, slot) pairs, tiles past ntiles not computed
  def setUp(self):
    self.rng = rng = np.random.default_rng(1)
    self.raw = random_iq3xxs(NSLOTS * PART, rng)
    self.pool = Tensor(self.raw).bitcast(dtypes.uint16).realize()
    # tiles: a slot's run of rows (padded to 16 with a row of its): slot 7 five tiles (chunks 1-3 and 4-5 of 4 tile chunks), slot 0 two
    # across a chunk's end; 10 live of 12 (the rest garbage)
    self.tslot = np.array([3, 7, 7, 7, 7, 7, 19, 0, 0, 11, 5, 5], np.int32)
    self.ntiles = 10

  def weight(self, slot:int, off:int, rows:int, cols:int) -> np.ndarray:
    return TestExpertLinear.weight(self, slot, off, rows, cols)

  def check(self, out:np.ndarray, x:np.ndarray, rows:np.ndarray, off:int, o:int, i:int):
    for tile in range(self.ntiles):
      w = self.weight(int(self.tslot[tile]), off, o, i)
      for p in range(tile * 16, tile * 16 + 16):
        ref = w @ x[rows[p]].astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(out[p], ref, rtol=0.02, atol=0.02 * np.abs(ref).max())

  def test_gate_up_gather(self):
    T = 24
    x = self.rng.standard_normal((T, LAT)).astype(np.float32)
    rows = self.rng.integers(0, T, len(self.tslot) * 16).astype(np.int32)
    rows[len(self.tslot) * 16 - 16:] = 10**6  # (past ntiles: never read)
    out = expert_gemm(self.pool, Tensor(x), Tensor(self.tslot), Tensor([self.ntiles], dtype=dtypes.int32), 2 * UNITS, LAT, PART // 2,
                      rows=Tensor(rows)).numpy()
    self.check(out, x, rows, 0, 2 * UNITS, LAT)

  def test_down_in_pair_order(self):  # no gather: row p is x's
    off = 2 * UNITS * ROW_UP
    h = self.rng.standard_normal((len(self.tslot) * 16, UNITS)).astype(np.float32)
    out = expert_gemm(self.pool, Tensor(h), Tensor(self.tslot), Tensor([self.ntiles], dtype=dtypes.int32), LAT, UNITS, PART // 2,
                      off // 2, dtype=dtypes.half).numpy()
    self.check(out.astype(np.float32), h, np.arange(len(h)), off, LAT, UNITS)

  def test_ntiles_from_a_kernel(self):  # ntiles computed on the gpu (the host's count in the engine), half x
    x = self.rng.standard_normal((len(self.tslot) * 16, LAT)).astype(np.float16)
    ntiles = (Tensor([self.ntiles * 2], dtype=dtypes.int32) // 2).contiguous()
    out = expert_gemm(self.pool, Tensor(x), Tensor(self.tslot), ntiles, 2 * UNITS, LAT, PART // 2).numpy()
    self.check(out, x, np.arange(len(x)), 0, 2 * UNITS, LAT)

  @unittest.skipIf(DEV.interface.startswith("MOCK"), "4 GB pool")
  def test_pool_past_2g_words(self):  # a slot past 2**31 words: 64 bit offsets
    far = 2 ** 31 // (PART // 2) + 2
    pool = Tensor.empty((far + 1) * PART // 2, dtype=dtypes.uint16).contiguous().realize()
    pool[far * PART // 2:].assign(Tensor(self.raw[3 * PART:4 * PART]).bitcast(dtypes.uint16)).realize()
    x = self.rng.standard_normal((32, LAT)).astype(np.float32)
    out = expert_gemm(pool, Tensor(x), Tensor(np.full(2, far, np.int32)), Tensor([2], dtype=dtypes.int32), 2 * UNITS, LAT, PART // 2).numpy()
    w = self.weight(3, 0, 2 * UNITS, LAT)
    for p in range(32):
      ref = w @ x[p].astype(np.float16).astype(np.float32)
      np.testing.assert_allclose(out[p], ref, rtol=0.02, atol=0.02 * np.abs(ref).max())

  def test_combine(self):  # a token's weighted sum of its K rows of the down gemm's output, in order
    T, K, P = 5, 16, 300
    y = self.rng.standard_normal((P, LAT)).astype(np.float32)
    dest, w = self.rng.integers(0, P, T * K).astype(np.int32), self.rng.uniform(0, 1, T * K).astype(np.float32)
    got = expert_combine(Tensor(y), Tensor(dest), Tensor(w), T, K).numpy()
    want = np.stack([sum(w[t * K + k] * y[dest[t * K + k]] for k in range(K)) for t in range(T)])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
  unittest.main()
