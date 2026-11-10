#!/usr/bin/env python3
# K256_XH (RDNA3 grouped K256 experts: lane halves decode one K half each, sub-blocks unrolled): the same bits as the default kernel
import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm.kernels.amd_qwen import moe_grouped_linear, QUANT_SIZES, IQ3_S, IQ4_XS, amd_custom_kernels_supported

def run(typ:int, xh:int, T=96, E=6, K=4, NE=12, D=512, F=256, act=True, tile=32, unroll=0) -> np.ndarray:
  rng = np.random.default_rng(1)
  O = 2*F if act else F
  b = rng.integers(0, 256, size=E*O*D//256*QUANT_SIZES[typ], dtype=np.uint8)
  b.reshape(-1, QUANT_SIZES[typ])[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
  w = Tensor(b.view(np.uint16)).realize()
  sel = np.argsort(rng.random((T, NE)), axis=-1)[:, :K]
  sel[-7:] = -1  # prefill's padding tokens select no expert
  sel = sel.reshape(-1)
  R = -(-(len(sel) + (tile-1)*E) // tile) * tile
  ent, til, at = np.full(R, -1, np.int32), np.full(R // tile, -1, np.int32), 0
  for e in range(E):
    idx = np.nonzero(sel == e)[0]
    ent[at:at+len(idx)], til[at//tile:at//tile + -(-len(idx)//tile)] = idx, e
    at += -(-len(idx)//tile) * tile
  x = Tensor(rng.normal(size=(T, D)).astype(np.float32)).realize()
  amd_qwen.K256_XH, amd_qwen.K256_UNROLL = xh, unroll
  amd_qwen._grouped_k256_kernel.cache_clear()
  out = moe_grouped_linear(w, typ, E, x, Tensor(ent), Tensor(til), slots=K, act=(1.0, 0.0, 0.0) if act else None).numpy()
  # (the rows of the selections in selection order: tiles of other sizes sort them alike)
  return out[ent >= 0][np.argsort(ent[ent >= 0])] if out.shape[0] == R else out

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "AMD custom kernels")
class TestK256XH(unittest.TestCase):
  def test_iq3_s_act(self): np.testing.assert_array_equal(run(IQ3_S, 0), run(IQ3_S, 1))
  def test_iq3_s(self): np.testing.assert_array_equal(run(IQ3_S, 0, act=False), run(IQ3_S, 1, act=False))
  def test_iq3_s_tile16(self): np.testing.assert_array_equal(run(IQ3_S, 0), run(IQ3_S, 1, tile=16))
  def test_iq3_s_unroll_chunks(self):
    ref = run(IQ3_S, 0)
    for stages in (2, 4):
      with self.subTest(stages=stages): np.testing.assert_array_equal(ref, run(IQ3_S, 0, unroll=stages))
  def test_iq3_s_unroll_chunks_tile16(self):
    ref = run(IQ3_S, 0)
    for stages in (2, 4):
      with self.subTest(stages=stages): np.testing.assert_array_equal(ref, run(IQ3_S, 0, tile=16, unroll=stages))
  def test_iq3_s_xh_unroll_chunks(self):
    ref = run(IQ3_S, 0)
    for stages in (2, 4):
      with self.subTest(stages=stages): np.testing.assert_array_equal(ref, run(IQ3_S, 1, tile=16, unroll=stages))
  def test_iq4_xs(self): np.testing.assert_array_equal(run(IQ4_XS, 0), run(IQ4_XS, 1))  # (not taken: IQ3_S only)

if __name__ == "__main__":
  unittest.main()
