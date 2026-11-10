#!/usr/bin/env python3
# K256_SKIP (grouped K256 experts: a tile's all-padding 16-row token tiles skip their WMMAs): the same bits on every row that is read
import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.llm.kernels import amd_glm
from tinygrad.llm.kernels.amd_glm import moe_grouped_linear, QUANT_SIZES, IQ2_S, IQ3_S, amd_custom_kernels_supported

def run(typ:int, skip:int, perm:int=0, T=96, E=6, K=4, NE=12, D=512, F=256, act=True, sorted_out=False, tile=64) -> np.ndarray:
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
  assert any(0 < int((ent[i*tile:(i+1)*tile] >= 0).sum()) <= tile - 16 for i in range(R // tile)), "no tile with a padding token tile"
  amd_glm.K256_SKIP, amd_glm.K256_SIGNPERM = skip, perm
  amd_glm._grouped_k256_kernel.cache_clear()
  if sorted_out:
    x = Tensor(rng.normal(size=(R, D)).astype(np.float32)).realize()
    out = moe_grouped_linear(w, typ, E, x, Tensor(ent), Tensor(til), selections=len(sel), sorted_out=True).numpy()
  else:
    x = Tensor(rng.normal(size=(T, D)).astype(np.float32)).realize()
    out = moe_grouped_linear(w, typ, E, x, Tensor(ent), Tensor(til), slots=K, act=(1.0, 0.0, 0.0) if act else None).numpy()
  amd_glm.K256_SKIP, amd_glm.K256_SIGNPERM = 0, 0
  return out[ent >= 0]  # (the padding rows aren't read)

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "AMD custom kernels")
class TestK256Skip(unittest.TestCase):
  def test_iq2_s_act(self): np.testing.assert_array_equal(run(IQ2_S, 0), run(IQ2_S, 1))
  def test_iq2_s(self): np.testing.assert_array_equal(run(IQ2_S, 0, act=False), run(IQ2_S, 1, act=False))
  def test_iq2_s_act_perm(self): np.testing.assert_array_equal(run(IQ2_S, 0), run(IQ2_S, 1, perm=1))
  def test_iq2_s_down_sorted(self):
    np.testing.assert_array_equal(run(IQ2_S, 0, act=False, sorted_out=True), run(IQ2_S, 1, perm=1, act=False, sorted_out=True))
  def test_iq3_s_unchanged(self):
    np.testing.assert_array_equal(run(IQ3_S, 0, act=False, sorted_out=True), run(IQ3_S, 1, perm=1, act=False, sorted_out=True))

def _perm(a:np.ndarray, b:np.ndarray, sel:int) -> np.ndarray:
  # v_perm_b32: selector byte 0-3 a byte of b, 4-7 of a, 12 0x00, 13 0xff
  out = np.zeros_like(a)
  for k in range(4):
    s = (sel >> (8*k)) & 255
    byte = (b >> (8*s)) & 255 if s < 4 else (a >> (8*(s-4))) & 255 if s < 8 else np.full_like(a, 255 if s == 13 else 0)
    out |= byte << (8*k)
  return out

class TestSignPerm(unittest.TestCase):
  def test_every_grid_word(self):
    # _iq_signed_word perm (K256_SIGNPERM) == the mask form, for every IQ2_S/IQ2_XS/IQ3_S/IQ3_XXS grid word and sign nibble
    from tinygrad.runtime.autogen import ggml_common as g
    words = np.array(sorted({(v >> (32*i)) & 0xffffffff for grid, n in ((g.iq2s_grid, 2), (g.iq2xs_grid, 2), (g.iq3s_grid, 1), (g.iq3xxs_grid, 1))
                             for v in grid for i in range(n)}), dtype=np.int64)
    M = 0xffffffff
    for s in range(16):
      ones = (s * 0x00204081) & 0x01010101
      ref = ((words ^ _perm(words*0, words*0, (ones + 0x0C0C0C0C) & M)) + ones) & M
      new = _perm(((words ^ M) + 0x01010101) & M, words, ((s * 0x00810204) & 0x04040404) | 0x03020100)
      np.testing.assert_array_equal(ref, new)

if __name__ == "__main__":
  unittest.main()
