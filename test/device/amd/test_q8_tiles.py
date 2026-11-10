#!/usr/bin/env python3
# Q8_TILES: quantized grouped-expert rows; default bits, tile16 bits per selection
import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm.kernels.amd_qwen import moe_grouped_linear, IQ4_NL, amd_custom_kernels_supported

def run(q8t:int, tile:int, T=96, E=6, K=4, NE=12, D=512, F=640) -> np.ndarray:
  rng = np.random.default_rng(2)
  b = rng.integers(0, 256, size=E*D*F//32*18, dtype=np.uint8)
  b.reshape(-1, 18)[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
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
  # (the down's input: a row per sorted row, the selection's row of x)
  x = rng.normal(size=(T*K, F)).astype(np.float16)
  amd_qwen.Q8_TILES = q8t
  amd_qwen._quant_linear_i8_wmma_kernel.cache_clear()
  out = moe_grouped_linear(w, IQ4_NL, E, Tensor(x[np.maximum(ent, 0)]), Tensor(ent), Tensor(til), selections=T*K, sorted_out=True).numpy()
  return out[ent >= 0][np.argsort(ent[ent >= 0])]

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "AMD custom kernels")
class TestQ8Tiles(unittest.TestCase):
  def test_q8_tiles(self): np.testing.assert_array_equal(run(0, 16), run(1, 16))
  def test_tile16_vs_32(self): np.testing.assert_array_equal(run(0, 32), run(1, 16))

if __name__ == "__main__":
  unittest.main()
