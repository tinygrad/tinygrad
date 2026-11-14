import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, UOp, dtypes
from tinygrad.uop.ops import KernelInfo
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm import qwen4exp

IQ4_XS, D = 23, 512

def table(rows:int, seed:int=0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(rows * D // 256 * 136,), dtype=np.uint8)
  for b in range(rows * D // 256): raw[b*136:b*136+2] = np.array([rng.uniform(0.001, 0.05)], dtype=np.float16).view(np.uint8)  # sane f16 scales
  return raw

class TestIQ4XSEmbed(unittest.TestCase):
  def test_value_matches_the_loader(self):
    # the arithmetic of the rows kernel (_iq4xs_value) run on the CPU (loops for the GPU's dims) against the gguf loader's dequant, then half
    raw, ids = table(37), np.array([0, 36, 5, 5, 17, 2], dtype=np.int32)
    ref = ggml_data_to_tensor(Tensor(raw, device="CPU"), 37 * D, IQ4_XS).reshape(37, D).cast("float16").float().numpy()
    def kern(out:UOp, raw:UOp, ids:UOp) -> UOp:
      r, blk, ib, lane = UOp.range(out.shape[0], 0), UOp.range(D//256, 1), UOp.range(8, 2), UOp.range(32, 3)
      base = ids[r].load().cast(dtypes.weakint)*(D//256*136) + blk*136
      return out[r, blk*256+ib*32+lane].store(amd_qwen._iq4xs_value(raw, base, ib, lane)).end(r, blk, ib, lane).sink(
        arg=KernelInfo(name="t", opts_to_apply=()))
    out = Tensor.custom_kernel(Tensor.empty(len(ids), D, device="CPU"), Tensor(raw, device="CPU"), Tensor(ids, device="CPU"), fxn=kern)[0].numpy()
    np.testing.assert_array_equal(out, ref[ids])

  def test_kernel_builds(self):
    # (the GPU kernel's graph lowers with the HIP renderer only: DEV=NULL:HIP:gfx1100)
    out = amd_qwen.iq4xs_embed(Tensor.empty(37 * D // 256 * 136, dtype=dtypes.uint8), Tensor.empty(6, dtype=dtypes.int32), D)
    self.assertEqual(out.shape, (6, D))
    if Device.DEFAULT.startswith("NULL:HIP"): out.realize()

  def test_embed_dispatch(self):
    # _embed reads an IQ4_XS table by its rows (not the generic dequant of the whole table); a Q8_0 table keeps its kernel
    class M:
      config = type("C", (), {"dim": D})()
      _embd_q8, _embd_iq4xs = None, Tensor.empty(10)
      token_embd = None
    toks = Tensor.empty(2, 3, dtype=dtypes.int32)
    with patch.object(qwen4exp, "amd_custom_kernels_supported", lambda d: True), \
         patch.object(qwen4exp, "iq4xs_embed", lambda raw, ids, dim: Tensor.empty(int(ids.shape[0]), dim)), \
         patch.object(qwen4exp, "q8_embed", side_effect=AssertionError("q8 path")):
      out = qwen4exp.Transformer._embed(M(), toks)  # type: ignore[arg-type]
    self.assertEqual(out.shape, (2, 3, D))

if __name__ == "__main__":
  unittest.main()
