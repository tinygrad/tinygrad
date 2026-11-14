# the V3.1 file's speed paths: the merged Q6_K projection (decode), the Q6_K gemm tiles (Q6K_TILES), IQ3_XXS experts in K256 (K256_IQ3XXS)
# the V3.1 file's GDN q / k / v / gate are Q6_K: batched decoding's merged projection (merged_linear, q8_linear_parts) takes them as one Q6_K
# Linear; its gemm (n_out 4, no split sums) renders for RDNA3 (gfx1100) at one device's real shape (2560 -> 4 x 1024)
import functools, unittest
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.renderer import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel, q6k_tile, Q6K_TILE_ORDER, QUANT_SIZES, Q6_K, Q8_0

class TestMergedQ6K(unittest.TestCase):
  def test_gemm_n_out_renders(self):
    ren = HIPRenderer(Target("AMD", arch="gfx1100"))
    K, N, n = 2560, 1024, 4
    for typ in (Q8_0, Q6_K):
      for tokens, tt in ((24, 32), (48, 32), (96, 96)):
        with self.subTest(ggml_type=typ, tokens=tokens):
          raw = Tensor.empty(n*N*K//256*QUANT_SIZES[typ]//2, dtype=dtypes.uint16)
          outs = [Tensor.empty(tokens, N) for _ in range(n)]
          fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=typ, block_n=64, k_splits=1, rdna4=False, n_out=n,
                                  out_features=n*N, in_features=K)
          res = Tensor.custom_kernel(*outs, raw, Tensor.empty(tokens, K, dtype=dtypes.float16), fxn=fxn)[0]
          ast = res.schedule_linear().src[-1].src[0]
          self.assertGreater(len(to_program(ast, ren).src[1].src), 100)

class TestQ6KTiles(unittest.TestCase):
  # Q6K_TILES: the V3.1 file's Q6_K projections of a prefill chunk (one device's outputs) in tiles that give the GPU enough workgroups
  def test_choice(self):
    self.assertEqual(q6k_tile(1152, 128), (32, 64))     # the shared expert's gate / up: 36 workgroups at 32 x 64, not 12 at 192 x 64
    self.assertEqual(q6k_tile(1152, 1024), (96, 64))    # the gated deltanets' q / k / v / gate
    self.assertEqual(q6k_tile(2304, 1024), (96, 128))
    self.assertEqual(q6k_tile(512, 1024), (64, 64))
    self.assertIsNone(q6k_tile(48, 1024))               # (no tile of the order divides the rows: the rule before)
  def test_every_tile_renders(self):
    ren = HIPRenderer(Target("AMD", arch="gfx1100"))
    K, N, M = 2560, 256, 384
    for tt, bn in Q6K_TILE_ORDER:
      with self.subTest(tile=tt, block_n=bn):
        raw = Tensor.empty(N*K//256*QUANT_SIZES[Q6_K]//2, dtype=dtypes.uint16)
        fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=Q6_K, block_n=bn, k_splits=1, rdna4=False, out_features=N,
                                in_features=K)
        res = Tensor.custom_kernel(Tensor.empty(M, N), raw, Tensor.empty(M, K, dtype=dtypes.float16), fxn=fxn)[0]
        ast = res.schedule_linear().src[-1].src[0]
        self.assertGreater(len(to_program(ast, ren).src[1].src), 100)

class TestK256IQ3XXS(unittest.TestCase):
  # K256_IQ3XXS: the V3.1 file's IQ3_XXS gate|up experts in the K256 kernel (with XH): renders for gfx1100, dispatched only with the flag
  def test_renders(self):
    from tinygrad.llm.kernels import amd_qwen
    from tinygrad.llm.kernels.amd_qwen import moe_grouped_linear, IQ3_XXS
    ren = HIPRenderer(Target("AMD", arch="gfx1100"))
    E, D, F, T, K, tile = 4, 512, 256, 64, 2, 32
    names = {}
    for flag in (0, 1):
      amd_qwen.K256_IQ3XXS = flag
      try:
        w = Tensor.empty(E*2*F*D//256*QUANT_SIZES[IQ3_XXS]//2, dtype=dtypes.uint16)
        R = T*K + (tile-1)*E
        R += -R % tile
        ent, til = Tensor.empty(R, dtype=dtypes.int32), Tensor.empty(R//tile, dtype=dtypes.int32)
        out = moe_grouped_linear(w, IQ3_XXS, E, Tensor.empty(T, D), ent, til, slots=K, act=(1.0, 0.0, 0.0))
        asts = [u.src[0] for u in out.schedule_linear().src if u.src and "grouped" in str(getattr(u.src[0].arg, "name", ""))]
        names[flag] = [a.arg.name for a in asts]
        if flag: self.assertGreater(len(to_program(asts[0], ren).src[1].src), 100)
      finally: amd_qwen.K256_IQ3XXS = 0
    self.assertEqual(names[1], ["grouped_iq3_xxs_k256"])
    self.assertNotIn("grouped_iq3_xxs_k256", names[0])

if __name__ == "__main__": unittest.main()
