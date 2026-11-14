# the V3.1 file's GDN q / k / v / gate are Q6_K: batched decoding's merged projection (merged_linear, q8_linear_parts) takes them as one Q6_K
# Linear; its gemm (n_out 4, no split sums) renders for RDNA3 (gfx1100) at one device's real shape (2560 -> 4 x 1024)
import functools, unittest
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.renderer import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel, QUANT_SIZES, Q6_K, Q8_0

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

if __name__ == "__main__": unittest.main()
