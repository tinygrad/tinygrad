# Qwen3.8-Flash-Next UD-IQ4_XS V3.1: the decode kernel of every expert type of the file renders for RDNA3 (gfx1100) at the file's real shapes
import functools, unittest
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.renderer import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.llm.kernels.amd_qwen import (_quant_decode_kernel, _iq_grid, q8_quantize, q8_quantize_cache_clear, QUANT_SIZES,
                                           SMALL_BLOCK_QUANTS, IQ2_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q5_0, Q6_K, Q8_0)

# (type, in_features, out_features): gate / up 2560 -> 640, down 640 -> 2560 (experts: one expert's matrix), shared experts, attention
CASES = ((IQ3_XXS, 2560, 640), (IQ4_XS, 2560, 640), (IQ2_S, 2560, 640), (Q5_0, 640, 2560), (IQ4_NL, 640, 2560), (Q2_0, 640, 2560),
         (Q6_K, 2560, 640), (Q8_0, 640, 2560))

class TestV31Types(unittest.TestCase):
  def test_decode_kernel_renders(self):
    ren = HIPRenderer(Target("AMD", arch="gfx1100"))
    for typ, K, N in CASES:
      with self.subTest(ggml_type=typ):
        words = N*K//32*SMALL_BLOCK_QUANTS[typ][0] if typ in SMALL_BLOCK_QUANTS else N*K//256*QUANT_SIZES[typ]//2
        raw = Tensor.empty(words, dtype=dtypes.uint16)
        xq = q8_quantize(Tensor.empty(4, K), 4, K)
        grids = (_iq_grid(raw.device, typ),) if typ in (IQ2_S, IQ3_XXS) else ()
        fxn = functools.partial(_quant_decode_kernel, out_features=N, in_features=K, ggml_type=typ)
        ast = Tensor.custom_kernel(Tensor.empty(4, N), raw, *xq, *grids, fxn=fxn)[0].schedule_linear().src[-1].src[0]
        self.assertGreater(len(to_program(ast, ren).src[1].src), 100)
        q8_quantize_cache_clear()

if __name__ == "__main__": unittest.main()
