import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from extra.models.llama import precompute_freqs_cis
from extra.thunder.amd.fa import flash_attention, fused_qkv_rope

class TestASMFP8FA(unittest.TestCase):
  def test_fused_rope_forward_backward(self):
    if Device[Device.DEFAULT].renderer.target.arch != 'gfx950': self.skipTest('requires gfx950')
    Tensor.manual_seed(13)
    # Fused RoPE backward is specialized to microbatch two.
    B = 2
    N, H, H_KV, D = 8192, 32, 8, 128
    base = (Tensor.randn(B,N,6144)*0.12).bfloat16().contiguous().realize()
    freqs = precompute_freqs_cis(D,N*2).cast(dtypes.bfloat16).contiguous().realize()
    do = (Tensor.randn(B,N,H,D)*0.1).bfloat16().contiguous().realize()
    def run(fp8):
      x = base.detach().clone().contiguous().realize()
      q,k,v,*qk8 = fused_qkv_rope(x,freqs,H,H_KV,D,prequantize_fp8=fp8)
      with patch('extra.thunder.amd.fa.getenv', side_effect=lambda key, default=0: 1 if key=='ASM_FP8_FA' else getenv(key,default)):
        out,_,lse = flash_attention(q,k,v,is_causal=True,fp8_qk=fp8,
                                    q_fp8=qk8[0] if fp8 else None,k_fp8=qk8[1] if fp8 else None)
        out.backward(do)
        Tensor.realize(out,lse,x.grad)
      return out,lse,x.grad
    actual,reference = run(True),run(False)
    for name,a,b,atol,rms_tol in zip(('output','LSE','gradient'),actual,reference,(0.025,0.005,0.003),(0.0005,0.001,0.0001)):
      aa,bb=a.float().numpy(),b.float().numpy()
      self.assertTrue(np.isfinite(aa).all(), name)
      err = np.abs(aa-bb)
      print(name, "max_abs", float(err.max()), "rms", float(np.sqrt(np.mean(err**2))))
      np.testing.assert_allclose(aa,bb,atol=atol,rtol=0.06 if name=="gradient" else 0.0)
      self.assertLess(float(np.sqrt(np.mean(err**2))),rms_tol,name)

if __name__ == '__main__': unittest.main()
