import functools, math, os, unittest
from collections import Counter
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.function import function
from tinygrad.uop.ops import Ops
from extra.models.llama import precompute_freqs_cis
from extra.thunder.amd.fa import custom_fp8_fa_backward_inputs, flash_attention, fused_qkv_rope

class TestASMFP8FA(unittest.TestCase):
  def test_backward_input_conversion(self):
    if Device[Device.DEFAULT].renderer.target.arch != 'gfx950': self.skipTest('requires gfx950')
    Tensor.manual_seed(14)
    B,N,H,H_KV,D = 2,32,8,2,128
    shapes = ((B,N,H,D),(B,N,H_KV,D),(B,N,H_KV,D))
    inputs = [(Tensor.randn(*s)*10).cast(dtypes.fp8e4m3).contiguous().realize() for s in shapes]
    scale = Tensor([0.0137],dtype=dtypes.float32).realize()
    outputs = Tensor.custom_kernel(*[Tensor.invalids(*s,dtype=dtypes.bfloat16) for s in shapes], *inputs, scale,
      fxn=functools.partial(custom_fp8_fa_backward_inputs,B=B,N=N,H=H,H_KV=H_KV,D=D))[:3]
    expected = (inputs[0].bfloat16(), inputs[1].bfloat16(), (inputs[2].float()*scale).bfloat16())
    for actual,reference in zip(outputs,expected):
      np.testing.assert_array_equal(actual.float().numpy(),reference.float().numpy())

  @patch.dict(os.environ, {"DEVICE_IN_FUNCTION_BUG":"1", "ASM_FP8_FA":"1"})
  def test_saved_operands_do_not_recompute_rope(self):
    device = Device.DEFAULT
    if Device[device].renderer.target.arch != 'gfx950': self.skipTest('requires gfx950')
    devices = (device, f"{device.split(':')[0]}:1")
    x = Tensor.empty(4,8192,6144,device=device,dtype=dtypes.bfloat16).shard(devices,axis=0)
    freqs = Tensor.empty(1,16384,1,64,2,device=device,dtype=dtypes.bfloat16).shard(devices,axis=None)
    do = Tensor.empty(4,8192,32,128,device=device,dtype=dtypes.bfloat16).shard(devices,axis=0)
    @function(precompile=True, precompile_backward=True)
    def layer(x, freqs):
      q,k,v,q8,k8 = fused_qkv_rope(x,freqs,32,8,128,prequantize_fp8=True)
      out,*saves = flash_attention(q,k,v,is_causal=True,fp8_qk=True,q_fp8=q8,k_fp8=k8,save_fp8=True)
      return out+0.1,*saves
    out,*_ = layer(x,freqs)
    out.backward(do)
    counts = Counter()
    calls = {}
    def count_calls(u):
      if u.op is Ops.CALL:
        kernel = u.src[0].src[0] if u.src[0].op is Ops.PROGRAM else u.src[0]
        if kernel.op is Ops.SINK: calls[kernel.arg.name] = u
        count_calls(u.src[0])
      elif u.op is Ops.LINEAR:
        for s in u.src: count_calls(s)
      elif u.op is Ops.SINK: counts[u.arg.name] += 1
      elif u.op is Ops.PROGRAM: counts[u.src[0].arg.name] += 1
    count_calls(out.schedule_linear(x.grad))
    self.assertEqual(counts["fused_qkv_rope_forward"],1)
    self.assertEqual(counts["asm_fa_fwd_fp8_causal_2_8192_32_8_128"],1)
    self.assertEqual(counts["asm_fa_bwd_main_fp8_matched_causal_2_8192_32_8_128"],1)
    # The BF16 autograd argument must reuse RoPE's V buffer, with no identity copy.
    self.assertIs(calls["fused_qkv_rope_forward"].src[3].buf_uop,
                  calls["asm_fa_fwd_fp8_causal_2_8192_32_8_128"].src[7].buf_uop)

  def test_large_scores_use_forward_quantized_operands(self):
    if Device[Device.DEFAULT].renderer.target.arch != 'gfx950': self.skipTest('requires gfx950')
    B, N, H, H_KV, D = 2, 8192, 32, 8, 128
    q = Tensor.full((B,N,H,D),8,dtype=dtypes.bfloat16).contiguous().realize()
    k = Tensor.full((B,N,H_KV,D),8,dtype=dtypes.bfloat16).contiguous().realize()
    v = Tensor.ones(B,N,H_KV,D,dtype=dtypes.bfloat16).contiguous().realize()
    scale = math.sqrt(D**-0.5 * math.log2(math.e))
    q8 = (q.float()*scale).cast(dtypes.fp8e4m3).contiguous().realize()
    k8 = (k.float()*scale).cast(dtypes.fp8e4m3).contiguous().realize()
    do = Tensor.cat(Tensor.ones(B,1,H,D),Tensor.zeros(B,N-1,H,D),dim=1).bfloat16().contiguous().realize()
    with patch('extra.thunder.amd.fa.getenv', side_effect=lambda key, default=0: 1 if key=='ASM_FP8_FA' else getenv(key,default)):
      out,_,lse = flash_attention(q,k,v,is_causal=True,fp8_qk=True,q_fp8=q8,k_fp8=k8)
      out.backward(do)
      Tensor.realize(out,lse,q.grad,k.grad,v.grad)
    # Only the first causal query receives dO. Its single probability must be 1:
    # dQ=dK=0 and four query heads contribute dV=4 to the first KV token.
    # Using unrounded Q/K with FP8 LSE instead gives dV about 4.65e23.
    np.testing.assert_array_equal(q.grad.float().numpy(),0)
    np.testing.assert_array_equal(k.grad.float().numpy(),0)
    dv = v.grad.float().numpy()
    np.testing.assert_allclose(dv[:,0],H/H_KV,atol=0.01,rtol=0)
    np.testing.assert_array_equal(dv[:,1:],0)

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
