"""DEV=AMD DEBUG=2 python -m extra.thunder.amd.bench_fa_fp8 (no ROCm runtime required)."""
import functools, time
import numpy as np
from tinygrad import Tensor, Device, Context, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv, GlobalCounters
from extra.thunder.amd.asm_fa_fp8 import flash_attention_fp8
from extra.thunder.amd.fa import custom_asm_fa_forward

def quantize(x:Tensor):
  descale = ((x.abs().max().float() + 1e-8) / 448).reshape(1).contiguous()
  return (x / descale).clamp(-448, 448).cast(dtypes.fp8e4m3).contiguous(), descale

def benchmark(fn, *args):
  jit = TinyJit(fn)
  for _ in range(3): jit(*args)
  Device[Device.DEFAULT].synchronize()
  gpu_start = GlobalCounters.time_sum_s
  st = time.perf_counter()
  for _ in range(20): jit(*args)
  Device[Device.DEFAULT].synchronize()
  return (time.perf_counter()-st)/20, (GlobalCounters.time_sum_s-gpu_start)/20

if __name__ == '__main__':
  assert getenv('DEBUG') >= 2, 'DEBUG>=2 is required for GPU timing'
  assert Device[Device.DEFAULT].renderer.target.arch == 'gfx950'
  B, N, H, H_KV, D = getenv('B', 2), 8192, 32, 8, 128
  Tensor.manual_seed(42)
  with Context(DEBUG=0):
    xs = [(Tensor.randn(B,N,h,D)*0.5).bfloat16().contiguous().realize() for h in (H,H_KV,H_KV)]
    qkv = [quantize(x) for x in xs]
    Tensor.realize(*(t for pair in qkv for t in pair))
    q,k,v = [p[0] for p in qkv]
    qs,ks,vs = [p[1] for p in qkv]
    # Compare against BF16 assembly applied to the same quantized physical Q/K/V.
    ref_inputs = [(p.float()*s).bfloat16().contiguous().realize() for p,s in qkv]
    o = Tensor.empty(B,N,H,D,dtype=dtypes.bfloat16)
    lse = Tensor.empty(B,H,1,N,dtype=dtypes.float32)
    ref, ref_lse = Tensor.custom_kernel(o,lse,*ref_inputs,
      fxn=functools.partial(custom_asm_fa_forward,B=B,N=N,H=H,H_KV=H_KV,D=D))[:2]
    Tensor.realize(ref,ref_lse)
  out, lse = flash_attention_fp8(q,k,v,qs,ks,vs)
  Tensor.realize(out,lse)
  with Context(DEBUG=0):
    for name,a,b in [('output',out,ref),('LSE',lse,ref_lse)]:
      aa,bb=a.float().numpy(),b.float().numpy()
      assert np.isfinite(aa).all()
      print(name,'max_abs',np.max(np.abs(aa-bb)),'rms',np.sqrt(np.mean((aa-bb)**2)))
      if name == "output":
        # Native FP8 also rounds softmax probabilities before PV. Bound both worst-case and RMS error.
        assert np.max(np.abs(aa-bb)) < 0.04 and np.sqrt(np.mean((aa-bb)**2)) < 0.001
      else: np.testing.assert_allclose(aa,bb,atol=0.015,rtol=0.003)
  def packed(q,k,v,qs,ks,vs): return Tensor.realize(*flash_attention_fp8(q,k,v,qs,ks,vs))
  def full(q,k,v):
    pairs=[quantize(x) for x in (q,k,v)]
    return Tensor.realize(*flash_attention_fp8(*(p[0] for p in pairs),*(p[1] for p in pairs)))
  for name,fn,args in [('prequantized',packed,(q,k,v,qs,ks,vs)),('quantize+FA',full,tuple(xs))]:
    host,gpu=benchmark(fn,*args)
    print(f'RESULT B={B} {name}: host {host*1e6:.2f} us, GPU {gpu*1e6:.2f} us, '
          f'{2*B*H*N*N*D/gpu/1e15:.3f} PFLOPS (causal useful ops; DEBUG>=2 required)')
