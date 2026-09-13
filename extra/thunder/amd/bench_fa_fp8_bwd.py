"""Run the full LLaMA backward kernels directly, without JIT or pytest.

DEV=AMD DEBUG=2 PYTHONPATH=. ROCM_PATH=/opt/rocm-7.1.1 python extra/thunder/amd/bench_fa_fp8_bwd.py
"""
import functools, math, os
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv
from tinygrad.engine.realize import lower_and_compile, run_linear
from extra.thunder.amd.fa import flash_attention, custom_asm_fa_backward, custom_asm_fa_backward_shuffle
from extra.thunder.amd.fa_fp8_bwd import custom_fp8_backward, unpack_dq

def main():
  with Context(DEBUG=0):
    # Select FP8 forward only for input preparation; backward is launched explicitly below.
    os.environ.update(FP8_FA="1", ASM_FP8_FA="1", FP8_FA_BWD="0")
    getenv.cache_clear()
    dev = Device[Device.DEFAULT]
    assert dev.renderer.target.arch == "gfx950", "requires DEV=AMD on gfx950"
    B,N,H,HK,D = 2,8192,32,8,128
    shape = (B,N,H,D)
    Tensor.manual_seed(getenv("SEED",13))
    q,k,v = [Tensor.randn(*s).bfloat16().contiguous().realize() for s in [shape,(B,N,HK,D),(B,N,HK,D)]]
    do = (Tensor.randn(*shape)*0.1).bfloat16().contiguous().realize()
    c = math.sqrt(D**-0.5*math.log2(math.e))
    q8,k8 = [(x.float()*c).cast(dtypes.fp8e4m3).detach().contiguous().realize() for x in (q,k)]
    out,_,lse,*saved = flash_attention(q,k,v,is_causal=True,fp8_qk=True,q_fp8=q8,k_fp8=k8,save_fp8=True)
    v8,vs = saved[-2:]
    Tensor.realize(out,lse,v8,vs)
    out = out.reshape(shape)
    dos = ((do.float().abs().max()+1e-8)/57344).reshape(1).contiguous()
    do8 = (do.float()/dos).clamp(-57344,57344).cast(dtypes.fp8e5m2).contiguous()
    delta = (out.float()*(do8.float()*dos)).sum(-1).transpose(1,2).contiguous()
    scales = Tensor.cat(vs.reshape(1),dos,Tensor([1/448]),Tensor([1e-4])).contiguous()
    # Initialize atomic output buffers before the measured backward launch.
    fp8 = [Tensor.empty(*shape,dtype=dtypes.bfloat16) for _ in range(3)]
    fp8 += [Tensor.empty(B,H,N//64,2,dtype=dtypes.float32),Tensor.zeros(2,dtype=dtypes.float32),
            q8,k8,v8,do8,delta,lse.contiguous(),scales]
    # BF16 reference uses the same rounded operands, with the existing physical gradient scale.
    ref_do = (do8.float()*dos).bfloat16().contiguous()
    bf16 = [Tensor.zeros(B,H,N,D,dtype=dtypes.bfloat16).contiguous()]
    bf16 += [Tensor.empty(*shape,dtype=dtypes.bfloat16) for _ in range(2)]
    bf16 += [q8.bfloat16().contiguous(),k8.bfloat16().contiguous(),(v8.float()*vs).bfloat16().contiguous(),ref_do,
             lse.contiguous(),(out.float()*ref_do.float()).sum(-1).transpose(1,2).contiguous()]
    Tensor.realize(*fp8,*bf16)
    for t in fp8[:5]: t.assign(0).realize()
  bf16 = Tensor.custom_kernel(*bf16,
    fxn=functools.partial(custom_asm_fa_backward,B=B,N=N,H=H,H_KV=HK,D=D,pre_scaled_fp8=True))
  with Context(DEBUG=0): bf16_schedule = lower_and_compile(Tensor.schedule_linear(*bf16[:3]))
  run_linear(bf16_schedule)
  fp8 = Tensor.custom_kernel(*fp8,
    fxn=functools.partial(custom_fp8_backward,B=B,N=N,H=H,H_KV=HK,arch=dev.renderer.target.arch))
  with Context(DEBUG=0): fp8_schedule = lower_and_compile(Tensor.schedule_linear(*fp8[:5]))
  run_linear(fp8_schedule)

  with Context(DEBUG=0):
    fp8[0] = unpack_dq(fp8[0])
    ref_q = Tensor.custom_kernel(Tensor.empty(*shape,dtype=dtypes.bfloat16),bf16[0],
      fxn=functools.partial(custom_asm_fa_backward_shuffle,B=B,N=N,H=H,D=D))[0].realize()
    for name,actual,expected in zip(("dQ","dK","dV"),fp8[:3],(ref_q,*bf16[1:3])):
      a,b = actual.float(),expected.float()
      error = ((a-b).square().sum()/b.square().sum()).sqrt().item()
      assert math.isfinite(error) and error < 0.1, f"{name} failed: relative L2 error {error}"

if __name__ == "__main__":
  main()
