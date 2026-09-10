import functools, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo

@functools.cache
def custom_fp8_backward(*args:UOp, B:int, N:int, H:int, H_KV:int, arch:str):
  assert arch == "gfx950" and N % 64 == 0 and H % H_KV == 0
  source = (pathlib.Path(__file__).parent / "fa_fp8_bwd.cpp").read_text()
  output_bf16 = args[0].dtype == dtypes.bfloat16
  options = [f"-I{pathlib.Path(__file__).parent / 'include'}", "-std=c++20", "-DKITTENS_CDNA4",
             "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffp-contract=off", f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}"]
  lib = HIPCCCompiler(arch, options+[f"-DOUTPUT_BF16={int(output_bf16)}"]).compile_cached(source)
  owned_rows = min(N, 128)
  sink = UOp.sink(*(a.base for a in args), UOp.special(owned_rows*4,"lidx0"), UOp.special(N//owned_rows,"gidx0"),
                  UOp.special(H,"gidx1"), UOp.special(B,"gidx2"),arg=KernelInfo(name="hk_fa_fp8_backward", estimates=Estimates(ops=5*B*H*N*N*128)))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=(*sink.src,sink)),UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=lib)))

def unpack_dq(dq:Tensor):
  B,N,H,D = dq.shape
  return dq.reshape(B,H,N//16,8,2,4,16,2).permute(0,2,5,4,7,1,3,6).reshape(B,N,H,D)

def fp8_backward(q8:Tensor, k8:Tensor, v8:Tensor, v_descale:Tensor, do:Tensor, out:Tensor, lse:Tensor,
                 p_descale:Tensor, ds_descale:Tensor, next_amax:Tensor|None=None, *, native:bool=False):
  """FP8 backward with explicit delayed scales; accumulates next amax locally.

  Native outputs preserve the expanded BF16 layout consumed by fused RoPE.
  """
  from extra.llama_kernels import alloc_like, local_abs_max
  B,N,H,D = q8.shape
  assert D == 128 and q8.dtype == k8.dtype == v8.dtype == dtypes.fp8e4m3
  axis = q8.uop.axis if isinstance(q8.device,tuple) else None
  assert axis in (None,0), "FP8 backward currently supports data parallelism only"
  H_KV = k8.shape[2]
  assert k8.shape == v8.shape == (B,N,H_KV,D) and do.shape == out.shape == q8.shape
  def alloc(shape,dtype=dtypes.float32): return alloc_like(shape,dtype,q8.device,axis)
  do_scale = ((local_abs_max(do.float())+1e-8)/57344.).reshape(1).contiguous()
  do8 = (do.float()/do_scale).clamp(-57344,57344).cast(dtypes.fp8e5m2).contiguous()
  delta = (out.float()*(do8.float()*do_scale)).sum(-1).transpose(1,2).contiguous()
  # Before the first amax observation, bound dS from the current dO and V
  # ranges. A fixed unit amax would underflow small training gradients.
  ds_descale = (ds_descale > 0).where(ds_descale, 4*D*do_scale*v_descale*448.)
  scales = Tensor.cat(v_descale.reshape(1),do_scale,p_descale.reshape(1),ds_descale.reshape(1)).contiguous()
  output_dtype = dtypes.bfloat16 if native else dtypes.float32
  dq = alloc(q8.shape,output_dtype).zeros_like().contiguous()
  dk,dv = [alloc(q8.shape,output_dtype) for _ in range(2)]
  amax = alloc((B,H,N//64,2))
  if next_amax is None: next_amax = Tensor.zeros(2,device=q8.device,dtype=dtypes.float32).contiguous()
  dev = q8.device[0] if isinstance(q8.device,tuple) else q8.device
  local_b = B//len(q8.device) if axis == 0 else B
  ret = Tensor.custom_kernel(dq,dk,dv,amax,next_amax,q8,k8,v8,do8,delta,lse.contiguous(),scales,
    fxn=functools.partial(custom_fp8_backward,B=local_b,N=N,H=H,H_KV=H_KV,arch=Device[dev].renderer.target.arch))
  dq,dk,dv = ret[:3]
  dq = unpack_dq(dq)
  dk = dk.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  dv = dv.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  return dq,dk,dv,ret[3]
