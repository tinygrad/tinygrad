import functools, pathlib, math
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo

@functools.cache
def custom_fp8_backward(*args:UOp, B:int, N:int, H:int, H_KV:int, arch:str):
  assert arch == "gfx950" and N % 64 == 0 and H % H_KV == 0
  source = (pathlib.Path(__file__).parent / "fa_fp8_bwd.cpp").read_text()
  options = [f"-I{pathlib.Path(__file__).parent / 'include'}", "-std=c++20", "-DKITTENS_CDNA4",
             "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffp-contract=off", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}"]
  lib = HIPCCCompiler(arch, options).compile_cached(source)
  sink = UOp.sink(*(a.base for a in args), UOp.special(64,"lidx0"), UOp.special(N//64,"gidx0"),
                  UOp.special(H,"gidx1"), UOp.special(B,"gidx2"),arg=KernelInfo(name="hk_fa_fp8_backward", estimates=Estimates(ops=5*B*H*N*N*128)))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=(*sink.src,sink)),UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=lib)))

@functools.cache
def cast_gradients(*args:UOp):
  idx = UOp.range(math.prod(args[0].shape), 0)
  stores = [args[i].flatten()[idx].store(args[i+3].flatten()[idx].cast(dtypes.bfloat16)) for i in range(3)]
  return UOp.group(*stores).end(idx).sink(arg=KernelInfo(name="fp8_fa_backward_cast", estimates=Estimates(mem=math.prod(args[0].shape)*18)))

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
  dq = alloc(q8.shape)
  dk,dv = [alloc(q8.shape).zeros_like().contiguous() for _ in range(2)]
  amax = alloc((B,H,N//64,2))
  if next_amax is None: next_amax = Tensor.zeros(2,device=q8.device,dtype=dtypes.float32).contiguous()
  dev = q8.device[0] if isinstance(q8.device,tuple) else q8.device
  local_b = B//len(q8.device) if axis == 0 else B
  ret = Tensor.custom_kernel(dq,dk,dv,amax,next_amax,q8,k8,v8,do8,delta,lse.contiguous(),scales,
    fxn=functools.partial(custom_fp8_backward,B=local_b,N=N,H=H,H_KV=H_KV,arch=Device[dev].renderer.target.arch))
  dq,dk,dv = ret[:3]
  if native:
    dq,dk,dv = Tensor.custom_kernel(*(alloc(q8.shape,dtypes.bfloat16) for _ in range(3)),dq,dk,dv,fxn=cast_gradients)[:3]
  dk = dk.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  dv = dv.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  return dq,dk,dv,ret[3]
