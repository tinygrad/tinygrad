import functools
from tinygrad import Tensor, Device, dtypes
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.codegen.opt import Opt, OptOps

@functools.cache
def custom_fp8_backward(*args:UOp, B:int, N:int, H:int, H_KV:int, arch:str):
  assert arch == "gfx950" and (B,N,H,H_KV) == (2,8192,32,8) and args[0].dtype == dtypes.bfloat16
  from extra.thunder.amd.asm_fa_fp8_bwd import build_kernel
  from tinygrad.dtype import AddrSpace
  lds = UOp.placeholder((132160,), dtypes.uint8, 0, addrspace=AddrSpace.LOCAL)
  sink = UOp.sink(*(a.base for a in args), lds, UOp.special(512,"lidx0"), UOp.special(N//256,"gidx0"),
                  UOp.special(H,"gidx1"), UOp.special(B,"gidx2"),
                  arg=KernelInfo(name="hk_fa_fp8_backward", estimates=Estimates(ops=5*B*H*N*N*128)))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(UOp(Ops.INS,arg=(inst,dtypes.void)) for inst in build_kernel(B,N,H,H_KV)))))

def unpack_dq(dq:Tensor):
  B,N,H,D = dq.shape
  return dq.reshape(B,H,N//16,8,2,4,16,2).permute(0,2,5,4,7,1,3,6).reshape(B,N,H,D)

@functools.cache
def custom_fp8_backward_init(dq:UOp, partial:UOp, do:UOp):
  size, groups = do.numel(), partial.numel()
  assert size % groups == 0
  g = UOp.range(groups, 0)
  r = UOp.range(size//groups, 1, AxisType.REDUCE)
  idx = g*(size//groups)+r
  do = do.after(dq.flatten()[idx].store(0.))
  value = do.flatten()[idx].cast(dtypes.float).abs().reduce(r,arg=Ops.MAX)
  return partial.flatten()[g].store(value).end(g).sink(arg=KernelInfo("fa_fp8_bwd_init"))

@functools.cache
def custom_fp8_backward_scales(scales:UOp, next_amax:UOp, partial:UOp, state:UOp, vs:UOp, *, D:int):
  assert scales.numel() == 5
  # Finalize once, rather than repeating the partial-amax reduction for every attention row in prep.
  r = UOp.range(partial.numel(),0,AxisType.REDUCE)
  numerator = partial.flatten()[r].reduce(r,arg=Ops.MAX)+1e-8
  scale = numerator/57344.
  pd,sd = (state[0]+1e-8)/448.,state[1]/57344.
  sd = (sd>0).where(sd,4*D*scale*vs[0]*448.)
  stores = [scales[i].store(v) for i,v in enumerate((vs[0],scale,pd,sd))]
  # Preserve the original prep's floating-point operation order before the division by 57344.
  stores.append(scales[4].store(numerator))
  stores.extend(next_amax[i].store(0.) for i in range(2))
  return UOp.group(*stores).sink(arg=KernelInfo("fa_fp8_bwd_scales"))

@functools.cache
def custom_fp8_backward_prep(do8:UOp, delta:UOp, do:UOp, out:UOp, scales:UOp, *reset_amax:UOp):
  B,N,H,D = do.shape
  row = UOp.range(B*N*H, 0)
  d = UOp.range(D, 1, AxisType.REDUCE)
  b,n,h = row//(N*H), row//H%N, row%H
  scale = scales[4]/57344. if scales.numel() == 5 else scales[1]
  rounded = (do[b,n,h,d].cast(dtypes.float)/scale).maximum(-57344).minimum(57344).cast(dtypes.fp8e5m2)
  out = out.after(do8[b,n,h,d].store(rounded))
  # Delta must use rounded FP8 dO, with descale applied before multiplication by O.
  # E5M2 is the high byte of IEEE half; decoding its bits also preserves rounding on emulated-FP8 backends.
  decoded = (rounded.bitcast(dtypes.uint8).cast(dtypes.uint16)<<8).bitcast(dtypes.half).cast(dtypes.float)
  value = (out[b,n,h,d].cast(dtypes.float)*(decoded*scale)).reduce(d, arg=Ops.ADD)
  stores = [delta[b,h,n].store(value)]
  if reset_amax:
    stores.extend(reset_amax[0][UOp.const(i).valid(row.eq(0))].store(0.) for i in range(2))
  opts = None
  dev = do.device[0] if isinstance(do.device,tuple) else do.device
  if (B,N,H,D) == (2,8192,32,128) and scales.numel() == 5 and Device[dev].renderer.target.arch == "gfx950":
    # Preserve the original eight-way strided sum, with four rows per workgroup and partial unrolling.
    opts = (Opt(OptOps.SPLIT,2,(4,AxisType.LOCAL)),Opt(OptOps.SPLIT,4,(8,AxisType.GROUP_REDUCE)),
            Opt(OptOps.SPLIT,5,(4,AxisType.UNROLL)))
  return UOp.group(*stores).end(row).sink(arg=KernelInfo("fa_fp8_bwd_prep",opts_to_apply=opts))

def fp8_backward(q8:Tensor, k8:Tensor, v8:Tensor, v_descale:Tensor, do:Tensor, out:Tensor, lse:Tensor,
                 p_descale:Tensor, ds_descale:Tensor, next_amax:Tensor|None=None, *, reset_next_amax:bool=False,
                 delayed_state:Tensor|None=None):
  """FP8 backward with explicit delayed scales; accumulates next amax locally."""
  from extra.llama_kernels import alloc_like, local_abs_max
  B,N,H,D = q8.shape
  assert (N,H,D) == (8192,32,128) and q8.dtype == k8.dtype == v8.dtype == dtypes.fp8e4m3
  axis = q8.uop.axis if isinstance(q8.device,tuple) else None
  assert axis in (None,0), "FP8 backward currently supports data parallelism only"
  H_KV = k8.shape[2]
  assert k8.shape == v8.shape == (B,N,H_KV,D) and do.shape == out.shape == q8.shape
  def alloc(shape,dtype=dtypes.float32): return alloc_like(shape,dtype,q8.device,axis)
  dq = alloc(q8.shape,dtypes.bfloat16)
  # Reuse the compulsory dQ initialization pass to scan dO for its current scale.
  partial = alloc((B,min(512,N*H*D)))
  dq,partial = Tensor.custom_kernel(dq,partial,do,fxn=custom_fp8_backward_init)[:2]
  if next_amax is None:
    next_amax = Tensor.empty(2,device=q8.device,dtype=dtypes.float32)
    reset_next_amax = True
  if delayed_state is not None:
    assert reset_next_amax
    # The assembly consumes the first four entries; prep also needs the unnormalized dO scale in entry four.
    scales,next_amax = Tensor.custom_kernel(Tensor.empty(5,device=q8.device,dtype=dtypes.float32),next_amax,
      partial,delayed_state,v_descale.reshape(1),fxn=functools.partial(custom_fp8_backward_scales,D=D))[:2]
  else:
    do_scale = ((local_abs_max(partial)+1e-8)/57344.).reshape(1)
    # Before the first amax observation, bound dS from the current dO and V ranges.
    ds_descale = (ds_descale > 0).where(ds_descale, 4*D*do_scale*v_descale*448.)
    scales = Tensor.cat(v_descale.reshape(1),do_scale,p_descale.reshape(1),ds_descale.reshape(1)).contiguous()
  prep = Tensor.custom_kernel(alloc(q8.shape,dtypes.fp8e5m2),alloc((B,H,N)),do,out,scales,
    *((next_amax,) if reset_next_amax and delayed_state is None else ()),fxn=custom_fp8_backward_prep)
  do8, delta = prep[:2]
  if reset_next_amax and delayed_state is None: next_amax = prep[5]
  dk,dv = [alloc(q8.shape,dtypes.bfloat16) for _ in range(2)]
  amax = alloc((B,H,N//64,2))
  dev = q8.device[0] if isinstance(q8.device,tuple) else q8.device
  local_b = B//len(q8.device) if axis == 0 else B
  # The assembly forward writes LSE in the backward kernel's contiguous B,H,N layout.
  ret = Tensor.custom_kernel(dq,dk,dv,amax,next_amax,q8,k8,v8,do8,delta,lse,scales,
    fxn=functools.partial(custom_fp8_backward,B=local_b,N=N,H=H,H_KV=H_KV,arch=Device[dev].renderer.target.arch))
  dq,dk,dv = ret[:3]
  dq = unpack_dq(dq)
  dk = dk.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  dv = dv.reshape(B,N,H_KV,H//H_KV,D).sum(3)
  return dq,dk,dv,ret[3]
