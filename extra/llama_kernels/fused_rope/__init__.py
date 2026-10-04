import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

# Fused RoPE: one kernel does the complex-multiply rotation + bf16 cast in a single pass over x, replacing
# apply_rotary_emb's reshape/complex_mult/cat/flatten/cast chain. x:[B,S,H,HD] bf16, cos/sin:[S,HD//2].
# For output index d, pair p=d//2:  d even -> a*cos - b*sin ;  d odd -> a*sin + b*cos   (a=x[2p], b=x[2p+1]).

@functools.cache
def _custom_rope_fwd(out:UOp, x:UOp, cos:UOp, sin:UOp, B:int, S:int, H:int, HD:int) -> UOp:
  b = UOp.range(B, 0); s = UOp.range(S, 1); h = UOp.range(H, 2); d = UOp.range(HD, 3)
  p = d // 2
  a  = x[b, s, h, p*2].cast(dtypes.float)
  bb = x[b, s, h, p*2 + 1].cast(dtypes.float)
  c = cos[s, p].cast(dtypes.float); sn = sin[s, p].cast(dtypes.float)
  val = (d % 2).eq(0).where(a*c - bb*sn, a*sn + bb*c)
  return out[b, s, h, d].store(val.cast(out.dtype)).end(d, h, s, b).sink(arg=KernelInfo(f"rope_fwd_{B}_{S}_{H}_{HD}"))

@functools.cache
def _custom_rope_bwd(dx:UOp, dout:UOp, cos:UOp, sin:UOp, B:int, S:int, H:int, HD:int) -> UOp:
  # rope is an orthogonal rotation -> grad is the transpose (rope with -sin):
  #   d even: dx = dout[2p]*cos + dout[2p+1]*sin ;  d odd: dx = -dout[2p]*sin + dout[2p+1]*cos
  b = UOp.range(B, 0); s = UOp.range(S, 1); h = UOp.range(H, 2); d = UOp.range(HD, 3)
  p = d // 2
  do_e = dout[b, s, h, p*2].cast(dtypes.float)
  do_o = dout[b, s, h, p*2 + 1].cast(dtypes.float)
  c = cos[s, p].cast(dtypes.float); sn = sin[s, p].cast(dtypes.float)
  val = (d % 2).eq(0).where(do_e*c + do_o*sn, -do_e*sn + do_o*c)
  return dx[b, s, h, d].store(val.cast(dx.dtype)).end(d, h, s, b).sink(arg=KernelInfo(f"rope_bwd_{B}_{S}_{H}_{HD}"))

def _fused_rope_bwd(gradient:UOp, kernel:UOp, cos_t:Tensor, sin_t:Tensor):
  # forward inputs are (out, x, cos, sin); gradient is d_out with x's shape
  _, x_u, _, _ = kernel.src[1:]
  device = x_u.device
  B, S, H, HD = x_u.shape
  if isinstance(device, tuple):
    axis = x_u.axis
    ndev = len(device)
    local = tuple(v//ndev if i == axis else v for i, v in enumerate((B, S, H, HD)))
    dx = Tensor(Tensor.invalids(*local, dtype=x_u.dtype, device=device).uop.unshard(axis), device=device)
    Bl, Sl = local[0], local[1]
  else:
    dx = Tensor.invalids(B, S, H, HD, dtype=x_u.dtype, device=device); Bl, Sl = B, S
  dout_t = Tensor(gradient, device=device)
  fxn = functools.partial(_custom_rope_bwd, B=Bl, S=Sl, H=H, HD=HD)
  dx = Tensor.custom_kernel(dx, dout_t, cos_t, sin_t, fxn=fxn)[0]
  return (None, dx.uop, None, None)

def fused_rope(x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
  # x:[B,S,H,HD] bf16 ; cos,sin:[S,HD//2]  -> rope'd x, same shape/dtype
  assert x.ndim == 4, f"expected (B,S,H,HD), got {x.shape}"
  B, S, H, HD = x.shape
  if isinstance(x.device, tuple):
    axis = x.uop.axis
    assert axis in (0, None), f"fused_rope expects batch(0)/replicated sharding, got axis={axis}"
    ndev = len(x.device)
    Bl = B // ndev if axis == 0 else B
    out = Tensor(Tensor.invalids(Bl, S, H, HD, dtype=x.dtype, device=x.device).uop.unshard(0), device=x.device) \
          if axis == 0 else Tensor.invalids(B, S, H, HD, dtype=x.dtype, device=x.device)
  else:
    Bl, out = B, Tensor.invalids(B, S, H, HD, dtype=x.dtype, device=x.device)
  fxn = functools.partial(_custom_rope_fwd, B=Bl, S=S, H=H, HD=HD)
  grad_fxn = functools.partial(_fused_rope_bwd, cos_t=cos, sin_t=sin)
  out = Tensor.custom_kernel(out, x, cos, sin, fxn=fxn, grad_fxn=grad_fxn)[0]
  return out
