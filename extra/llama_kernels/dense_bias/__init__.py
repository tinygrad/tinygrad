from __future__ import annotations
import functools, math
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops

# A dense MX GEMM consumes the same output gradient in two places: its weight-gradient path transpose-quantizes it,
# while the following bias add sums it over rows. The bias backward runs first, so produce both from one BF16 read and
# hand the columnwise MX operands to the GEMM backward through this graph-construction-time mailbox.
_dense_bias_grad_mailbox: dict[UOp, tuple[UOp, ...]] = {}

def dense_bias_grad_lookup(gradient:UOp) -> tuple[UOp, ...]|None:
  seen, stack = set(), [gradient]
  while stack and len(seen) < 32:
    u = stack.pop()
    if id(u) in seen: continue
    seen.add(id(u))
    if (hit := _dense_bias_grad_mailbox.get(u)) is not None: return hit
    if u is not u.base and (hit := _dense_bias_grad_mailbox.get(u.base)) is not None: return hit
    if u.op in (Ops.AFTER, Ops.RESHAPE, Ops.SHRINK, Ops.PAD, Ops.STAGE, Ops.CAST): stack.extend(u.src)
  return None

@functools.cache
def _dense_bias_fwd_fxn(x_p:UOp, bias_p:UOp, device):
  return Tensor(x_p, device=device) + Tensor(bias_p, device=device)

def _dense_bias_bwd(gradient:UOp, call:UOp) -> tuple[UOp, UOp]:
  x_u, bias_u = call.src[1:3]
  return dense_bias_backward(gradient, x_u, bias_u)

def dense_bias_backward(gradient:UOp, x_u:UOp, bias_u:UOp) -> tuple[UOp, UOp]:
  g = Tensor(gradient, device=x_u.device)
  assert g.dtype == dtypes.bfloat16 and bias_u.dtype == dtypes.bfloat16
  assert g.shape[-1] == bias_u.shape[0] and math.prod(g.shape[:-1]) % 32 == 0, \
    f"unsupported dense bias gradient: g={g.shape}/{g.uop.shard_shape} axis={g.uop.axis}, bias={bias_u.shape}/{bias_u.shard_shape}"
  if getenv("FUSED_DENSE_DUAL_QUANT", 0):
    from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8_bias_dual
    g_col, g_col_e8, g_col_si, bias_partial, g_row, g_row_e8, g_row_si = transpose_quantize_mxfp8_bias_dual(g.reshape(-1, g.shape[-1]))
    entry = (g_col.uop, g_col_e8.uop, g_col_si.uop, g_row.uop, g_row_e8.uop, g_row_si.uop)
  else:
    from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8_bias
    g_col, g_col_e8, g_col_si, bias_partial = transpose_quantize_mxfp8_bias(g.reshape(-1, g.shape[-1]))
    entry = (g_col.uop, g_col_e8.uop, g_col_si.uop)
  for u in (gradient, gradient.base): _dense_bias_grad_mailbox[u] = entry
  return gradient, bias_partial.sum(0)[:bias_u.shape[0]].cast(bias_u.dtype).uop

def dense_bias_add(x:Tensor, bias:Tensor) -> Tensor:
  assert bias.shape == (x.shape[-1],) and x.device == bias.device
  fxn = _dense_bias_fwd_fxn(x.as_param(0).uop, bias.as_param(1).uop, x.device)
  return Tensor(fxn.uop.call_with_output(x.uop, bias.uop, grad_fxn=_dense_bias_bwd))
