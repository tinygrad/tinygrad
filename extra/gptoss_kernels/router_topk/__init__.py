import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.hipcc import hip_program
from extra.llama_kernels import alloc_like, compile_hip, owned_empty

FUSED_ROUTER_BIAS_GRAD = getenv("FUSED_ROUTER_BIAS_GRAD", 0)
_router_bias_grad_mailbox: dict[UOp, UOp] = {}

def _router_bias_grad_lookup(gradient:UOp) -> UOp|None:
  if not FUSED_ROUTER_BIAS_GRAD or not _router_bias_grad_mailbox: return None
  seen, stack = set(), [gradient]
  while stack and len(seen) < 64:
    u = stack.pop()
    if id(u) in seen: continue
    seen.add(id(u))
    if (hit := _router_bias_grad_mailbox.get(u)) is not None: return hit
    if u is not u.base and (hit := _router_bias_grad_mailbox.get(u.base)) is not None: return hit
    if u.op in (Ops.AFTER, Ops.RESHAPE, Ops.SHRINK, Ops.STAGE, Ops.CAST): stack.extend(u.src)
  return None

@functools.cache
def _router_topk_fwd(weights:UOp, indices:UOp, logits:UOp) -> UOp:
  tokens = math.prod(logits.shape[:-1])
  sink = UOp.sink(weights.base, indices.base, logits.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  return hip_program(sink, src, compile_hip(src, [f"-DTOKENS={tokens}"]))

@functools.cache
def _router_topk_bwd_kernel(grad_logits:UOp, bias_partials:UOp, grad_weights:UOp, weights:UOp, indices:UOp) -> UOp:
  tokens = math.prod(grad_logits.shape[:-1])
  sink = UOp.sink(grad_logits.base, bias_partials.base, grad_weights.base, weights.base, indices.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_bwd_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"backward.cpp").read_text()
  return hip_program(sink, src, compile_hip(src, [f"-DTOKENS={tokens}"]))

def router_topk_backward(gradient:Tensor, weights:Tensor, indices:Tensor, *, local_bias:bool=False) -> tuple[Tensor, Tensor]:
  groups, tokens, _ = weights.shape
  grad_logits = owned_empty(alloc_like((groups, tokens, 32), dtypes.float32, weights.device, weights.uop.axis))
  bias_partials = alloc_like(((tokens+255)//256, 32), dtypes.float32, weights.device) if local_bias else \
    alloc_like((groups, (tokens+255)//256, 32), dtypes.float32, weights.device, weights.uop.axis)
  grad_logits, bias_partials, *_ = Tensor.custom_kernel(grad_logits, bias_partials, gradient.contiguous(), weights, indices,
                                                        fxn=_router_topk_bwd_kernel)
  return grad_logits, bias_partials

def _router_topk_bwd(gradient:UOp, kernel:UOp) -> tuple:
  weights_u, indices_u = kernel.src[1:3]
  grad_logits, partials = router_topk_backward(Tensor(gradient), Tensor(weights_u.after(kernel)), Tensor(indices_u.after(kernel)), local_bias=True)
  if FUSED_ROUTER_BIAS_GRAD:
    grad_bias = partials.sum(0)
    for u in (grad_logits.uop, grad_logits.uop.base): _router_bias_grad_mailbox[u] = grad_bias.uop
  return None, None, grad_logits.uop

def _router_bwd(gradient:UOp, call:UOp) -> tuple:
  x, weight, bias = (Tensor(u) for u in call.src[1:4])
  weights, indices = (Tensor(u) for u in call.unbound_outputs)
  grad_logits, bias_partials = router_topk_backward(Tensor(gradient), weights, indices)
  grad_x, grad_weight = (x.float() @ weight.float().T).gradient(x, weight, gradient=grad_logits.reshape(*x.shape[:-1], 32))
  return grad_x.uop, grad_weight.uop, bias_partials.sum((0, 1)).cast(bias.dtype).uop

@function(grad_fxn=_router_bwd)
def fused_router(x:Tensor, weight:Tensor, bias:Tensor) -> tuple[Tensor, Tensor]:
  from extra.gemm.moe_routing import router_mfma
  logits = router_mfma(x, weight, bias) if getenv("ROUTER_MFMA", 0) else x.float() @ weight.float().T + bias.float()
  groups = len(x.device) if isinstance(x.device, tuple) else 1
  return fused_router_topk(logits.reshape(groups, -1, 32))

def fused_router_topk(logits:Tensor, out:tuple[Tensor, Tensor]|None=None) -> tuple[Tensor, Tensor]:
  assert logits.ndim == 3 and logits.shape[-1] == 32 and logits.dtype == dtypes.float32
  axis = logits.uop.axis
  if out is None:
    weights = alloc_like((*logits.shape[:-1], 4), dtypes.float32, logits.device, axis)
    indices = alloc_like(weights.shape, dtypes.int32, logits.device, axis)
  else:
    weights, indices = out
    assert weights.shape == indices.shape == (*logits.shape[:-1], 4)
    assert weights.device == indices.device == logits.device and weights.uop.axis == indices.uop.axis == axis
    assert weights.dtype == dtypes.float32 and indices.dtype == dtypes.int32
  weights, indices, *_ = Tensor.custom_kernel(weights, indices, logits, fxn=_router_topk_fwd, grad_fxn=_router_topk_bwd)
  return weights, indices
