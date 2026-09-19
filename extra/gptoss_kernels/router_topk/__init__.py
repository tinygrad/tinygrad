import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _router_topk_fwd(weights:UOp, indices:UOp, logits:UOp) -> UOp:
  tokens = math.prod(logits.shape[:-1])
  sink = UOp.sink(weights.base, indices.base, logits.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                  UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

@functools.cache
def _router_topk_bwd_kernel(grad_logits:UOp, bias_partials:UOp, grad_weights:UOp, weights:UOp, indices:UOp) -> UOp:
  tokens = math.prod(grad_logits.shape[:-1])
  sink = UOp.sink(grad_logits.base, bias_partials.base, grad_weights.base, weights.base, indices.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_bwd_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"backward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                  UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

def router_topk_backward(gradient:Tensor, weights:Tensor, indices:Tensor) -> tuple[Tensor, Tensor]:
  groups, tokens, _ = weights.shape
  grad_logits = alloc_like((groups, tokens, 32), dtypes.float32, weights.device, weights.uop.axis)
  bias_partials = alloc_like((groups, (tokens+255)//256, 32), dtypes.float32, weights.device, weights.uop.axis)
  grad_logits, bias_partials, *_ = Tensor.custom_kernel(grad_logits, bias_partials, gradient.contiguous(), weights, indices,
                                                        fxn=_router_topk_bwd_kernel)
  return grad_logits, bias_partials

def _router_topk_bwd(gradient:UOp, kernel:UOp) -> tuple:
  weights_u, indices_u = kernel.src[1:3]
  grad_logits, _ = router_topk_backward(Tensor(gradient), Tensor(weights_u.after(kernel)), Tensor(indices_u.after(kernel)))
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

def fused_router_topk(logits:Tensor) -> tuple[Tensor, Tensor]:
  assert logits.ndim == 3 and logits.shape[-1] == 32 and logits.dtype == dtypes.float32
  axis = logits.uop.axis
  weights = alloc_like((*logits.shape[:-1], 4), dtypes.float32, logits.device, axis)
  indices = alloc_like(weights.shape, dtypes.int32, logits.device, axis)
  weights, indices, *_ = Tensor.custom_kernel(weights, indices, logits, fxn=_router_topk_fwd, grad_fxn=_router_topk_bwd)
  return weights, indices
