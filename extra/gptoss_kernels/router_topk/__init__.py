import functools, math, pathlib
from tinygrad import Tensor, dtypes
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
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

@functools.cache
def _router_topk_bwd_kernel(grad_logits:UOp, grad_weights:UOp, weights:UOp, indices:UOp) -> UOp:
  tokens = math.prod(grad_logits.shape[:-1])
  sink = UOp.sink(grad_logits.base, grad_weights.base, weights.base, indices.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_bwd_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"backward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

def _router_topk_bwd(gradient:UOp, kernel:UOp) -> tuple:
  weights_u, indices_u, logits_u = kernel.src[1:4]
  grad_logits = alloc_like(logits_u.shape, dtypes.float32, logits_u.device, logits_u.axis)
  # The producer uses Routing's grouped shape, so these are the exact saved layer outputs.
  weights, indices = Tensor(weights_u.after(kernel)), Tensor(indices_u.after(kernel))
  grad_logits, *_ = Tensor.custom_kernel(grad_logits, Tensor(gradient).contiguous(), weights, indices, fxn=_router_topk_bwd_kernel)
  return None, None, grad_logits.uop

def fused_router_topk(logits:Tensor) -> tuple[Tensor, Tensor]:
  assert logits.ndim == 3 and logits.shape[-1] == 32 and logits.dtype == dtypes.float32
  axis = logits.uop.axis
  weights = alloc_like((*logits.shape[:-1], 4), dtypes.float32, logits.device, axis)
  indices = alloc_like(weights.shape, dtypes.int32, logits.device, axis)
  weights, indices, *_ = Tensor.custom_kernel(weights, indices, logits, fxn=_router_topk_fwd, grad_fxn=_router_topk_bwd)
  return weights, indices
