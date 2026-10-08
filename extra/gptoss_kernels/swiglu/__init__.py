import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _swiglu_quantize(q:UOp, e8:UOp, h:UOp) -> UOp:
  rows = h.shape[0]
  sink = UOp.sink(q.base, e8.base, h.base, UOp.special(256, "lidx0"), UOp.special(rows*96//64, "gidx0"),
                  arg=KernelInfo("gptoss_swiglu_quantize"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  lib = compile_hip(src, ["-fno-fast-math", "-ffp-contract=off"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _swiglu_quantize_backward(gradient:UOp, kernel:UOp) -> tuple:
  _, e8, h = kernel.src[1:4]
  x = Tensor(h)
  scales = (127.0 - Tensor(e8.after(kernel)).float()).exp2().reshape(-1, 96, 1).expand(-1, 96, 32).reshape(-1, 3072)
  dy = (Tensor(gradient).float() * scales).cast(dtypes.bfloat16)[:, :2880]
  glu, linear = x[:, ::2].clamp(max_=7.0), x[:, 1::2].clamp(-7.0, 7.0)
  y = (glu * (1.702 * glu).sigmoid()) * (linear + 1)
  return None, None, y.gradient(x, gradient=dy)[0].uop

def fused_swiglu_quantize(h:Tensor) -> tuple[Tensor, Tensor]:
  assert h.ndim == 2 and h.shape[1] == 5760 and h.dtype == dtypes.bfloat16
  assert h.uop.shard_shape[0] % 2 == 0
  rows = h.shape[0]
  q = alloc_like((rows, 3072), dtypes.fp8e4m3, h.device, h.uop.axis)
  e8 = alloc_like((rows, 96), dtypes.uint8, h.device, h.uop.axis)
  q, e8, _ = Tensor.custom_kernel(q, e8, h, fxn=_swiglu_quantize, grad_fxn=_swiglu_quantize_backward)
  return q, e8
