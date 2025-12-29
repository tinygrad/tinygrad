from __future__ import annotations
from typing import Callable, Any
from tinygrad import Tensor, dtypes, UOp, Device
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import getenv
from tinygrad import nn

GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))) if getenv("GPUS", 1) > 1 else Device.DEFAULT 

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  x_abs_max = x.abs().max(axis=axis, keepdim=True).detach()
  scale = 448. / (x_abs_max + 1e-8)
  x_scaled = x * scale
  x_det = x_scaled.detach()
  x_clamped = x_det.maximum(-448.0).minimum(448.0)
  x_clamped_ste = x_scaled + (x_clamped - x_det)
  res = x_clamped_ste.cast(dtype).contiguous()
  return res, scale.float().reciprocal().contiguous()

def custom_matmul(C: UOp, A: UOp, B: UOp) -> UOp:
  SEQ = A.shape[1]
  OUT = B.shape[0]
  IN = B.shape[-1]
  c2 = UOp.range(SEQ, 2, AxisType.LOOP)        
  c5 = UOp.range(OUT, 3, AxisType.LOOP)        
  c8 = UOp.range(C.size//SEQ//OUT, 1, AxisType.LOOP)  
  c16 = UOp.range(IN, 0, AxisType.REDUCE)     
  c27 = (A.index((c2*IN+c16+c8*IN*SEQ)) * B.index((c5*IN+c16))).cast(dtypes.float)
  c28 = c27.reduce(c16, arg=Ops.ADD)
  c30 = C.index((c2*OUT+c5+c8*OUT*SEQ), ptr=True).store(c28).end(c8, c2, c5)
  return c30.sink(arg=KernelInfo(name=f"custom_matmul_{A.shape}x{B.shape}"))

def custom_matmul_backward(gradient: UOp, kernel: UOp) -> tuple[UOp, UOp]:
  _, a, b = kernel.src
  a_tensor = Tensor(a, device=a.device)
  g_tensor = Tensor(gradient, device=gradient.device)
  b_tensor = Tensor(b, device=b.device)
  g_quantized, scale = quantize_to_fp8(g_tensor)
  scale_scalar = scale.reshape(())
  grad_b = Tensor.einsum('bso,bsi->oi', g_quantized, a_tensor, dtype=dtypes.float)
  grad_b = grad_b * scale_scalar
  g_2d = g_quantized.reshape(g_tensor.shape[0] * g_tensor.shape[1], g_tensor.shape[-1])
  grad_a = (g_2d.dot(b_tensor, dtype=dtypes.float)).contiguous().reshape(a_tensor.shape) * scale
  return (None, grad_a.uop, grad_b.uop)

class FP8Linear:
  def __init__(self, in_features:int, out_features:int, bias:bool=True):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None

  def __call__(self, x: Tensor) -> Tensor:
    original_ndim = len(x.shape)
    if original_ndim == 2:
      # (batch, in_features) -> (batch, 1, in_features)
      batch, in_feat = x.shape
      assert in_feat == self.in_features, f"Input size {in_feat} doesn't match layer size {self.in_features}"
      x = x.reshape(batch, 1, in_feat)
    elif original_ndim == 3:
      batch, seq, in_feat = x.shape
      assert in_feat == self.in_features, f"Input size {in_feat} doesn't match layer size {self.in_features}"
    else: raise ValueError(f"FP8Linear only supports 2D or 3D inputs, got {original_ndim}D: {x.shape}")
    batch, seq, _ = x.shape
    w_fp8, w_scale = self._get_quantized_weight()
    x_fp8, x_scale = quantize_to_fp8(x)
    if isinstance(GPUS, tuple) and len(GPUS) > 1:
      y = Tensor(Tensor.empty((batch//len(GPUS), seq, self.out_features), dtype=dtypes.float, device=GPUS).uop.multi(0), device=GPUS)
    else:
      y = Tensor.empty((batch, seq, self.out_features), dtype=dtypes.float)
    y = Tensor.custom_kernel(y, x_fp8, w_fp8, fxn=custom_matmul, grad_fxn=custom_matmul_backward)[0]
    y = y * w_scale * x_scale
    if self.bias is not None: y = y.cast(dtypes.half) + self.bias.cast(dtypes.half)
    # (batch, 1, out_features) -> (batch, out_features)
    if original_ndim == 2: y = y.reshape(batch, self.out_features)
    return y.cast(x.dtype) if original_ndim == 3 else y.cast(dtypes.half)

def _swap_linear_with_fp8(model:Any, module_filter_fn:Callable[[Any, str],bool]|None=None, fqn:str="", parent:Any|None=None,
                          attr_name:str="", visited:set|None=None):
  if visited is None: visited = set()
  obj_id = id(model)
  if obj_id in visited: return
  visited.add(obj_id)
  if isinstance(model, nn.Linear):
    if module_filter_fn is not None and not module_filter_fn(model, fqn): return
    fp8_linear = FP8Linear(model.weight.shape[1], model.weight.shape[0], model.bias is not None)
    fp8_linear.weight = model.weight # TODO
    if model.bias is not None: fp8_linear.bias = model.bias
    # Swap in parent
    if parent is not None and attr_name:
      setattr(parent, attr_name, fp8_linear)
    return
  if isinstance(model, (str, int, float, bool, type(None), Tensor, UOp)): return
  if isinstance(model, list):
    for i, item in enumerate(model):
      child_fqn = f"{fqn}.{i}" if fqn else str(i)
      if isinstance(item, nn.Linear):
        if module_filter_fn is None or module_filter_fn(item, child_fqn):
          fp8_linear = FP8Linear(item.weight.shape[1], item.weight.shape[0], item.bias is not None)
          fp8_linear.weight = item.weight
          if item.bias is not None: fp8_linear.bias = item.bias
          model[i] = fp8_linear
      else:
        _swap_linear_with_fp8(item, module_filter_fn, child_fqn, None, "", visited)
    return
  if isinstance(model, dict):
    for key, item in list(model.items()):
      child_fqn = f"{fqn}.{key}" if fqn else str(key)
      if isinstance(item, nn.Linear):
        if module_filter_fn is None or module_filter_fn(item, child_fqn):
          fp8_linear = FP8Linear(item.weight.shape[1], item.weight.shape[0], item.bias is not None)
          fp8_linear.weight = item.weight
          if item.bias is not None: fp8_linear.bias = item.bias
          model[key] = fp8_linear
      else:
        _swap_linear_with_fp8(item, module_filter_fn, child_fqn, None, "", visited)
    return
  # Handle object attributes - only look at instance attributes, not methods
  if not hasattr(model, "__dict__"): return

  for attr_key in list(vars(model).keys()):
    try:
      attr = getattr(model, attr_key)
    except Exception:
      continue

    child_fqn = f"{fqn}.{attr_key}" if fqn else attr_key
    _swap_linear_with_fp8(attr, module_filter_fn, child_fqn, model, attr_key, visited)

def convert_to_float8_training(model: Any, module_filter_fn: Callable[[Any, str], bool] | None = None) -> Any:
  _swap_linear_with_fp8(model, module_filter_fn, "", None, "")
  return model
