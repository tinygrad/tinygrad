from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG, getenv, prod
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x, _to_torch_dtype(x.dtype))
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend: pass
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend)
torch.utils.generate_methods_for_privateuse1_backend()

@torch.library.impl("aten::addmm", "privateuseone")
def addmm(input, mat1, mat2, *, beta=1, alpha=1): return wrap(unwrap(input) * beta + (unwrap(mat1).matmul(unwrap(mat2))) * alpha)

@torch.library.impl("aten::zero_", "privateuseone")
def zero_(x):
  tt = unwrap(x)
  tt.replace(tt.zeros_like())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
def fill_scalar(x, y):
  tt = unwrap(x)
  tt.replace(tt.full_like(y))

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()

@torch.library.impl("aten::masked_select", "privateuseone")
def masked_select(self, mask):
  # err, bad
  return wrap(Tensor(self.cpu().numpy()[mask.cpu().numpy()]))

# @torch.library.impl("aten::as_strided", "privateuseone")
# def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
#   storage_offset_int = storage_offset if storage_offset is not None else tensor.storage_offset()

#   #return tensor.cpu().as_strided(size, stride).tiny()
#   if TORCH_DEBUG >= 1: print("** NOTE: this as_strided is wrong", tensor.shape, size, stride, storage_offset)

#   if tuple(x for x in tensor.shape if x != 1) == tuple(x for x in size if x != 1):
#     # this is squeeze/unsqueeze
#     return tensor.reshape(size)

#   # TODO: how do i know this is permute?
#   if len(size) == 2 and size == [tensor.shape[1], tensor.shape[0]] and stride == [0, 1]: return wrap(unwrap(tensor).permute(1,0))

#   #print(tensor.cpu().numpy())
#   raise NotImplementedError("fix as_strided")

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor, size, stride, storage_offset=None):
  # print('as_strided:', tensor.shape, size, stride, storage_offset, end='')
  tg_tensor = unwrap(tensor)
  if len(size) != len(stride):raise ValueError(f"Length of size ({len(size)}) must match length of stride ({len(stride)})")
  storage_offset = storage_offset if storage_offset is not None else tensor.storage_offset()
  storage_offset = unwrap(storage_offset) if isinstance(storage_offset, torch.Tensor) else storage_offset
  if storage_offset: tg_tensor = tg_tensor[storage_offset:]
  tg_tensor = tg_tensor.contiguous().realize()
  total_elements = prod(tg_tensor.shape)
  target_elements = prod(size) if size else 0
  if tuple(x for x in tg_tensor.shape if x != 1) == tuple(x for x in size if x != 1): return wrap(tg_tensor.reshape(size))
  if type(storage_offset) == Tensor: print(storage_offset.shape)
  if size and total_elements != target_elements: raise ValueError(f"Size mismatched, can't reshape {tg_tensor.shape} ({total_elements} elements) -> {size} ({target_elements} elements)")
  if size: tg_tensor = tg_tensor.reshape(size)
  # print(', END:', tg_tensor.shape, tg_tensor.requires_grad)
  return wrap(tg_tensor)

def prod(shape):
  result = 1
  for dim in shape:
    result *= dim
  return result

@torch.library.impl("aten::nll_loss_forward", "privateuseone")
def nll_loss_forward_output(self, target, weight=None, reduction=1, ignore_index=-100):
  input_tg = unwrap(self)
  target_tg = unwrap(target) if isinstance(target, torch.Tensor) else target
  weight_tg = unwrap(weight) if weight is not None else None
  reduction_str = {0: "none", 1: "mean", 2: "sum"}.get(reduction, "mean")
  loss = input_tg.nll_loss(target_tg, weight=weight_tg, ignore_index=ignore_index, reduction=reduction_str)
  total_weight = Tensor.ones(1) if weight_tg is None else weight_tg.sum()
  return wrap(loss), wrap(total_weight)

@torch.library.impl("aten::nll_loss_backward.grad_input", "privateuseone")
def nll_loss_backward_grad_input(grad_output, self, target, weight=None, reduction=1, ignore_index=-100, total_weight=None, *, grad_input):
  grad_output_tg = unwrap(grad_output)
  input_tg = unwrap(self).requires_grad_(True)
  target_tg = unwrap(target) if isinstance(target, torch.Tensor) else target
  weight_tg = unwrap(weight) if weight is not None else None
  total_weight_tg = unwrap(total_weight) if total_weight is not None else Tensor.ones(1)
  reduction_str = {0: "none", 1: "mean", 2: "sum"}.get(reduction, "mean")
  loss = input_tg.nll_loss(target_tg, weight=weight_tg, ignore_index=ignore_index, reduction=reduction_str)
  loss.backward()
  grad = input_tg.grad * grad_output_tg if reduction_str != "none" else input_tg.grad
  unwrap(grad_input).replace(grad)
  return grad_input

@torch.library.impl("aten::_log_softmax_backward_data.out", "privateuseone")
def log_softmax_backward_data_out(grad_output, output, dim, input_dtype, *, out):
  grad_output_tg = unwrap(grad_output)
  output_tg = unwrap(output).requires_grad_(True)
  input_tg = output_tg.softmax(dim)
  input_tg.backward(grad_output_tg)
  unwrap(out).replace(output_tg.grad)
  return out

@torch.library.impl("aten::max_pool2d_with_indices_backward.grad_input", "privateuseone")
def max_pool2d_with_indices_backward_grad_input(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, *, grad_input):
  grad_output_tg = unwrap(grad_output)
  input_tg = unwrap(self).requires_grad_(True)
  stride = stride if stride is not None else kernel_size
  out = input_tg.max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
  out.backward(grad_output_tg)
  unwrap(grad_input).replace(input_tg.grad)
  return grad_input

@torch.library.impl("aten::native_batch_norm", "privateuseone")
def native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps):
  input_tg = unwrap(input)
  weight_tg = unwrap(weight) if weight is not None else None
  bias_tg = unwrap(bias) if bias is not None else None
  running_mean_tg = unwrap(running_mean) if running_mean is not None else None
  running_var_tg = unwrap(running_var) if running_var is not None else None
  if training:
    mean = input_tg.mean(axis=(0, 2, 3), keepdim=True)
    var = input_tg.var(axis=(0, 2, 3), keepdim=True)
    running_mean_tg = running_mean_tg if running_mean_tg is None else running_mean_tg.lerp(mean, momentum)
    running_var_tg = running_var_tg if running_var_tg is None else running_var_tg.lerp(var, momentum)
  else:
    mean = running_mean_tg if running_mean_tg is not None else input_tg.new_zeros(input_tg.shape[1], dtype=input_tg.dtype).view(1, -1, 1, 1)
    var = running_var_tg if running_var_tg is not None else input_tg.new_ones(input_tg.shape[1], dtype=input_tg.dtype).view(1, -1, 1, 1)
  invstd = 1 / Tensor.sqrt(var + eps)
  out = (input_tg - mean) * invstd
  if weight_tg is not None: out = out * weight_tg.view(1, -1, 1, 1)
  if bias_tg is not None: out = out + bias_tg.view(1, -1, 1, 1)
  return wrap(out), wrap(mean), wrap(invstd)

@torch.library.impl("aten::native_batch_norm_backward", "privateuseone")
def native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
  grad_output_tg = unwrap(grad_output)
  input_tg = unwrap(input).requires_grad_(True)
  weight_tg = unwrap(weight).requires_grad_(True) if weight is not None else None
  running_mean_tg = unwrap(running_mean) if running_mean is not None else None
  running_var_tg = unwrap(running_var) if running_var is not None else None
  save_mean_tg = unwrap(save_mean) if save_mean is not None else None
  save_invstd_tg = unwrap(save_invstd) if save_invstd is not None else None
  out = input_tg.batchnorm(weight=weight_tg, bias=None, mean=running_mean_tg, invstd=running_var_tg)
  out.backward(grad_output_tg)
  grad_input = input_tg.grad if output_mask[0] else None
  grad_weight = weight_tg.grad if output_mask[1] and weight_tg is not None else None
  grad_bias = weight_tg.grad if output_mask[2] and weight_tg is not None else None
  return tuple(wrap(g) if g is not None else None for g in (grad_input, grad_weight, grad_bias))

@torch.library.impl("aten::threshold_backward.grad_input", "privateuseone")
def threshold_backward_grad_input(grad_output, self, threshold, *, grad_input):
  grad_output_tg = unwrap(grad_output)
  input_tg = unwrap(self).requires_grad_(True)
  out = (input_tg <= threshold).where(input_tg, 0) # TODO: What is the function for this
  out.backward(grad_output_tg)
  unwrap(grad_input).replace(input_tg.grad)
  return grad_input

@torch.library.impl("aten::lerp.Scalar_out", "privateuseone")
def lerp_scalar_out(self, end, weight, out):
  self_tg = unwrap(self)
  end_tg = unwrap(end)
  result_tg = self_tg.lerp(end_tg, weight)
  unwrap(out).replace(result_tg)
  return out

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  if TORCH_DEBUG: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype))
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if TORCH_DEBUG: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype()))
  return wrap(ret)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: support return_indices in tinygrad
  ret = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
  # TODO: this is wrong
  return (wrap(ret), wrap(Tensor.zeros_like(ret, dtype=dtypes.int64)))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  return wrap(unwrap(input).conv2d(unwrap(weight), unwrap(bias) if bias is not None else None,
                                   groups=groups, stride=stride, dilation=dilation, padding=padding))
  #raise NotImplementedError("need convolution")

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  grad_output_tg = unwrap(grad_output)
  input_tg = unwrap(input).requires_grad_(True)
  weight_tg = unwrap(weight).requires_grad_(True)
  output_tg = input_tg.conv2d(weight_tg,stride=stride,padding=padding,dilation=dilation,groups=groups)
  output_tg.backward(grad_output_tg)
  grad_input = input_tg.grad if output_mask[0] else None
  grad_weight = weight_tg.grad if output_mask[1] else None
  if grad_input is None and grad_weight is None: 
    return (Tensor.empty(), Tensor.empty(), Tensor.empty())
  elif grad_input is None:
    # print(grad_input, grad_weight)
    return tuple(wrap(g) for g in (Tensor.empty(grad_weight.shape[1]), grad_weight, Tensor.empty(grad_weight.shape[0])))
  elif grad_weight is None: 
    print(grad_input, '-A-', grad_weight)
    return tuple(wrap(g) for g in (grad_input, Tensor.empty(grad_weight.shape[1]), Tensor.empty(grad_weight.shape[0])))
  return tuple(wrap(g) for g in (grad_input, grad_weight, Tensor.empty(grad_weight.shape[0])))

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src, dest):
  if str(src.device) == "tiny" and str(dest.device) == "tiny":
    unwrap(dest).replace(unwrap(src), allow_shape_mismatch=True)
  elif str(src.device) == "tiny" and str(dest.device) == "cpu":
    # TODO: is there a better way?
    dest.resize_(src.numel()).resize_(src.shape)
    dest.copy_(torch.from_numpy(unwrap(src).numpy()))
  elif str(src.device) == "cpu" and str(dest.device) == "tiny":
    unwrap(dest).assign(Tensor(src.numpy()))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
def cat_out(tensors, out, dim=0): unwrap(out).replace(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim), allow_shape_mismatch=True)

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y): return wrap(unwrap(x)[y[0].tolist()])


@torch.library.impl("aten::addcmul.out", "privateuseone")
def addcmul_out(self, tensor1, tensor2, *, out, value=1):
  result = unwrap(self) + (unwrap(tensor1) * unwrap(tensor2)) * value
  return wrap(unwrap(out).assign(result.detach())).detach()

@torch.library.impl("aten::addcdiv.out", "privateuseone")
def addcdiv_out(self, tensor1, tensor2, *, out, value=1):
  result = unwrap(self) + (unwrap(tensor1) / unwrap(tensor2)) * value
  return wrap(unwrap(out).assign(result.detach())).detach()

# register some decompositions
# from torch._decomp import get_decompositions
# aten = torch.ops.aten
# decomps = [
#   aten.addmm,
# ]
# for k,v in get_decompositions(decomps).items():
#   key = str(k._schema).split("(")[0]
#   if TORCH_DEBUG >= 2: print("register decomp for", k)
#   torch.library.impl(key, "privateuseone")(v)

def call_func(obj, func_str):
  func = getattr(obj, func_str, None)
  if callable(func):
    func()
  else:
    print(f"Function {func_str} not found")

# TODO: Categorize these functions and move them to a separate file
IDENTICAL_FUNCS = ["abs", "bitwise_not", "logical_not", "argmin", "argmin", "all", "flip", "tril", "triu", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "celu", "clamp", "cos", "cosh", "elu", "erf", "hardsigmoid", "hardswish", "hardtanh", "tan", "tanh", "trunc", "sigmoid", "sign", "silu", "sin", "sinh", "softplus", "round", "log", "log2", "min", "mish", "gather", "argmax", "reciprocal", "mean", "sqrt", "rsqrt", "neg", "atan", "acos", "acosh", "asin", "asinh", "atanh", "ceil", "celu", "clamp", "cos", "cosh", "elu", "erf", "abs", "exp", "exp2", "min", "max", "relu"]
IDENTICAL_ASSIGN_FUNCS = [
  # "relu", 
  "softmax"]
identical_dict = {f"aten.{func}": getattr(Tensor, func) for func in IDENTICAL_FUNCS}
identical_assign_dict = {f"aten.{func}": lambda x: x.assign(getattr(Tensor, func)(x)) for func in IDENTICAL_ASSIGN_FUNCS}

tiny_backend = {
  "aten.mul.out": lambda x,y,*,out: out.assign(x * y),
  "aten.div.out": lambda x,y,*,out: out.assign(x / y),
  "aten.relu_": lambda x: x.assign(x.relu()),
  "aten.bitwise_and.Tensor_out": lambda x,y,*,out: out.assign(x & y),
  "aten.bitwise_or.Tensor_out": lambda x,y,*,out: out.assign(x | y),
  "aten.bitwise_xor.Tensor_out": lambda x,y,*,out: out.assign(x ^ y),
  "aten.view": Tensor.reshape,
  "aten.add.Tensor": Tensor.add,
  "aten.sub.Tensor": Tensor.sub,
  "aten.mul.Tensor": Tensor.mul,
  "aten.div.Tensor": Tensor.div,
  "aten.add_.Tensor": lambda x,y,alpha=1: x.assign(x.add(y)*alpha),
  "aten.pow.Tensor_Scalar": Tensor.pow, "aten.pow.Tensor_Tensor": Tensor.pow, "aten.pow.Scalar": lambda x,y: y.pow(x, reverse=True),
  "aten.bitwise_and.Tensor": Tensor.bitwise_and,
  "aten.eq.Tensor": Tensor.eq, "aten.eq.Scalar": Tensor.eq,
  "aten.ne.Tensor": Tensor.ne, "aten.ne.Scalar": Tensor.ne,
  "aten.gt.Tensor": Tensor.__gt__, "aten.gt.Scalar": Tensor.__gt__,
  "aten.lt.Tensor": Tensor.__lt__, "aten.lt.Scalar": Tensor.__lt__,
  "aten.le.Tensor": Tensor.__le__, "aten.le.Scalar": Tensor.__le__,
  "aten.relu_": lambda x: x.assign(x.relu()),
  "aten.mean.dim": Tensor.mean,
  "aten.mm": Tensor.matmul,
  "aten.var.correction": Tensor.var,
  # TODO: support var_mean in tinygrad
  "aten.var_mean.correction": lambda self, dims, keepdim=False, correction=1: (self.var(dims, keepdim, correction), self.mean(dims, keepdim)),
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,out=None:
    out.replace(Tensor.sum(self, axis if len(axis) else None, keepdim), allow_shape_mismatch=True),
  "aten.scatter.value": Tensor.scatter,
  "aten.where.self": Tensor.where,
  "aten._log_softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten.random_": lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype)),
  "aten.uniform_": lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high)),
  "aten.normal_": lambda self, low=0, high=1: self.assign(Tensor.normal(*self.shape, low=low, high=high)),
}

tiny_backend.update(identical_dict)
tiny_backend.update(identical_assign_dict)

# NOTE: there's earlier things to hook these, so the .out form isn't needed
#"aten.add.out": lambda x,y,out: out.replace(x+y, allow_shape_mismatch=True),
#"aten.abs.out": lambda x,out: out.replace(x.abs(), allow_shape_mismatch=True),
#"aten.ceil.out": lambda x,out: out.replace(x.ceil(), allow_shape_mismatch=True),
#"aten.exp2.out": lambda x,out: out.replace(x.exp2(), allow_shape_mismatch=True),

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG: print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
                          {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()})
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    out = f(*args, **kwargs)
    if isinstance(out, Tensor): return wrap(out)
    elif isinstance(out, tuple): return tuple(wrap(x) for x in out)
    else: raise RuntimeError(f"unknown output type {type(out)}")
  return nf

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      # print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  DispatchLog().__enter__()
