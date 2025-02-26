from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG, getenv, prod
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x, _to_torch_dtype(x.dtype))
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_available(self): return True
  def current_device(self): return 0
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()

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

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  #return tensor.cpu().as_strided(size, stride).tiny()
  if TORCH_DEBUG >= 1: print("** NOTE: this as_strided might be wrong", tensor.shape, size, stride, storage_offset)

  nz_strides = [st for s,st in zip(size, stride) if s != 1]
  decending_strides = all(x>=y for x,y in zip(nz_strides[:-1], nz_strides[1:]))

  # this is reshape (squeeze/unsqueeze), strides must be in decending order
  if tuple(x for x in tensor.shape if x != 1) == tuple(x for x in size if x != 1) and decending_strides:
    return tensor.reshape(size)

  # this is also expand, hit?
  if tensor.numel() == 1:
    assert all(x == 0 for x in stride)
    return wrap(unwrap(tensor).reshape([1]*len(size)).expand(size))

  # this is expand
  if len(tensor.shape) == len(size) and all(x == y or x == 1 for x,y in zip(tensor.shape, size)) and decending_strides:
    return wrap(unwrap(tensor).expand(size))

  # this is permute because we are flipping strides
  if len(tensor.shape) == 2 and tuple(tensor.shape)[::-1] == tuple(size) and stride == [0, 1]:
    return wrap(unwrap(tensor).permute(1,0))

  #print(tensor.cpu().numpy())
  raise NotImplementedError(f"fix as_strided {tensor.shape} -> {size} {stride} {storage_offset}")

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

# register some decompositions
from torch._decomp import get_decompositions
aten = torch.ops.aten
decomps = {
  "post_autograd": [
    aten.native_batch_norm, aten.native_batch_norm_backward,
    aten.addmm,
    aten.addcmul,
    aten.addcdiv,
    aten._log_softmax_backward_data,
    aten.threshold_backward,
    aten.softplus_backward,
    aten.elu,  # elu has a scale + input_scale param
    aten.softplus,
    aten.threshold,
    aten.nll_loss_forward,
    aten.nll_loss_backward,
    # AttributeError: 'int' object has no attribute '_broadcasted'
    aten.sigmoid_backward,
    aten.tanh_backward,
    aten.sinc,
    aten._prelu_kernel,
    aten.softshrink,
    aten.hardshrink,
    aten.log_sigmoid_forward,
    aten.isneginf,
    aten.isposinf,
    aten.nan_to_num,
    aten.logit,
    aten.rsub,
    # activations
    aten.hardswish, aten.hardswish_backward,
    aten.hardtanh, aten.hardtanh_backward,
    aten.gelu, aten.gelu_backward,
    # NOTE: many of these don't work or cause infinite loops
    #aten.var_mean,
    #aten.var,
    #aten.rsqrt,
    #aten.max_pool2d_with_indices,
    # NOTE: these are prims
    #aten.digamma,
    #aten.erfinv,
    #aten.lgamma,
    # this needs copy_strided
    #aten.lerp,
  ],
  "meta": [
    aten.max_pool2d_with_indices_backward,
    aten.convolution_backward,
  ],
}

for dctype,lst in decomps.items():
  for k,v in get_decompositions(lst, type=dctype).items():
    key = str(k._schema).split("(")[0]
    if TORCH_DEBUG >= 2: print("register decomp for", k)
    torch.library.impl(key, "privateuseone")(v)

# NOTE: we should only implement the "out" form, it should be 0 overhead
# TODO: due to issue with empty / is_realized, it is slow to use assign so we use replace
# the goal is to make as much as we can this
simple_tensor_methods = [
  # unary (ish)
  "log", "log2", "sqrt", "rsqrt", "sign", "silu", "hardsigmoid", "exp", "exp2", "neg", "reciprocal", "bitwise_not",
  "sigmoid", "clamp", "mish", "erf",
  # trig
  "acos", "acosh", "cos", "cosh", "asin", "asinh", "sin", "sinh", "atan", "atanh", "tan", "tanh",
  # rounding
  "ceil", "round", "floor", "trunc",
  # binary
  "mul", "div", "maximum", "minimum",
  # modify
  "tril", "triu",
  # reduce
  "all", "any", "argmax", "argmin", "cumsum",
  # complex
  "avg_pool2d", "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.mul.out": operator.mul,
  "aten.leaky_relu.out": Tensor.leakyrelu, # TODO: this should be renamed in tinygrad
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": lambda x,y: x^y,  # TODO: tinygrad lacks bitwise_xor, add it
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
}}

# we add the "out" here
def wrap_out(f):
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert getenv("ALLOW_DTYPE_MISMATCH") or out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    return out.replace(assigned)
  return _wrap_out

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.view": Tensor.reshape,
  "aten.floor_divide": lambda x,y: x//y,
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  # relu doesn't have an out form?
  "aten.relu": Tensor.relu,
  "aten.relu_": lambda x: x.assign(x.relu()),
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.mm": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.prod": Tensor.prod,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,
  "aten.std_mean.correction": Tensor.std_mean,
  "aten.var.correction": Tensor.var,
  # TODO: support var_mean in tinygrad
  "aten.var_mean.correction": lambda self, dims, keepdim=False, correction=1: (self.var(dims, keepdim, correction), self.mean(dims, keepdim)),
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,out=None:
    out.replace(Tensor.sum(self, axis if len(axis) else None, keepdim), allow_shape_mismatch=True),
  "aten.scatter.value": Tensor.scatter,
  "aten.gather": Tensor.gather,
  "aten.where.self": Tensor.where,
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),
  "aten.random_": lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype)),
  "aten.random_.from": lambda self, from_, to:
    self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype)),
  "aten.uniform_": lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high)),
  "aten.normal_": lambda self, low=0, high=1: self.assign(Tensor.normal(*self.shape, low=low, high=high)),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
}}

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
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  DispatchLog().__enter__()

# NOTE: patch torch optimizer step to avoid continously growing the computation graph
def realize_optimizer_step(optimizer: torch.optim.Optimizer, *args, **kwargs):
  tinygrad_tensors = []
  for param_group in optimizer.param_groups:
    for param in param_group["params"]:
      if param is None: continue
      tinygrad_tensors.append(param.data)
  for state_dict in optimizer.state.values():
    for key, value in state_dict.items():
      if torch.is_tensor(value) and str(value.device) == "tiny": tinygrad_tensors.append(value)
  Tensor.realize(*[unwrap(x) for x in tinygrad_tensors])

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init