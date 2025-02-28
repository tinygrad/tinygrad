from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG, getenv, prod
import torch.lib
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator, functools
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype
from tinygrad.ops import Ops

# https://pytorch.org/docs/stable/torch.compiler_ir.html

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x, _to_torch_dtype(x.dtype))
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_initialized(self): return True
  def is_available(self): return True
  def current_device(self): return 0
  def _is_in_bad_fork(self): return False
  def manual_seed_all(self, seed: int): Tensor.manual_seed(seed)
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()

# *** bad functions on CPU ***

@torch.library.impl("aten::masked_select", "privateuseone")
def masked_select(self, mask):
  # err, bad
  return wrap(Tensor(self.cpu().numpy()[mask.cpu().numpy()]))

@torch.library.impl("aten::topk", "privateuseone")
def topk(self, k, dim=-1, largest=True, sorted=True):
  # TODO: move to tinygrad
  t1, t2 = torch.topk(self.cpu(), k, dim, largest, sorted)
  return torch.return_types.topk((t1.tiny(), t2.tiny()))

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  return aten._index_put_impl_(self.cpu(), [x.cpu() for x in indices], values.cpu(), accumulate, unsafe).tiny()

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return aten.index(x.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in y]).tiny()

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None): out.copy_(torch.randperm(n, generator=generator, device="cpu").tiny())

# *** end bad functions on CPU ***

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

@functools.lru_cache(None)
def cached_to_movement_ops(shape, st) -> list:
  mops = to_movement_ops(st)
  if mops[0] == (MovementOps.RESHAPE, shape): mops = mops[1:]
  return mops

from tinygrad.shape.shapetracker import ShapeTracker, View
from extra.to_movement_ops import to_movement_ops, apply_mop, MovementOps
@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  # TODO: this is heavyweight
  st = ShapeTracker((View.create(tuple(tensor.shape)), View.create(tuple(size), tuple(stride), 0 if storage_offset is None else storage_offset)))
  ret = unwrap(tensor)
  if prod(size) == 1: return wrap(ret.flatten()[storage_offset].reshape(size))
  if TORCH_DEBUG >= 1: print("**** as_strided", tensor.shape, size, stride, st)
  for mo in cached_to_movement_ops(tuple(tensor.shape), st): ret = apply_mop(ret, mo)
  return wrap(ret)

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
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  # TODO: support return_indices in tinygrad
  ret = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
  # TODO: this is wrong
  return (wrap(ret), wrap(Tensor.zeros_like(ret, dtype=dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:Tensor, self:Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  if stride is not None and len(stride) == 0: stride = None
  # TODO: utilize input indices once they are correct
  grad_out, self = unwrap(grad_out), unwrap(self)
  out = Tensor.max_pool2d(self, kernel_size, stride, dilation, padding, ceil_mode)
  return wrap(out.gradient(self, gradient=grad_out)[0])

@torch.library.impl("aten::arange", "privateuseone")
def arange(end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(0, end, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::arange.start", "privateuseone")
def arange_start(start, end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::arange.start_step", "privateuseone")
def arange_start_step(start, end, step, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, step, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  if not transposed:
    return wrap(unwrap(input).conv2d(unwrap(weight), unwrap(bias) if bias is not None else None,
                                   groups=groups, stride=stride, dilation=dilation, padding=padding))
  return wrap(unwrap(input).conv_transpose2d(unwrap(weight), unwrap(bias) if bias is not None else None,
                                   groups=groups, stride=stride, dilation=dilation, padding=padding))

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out), unwrap(input), unwrap(weight), Tensor.zeros(weight.shape[0])
  if not transposed:
    out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else:
    out = Tensor.conv_transpose2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

@torch.library.impl("aten::slice.Tensor", "privateuseone")
def slice_tensor(self, dim=0, start=None, end=None, step=1):
  # TODO: Do we need more dims?
  # TODO: rewrite this...
  print(f"{self.shape=}")
  if self.ndim == 1:
    return wrap(unwrap(self)[start:end:step])
  elif self.ndim == 2:
    if dim == 0:
      return wrap(unwrap(self)[start:end:step, :])
    else:
      return wrap(unwrap(self)[:, start:end:step])
  else:
    if dim == 0:
      return wrap(unwrap(self)[start:end:step, :, :])
    elif dim == 1:
      return wrap(unwrap(self)[:, start:end:step, :])
    else:
      return wrap(unwrap(self)[:, :, start:end:step])

# @torch.library.impl("aten::convolution_backward", "privateuseone")
# TODO: fix this....
def convolution_backward(grad_out, input, weight, bias=None, stride=1, padding=0, dilation=1, transposed=False, output_padding=0, groups=1, output_mask=None):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {bias=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out), unwrap(input), unwrap(weight), Tensor.zeros(*bias)
  if not transposed:
    out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else:
    out = Tensor.conv_transpose2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src, dest, non_blocking=False):
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
def cat_out(tensors, dim=0, out=None):
  return wrap(unwrap(out).replace(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim), allow_shape_mismatch=True))

# register some decompositions
from torch._decomp import get_decompositions
aten = torch.ops.aten
decomps = [
  aten.native_batch_norm, aten.native_batch_norm_backward,
  aten.native_layer_norm_backward,
  aten.addmm,
  aten.addcmul,
  aten.addcdiv,
  aten._log_softmax_backward_data,
  aten.threshold_backward,
  aten.softplus_backward,
  aten.elu,  # elu has a scale + input_scale param
  aten.elu_backward,
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
  aten.index_select,
  aten.native_dropout, aten.native_dropout_backward,
  aten._softmax_backward_data, aten.embedding_dense_backward,
  aten.linalg_vector_norm,
  aten.binary_cross_entropy, aten.binary_cross_entropy_backward,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.cumprod,
  aten.eye,
  aten.binary_cross_entropy,
  aten.binary_cross_entropy_backward,
  aten.hardsigmoid_backward,
  aten.logical_or,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
  # aten.slice_backward,
  # aten.upsample_linear1d,
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
]
for k,v in get_decompositions(decomps).items():
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
  "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.mul.out": operator.mul,
  "aten.bmm.out": operator.matmul,
  "aten.leaky_relu.out": Tensor.leaky_relu,
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.pow.Scalar_out": lambda x,y: x**y,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": Tensor.bitwise_xor,
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  # TODO: support this in tinygrad
  "aten.bitwise_left_shift.Tensor_out": lambda self, other: self.cast(dtypes.uint) << other,
  "aten.bitwise_right_shift.Tensor_out": lambda self, other: self.cast(dtypes.uint) >> other,
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
  "aten.scatter.value_out": Tensor.scatter,
  "aten.where.self_out": Tensor.where,
}}

# we add the "out" here
def wrap_out(f):
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    return out.replace(assigned)
  return _wrap_out

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,  # when are views unsafe, and do we care?
  "aten.remainder.Scalar_Tensor": lambda x,y: x%y,
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": lambda x,y: x.assign(x//y),
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": lambda x,y: x.assign(x*(2**y)),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": lambda x,y: x.assign(x//(2**y)),
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
  "aten.var_mean.correction": Tensor.var_mean,
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,dtype=None,out=None:
    out.replace(Tensor.sum(self, axis if axis is None or len(axis) else None, keepdim), allow_shape_mismatch=True),
  "aten.scatter.value": Tensor.scatter,
  # my changes
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.scatter.src": Tensor.scatter, # This might be wrong??
  # ==== 
  "aten.gather": lambda self, dim, index: Tensor.gather(self, dim, index.cast(dtypes.int)),
  "aten.where.self": Tensor.where, # NOTE: this is needed as well as the out type
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: Tensor.log_softmax(self,dim),
  "aten.random_": lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype)),
  "aten.random_.from": lambda self, from_, to:
    self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype)),
  "aten.uniform_": lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high)),
  "aten.normal_": lambda self, mean=0, std=1: self.assign(Tensor.normal(*self.shape, mean=mean, std=std)),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
  "aten.masked_fill_.Scalar": lambda self,mask,value: self.assign(mask.where(self, value)),
  "aten.masked_fill_.Tensor": lambda self,mask,value: self.assign(mask.where(self, value)),
  "aten.multinomial": Tensor.multinomial,
  # my changes start here:
  "aten.all": Tensor.all,
  "aten.sgn": Tensor.sign,
  "aten.acos": Tensor.acos,
  "aten.any": Tensor.any,
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.asinh": Tensor.asinh,
  "aten.mul": Tensor.mul,
  "aten.atanh": Tensor.atanh,
  "aten.fill_.Tensor": Tensor.full,
  "aten.flip": Tensor.flip,
  "aten.scatter_add": lambda self, dim, index, src: Tensor.scatter_reduce(self, dim, index, src, reduce='sum'),
  "aten.scatter_add.out": lambda self, dim, index, src, out: out.replace(Tensor.scatter_reduce(self, dim, index, src, reduce='sum')),
  "aten.scatter_reduce.two": lambda self, dim, index, src, reduce, include_self=True: Tensor.scatter_reduce(self, dim, index, src, reduce=reduce, include_self=include_self),
  "aten.avg_pool2d": lambda self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True: Tensor.avg_pool2d(self, kernel_size, stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
  # "aten.avg_pool2d_backward": lambda grad_out, input, kernel_size, stride=None, padding=1, ceil_mode=False, count_include_pad=True, divisor_override=None: Tensor.avg_pool2d(input, kernel_size, stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad).backward(grad_out),
  "aten.avg_pool3d": lambda self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True: Tensor.avg_pool2d(self, kernel_size, stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
  # "aten.convolution": lambda self, weight, bias=None, stride=1, padding=0, dilation=1, transposed=False, output_padding=1, groups=1: Tensor.conv2d(self, weight, bias, groups, stride, dilation, padding) if not transposed else Tensor.conv_transpose2d(self, weight, bias, groups, stride, dilation, padding),
  "aten.cummax": Tensor.cummax,
  "aten.upsample_nearest1d": lambda self, size: Tensor.interpolate(self, size, mode="nearest"),
  "aten.upsample_nearest1d_backward": lambda self, size, gradient: Tensor.interpolate(self, size, mode="nearest").backward(Tensor(gradient)),
  "aten.roll": Tensor.roll,
  "aten.where.self_out": lambda self, x, y, out: out.replace(Tensor.where(self, x, y)),
  "aten.where.self": lambda self, x, y: Tensor.where(self, x, y),
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.prod.int_out": lambda self, dim, out: out.replace(Tensor.prod(self, axis=dim)),
  "aten.constant_pad_nd": lambda self, padding, value=0.0: Tensor.pad(self, padding, mode="constant", value=value),
  # "aten.slice.Tensor": lambda self, dim=0, start=None, end=None, step=1: self[:, start:end:step] if dim else self[start:end:step, :],
  "aten.ones_like": lambda self, **kwargs: Tensor.ones_like(self),
  "aten.logsumexp": lambda self, axis, keepdim=False: Tensor.logsumexp(self, *axis, keepdim=keepdim),
  "aten.prod": lambda self: Tensor.prod(self),
  "aten.prod.int_out": lambda self, dim, out: out.replace(Tensor.prod(self, dim)),
  "aten.repeat": Tensor.repeat,
  "aten.split.Tensor": Tensor.split,
  "aten.lerp.Tensor": Tensor.lerp,
  "aten.expand": Tensor.expand,
  "aten.index_put": Tensor.assign,
  "aten.mul.Tensor": Tensor.mul,
  # "aten.add.Tensor": Tensor.add,
  # "aten.logical_or": Tensor.__bool__,
  # "aten.logical_or.out": lambda self, s, out: out.replace(Tensor.__bool__(self)),
  # "aten.nonzero": Tensor.nonzero,
  # "aten.eye": lambda n, **kwargs: Tensor.eye(n),
  # "aten.eye.m_out": lambda n, m, out: out.replace(Tensor.eye(m, n)),
  # "aten.std": Tensor.std,
  # "aten.mean": Tensor.mean,
  # "aten.squeeze.dim": lambda self,dim: Tensor.squeeze(self, dim),
  # "aten.unsqueeze": Tensor.unsqueeze,
  # "aten.slice_backward": lambda self: self,
  # "aten.amax": Tensor.argmax,
  # "aten.upsample_linear1d": Tensor.interpolate,
  # "aten.upsample_linear1d": lambda self,size,align: Tensor.interpolate(self, size, mode="linear", align_corners=align),
  # "aten.upsample_linear1d_backward.grad_input": lambda self, input_size, op_size, align, grad_input: Tensor.interpolate(self, op_size, mode="linear", align_corners=align).backward(grad_input),
  "aten.reflection_pad2d": functools.partial(Tensor.pad, mode="reflect"),
}}

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG:
      print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
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
      if torch.is_tensor(value): tinygrad_tensors.append(value)
  real_tinygrad_tensors = [unwrap(x) for x in tinygrad_tensors if str(x.device) == "tiny"]
  if len(real_tinygrad_tensors): Tensor.realize(*real_tinygrad_tensors)

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init