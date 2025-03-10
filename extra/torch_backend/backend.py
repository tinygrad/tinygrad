# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv, prod
import torch.lib
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator, functools, inspect
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")])
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

# in place operations with views
def is_view(self: torch.Tensor) -> bool: return getattr(self, "_base", None) is not None
def realize_with_views(self: torch.Tensor, views: list[torch.Tensor]):
  assert self.device.type == "tiny"
  self = unwrap(self)
  if not self.lazydata.st.contiguous: raise ValueError("base of view must be contiguous") # TODO: support?
  self.replace(self.clone().realize())
  for v in views:
    v = unwrap(v)
    ret = self
    st = ShapeTracker(self.lazydata.st.views + v.lazydata.st.views) # TODO: is this right?
    for mo in cached_to_movement_ops(self.shape, st): ret = apply_mop(ret, mo)
    v.replace(ret)
def maybe_realize_storage(self: torch.Tensor) -> bool:
  if realize:=is_view(self): realize_with_views(self._base, [self]) # TODO: other views could exist
  return realize
def inplace_fn(outvars: str|list[str]):
  if type(outvars) is str: outvars = [outvars]
  def decorator(fn):
    sig = inspect.signature(fn)
    def wrapper(*args, **kwargs):
      bound = sig.bind(*args, **kwargs)
      outs = [kwargs.get(v, bound.arguments.get(v)) for v in outvars]
      realize = any(maybe_realize_storage(o) for o in outs)
      ret = fn(*args, **kwargs)
      if realize: Tensor.realize(*(unwrap(o) for o in outs))
      return ret
    return wrapper
  return decorator

# *** bad functions on CPU ***

@torch.library.impl("aten::masked_select", "privateuseone")
def masked_select(self, mask):
  # err, bad
  return wrap(Tensor(self.cpu().numpy()[mask.cpu().numpy()]))

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
@inplace_fn("self")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  ret = aten._index_put_impl_(self.cpu(), [x.cpu() if isinstance(x, torch.Tensor) else None for x in indices], values.cpu(), accumulate, unsafe).tiny()
  return wrap(unwrap(self).assign(unwrap(ret)))

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return aten.index(x.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in y]).tiny()

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None): out.copy_(torch.randperm(n, generator=generator, device="cpu").tiny())

# *** end bad functions on CPU ***

@torch.library.impl("aten::zero_", "privateuseone")
@inplace_fn("x")
def zero_(x):
  if TORCH_DEBUG: print(f"zero_ {x.shape}")
  tt = unwrap(x)
  # NOTE: unconditional contiguous covers if x is contiguous (match it) or if x is view (realize for inplace)
  # TODO: consolidate
  tt.assign(tt.zeros_like().contiguous())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
@inplace_fn("x")
def fill_scalar(x, y):
  if TORCH_DEBUG: print(f"fill_.Scalar {x.shape} {y}")
  tt = unwrap(x)
  tt.assign(tt.full_like(y).contiguous())

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
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())).contiguous()
  return wrap(ret)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  # TODO: support return_indices in tinygrad
  ret = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
  # TODO: this is wrong
  return (wrap(ret), wrap(Tensor.zeros_like(ret, dtype=dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:torch.Tensor, self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  if stride is not None and len(stride) == 0: stride = None
  # TODO: utilize input indices once they are correct
  self_ = unwrap(self)
  out = Tensor.max_pool2d(self_, kernel_size, stride, dilation, padding, ceil_mode)
  return wrap(out.gradient(self_, gradient=unwrap(grad_out))[0])

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
  return wrap(unwrap(input).conv2d(unwrap(weight), unwrap(bias) if bias is not None else None,
                                   groups=groups, stride=stride, dilation=dilation, padding=padding))

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out), unwrap(input), unwrap(weight), Tensor.zeros(weight.shape[0])
  out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

def upsample(self, size, align_corners=False, mode=None): return wrap(Tensor.interpolate(unwrap(self), size, mode=mode, align_corners=align_corners))
for i,pre in enumerate(["", "bi", "tri"]):
  torch.library.impl(f"aten::upsample_{pre}linear{i+1}d", "privateuseone")(functools.partial(upsample, mode="linear"))
  torch.library.impl(f"aten::upsample_nearest{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest"))
  torch.library.impl(f"aten::_upsample_nearest_exact{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest-exact"))

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  realize = str(dest.device) == "tiny" and maybe_realize_storage(dest)
  cast_dtype = _from_torch_dtype(dest.dtype)
  if str(src.device) == "tiny" and str(dest.device) == "tiny":
    unwrap(dest).assign(unwrap(src).cast(cast_dtype))
    if realize: Tensor.realize(unwrap(dest))
  elif str(src.device) == "tiny" and str(dest.device) == "cpu":
    # TODO: is there a better way?
    dest.resize_(src.numel()).resize_(src.shape)
    dest.copy_(torch.from_numpy(unwrap(src).cast(cast_dtype).numpy()))
  elif str(src.device) == "cpu" and str(dest.device) == "tiny":
    unwrap(dest).assign(Tensor(src.numpy()).cast(cast_dtype))
    if realize: Tensor.realize(unwrap(dest))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
@inplace_fn("out")
def cat_out(tensors, dim=0, out=None):
  unwrap(out).assign(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim))

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
  aten.upsample_nearest2d.out,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.randint,
  aten.eye,
  aten.hardsigmoid_backward,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
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
  "sigmoid", "clamp", "mish", "erf", "leaky_relu",
  # trig
  "acos", "acosh", "cos", "cosh", "asin", "asinh", "sin", "sinh", "atan", "atanh", "tan", "tanh",
  # rounding
  "ceil", "round", "floor", "trunc",
  # binary
  "mul", "div", "maximum", "minimum", "copysign",
  # modify
  "tril", "triu",
  # reduce
  "all", "any", "argmax", "argmin", "cumsum",
  # complex
  "avg_pool2d", "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.div.out_mode": Tensor.div,
  "aten.mul.out": operator.mul,
  "aten.bmm.out": operator.matmul,
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.pow.Scalar_out": lambda input,exponent: input**exponent,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": Tensor.bitwise_xor,
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  "aten.clamp_max.Tensor_out": lambda input,max_: input.clamp(max_=max_),
  "aten.clamp_min.Tensor_out": lambda input,min_: input.clamp(min_=min_),
  "aten.fmod.Tensor_out": lambda input,other: input-input.div(other, rounding_mode="trunc")*other,
  # TODO: this might result in overflow issues
  "aten.round.decimals_out": lambda self,decimals: (self*10**decimals).round()/10**decimals,
  # TODO: support this in tinygrad
  "aten.bitwise_left_shift.Tensor_out": lambda x,y: x*(2**y),
  "aten.bitwise_right_shift.Tensor_out": lambda x,y: x//(2**y),
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  "aten.fmax.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.maximum(input, other))),
  "aten.fmin.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.minimum(input, other))),
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
  "aten.scatter.value_out": Tensor.scatter,
  "aten.where.self_out": Tensor.where,
  "aten.prod.int_out": Tensor.prod,
  "aten.scatter_add.out": functools.partial(Tensor.scatter_reduce, reduce='sum'),
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,dtype=None:
    self.sum(axis if axis is None or len(axis) else None, keepdim,
                         dtype = _from_torch_dtype(dtype) if dtype is not None else None),
}}

# we add the "out" here
def wrap_out(f):
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    if out.lazydata.is_realized: assigned = assigned.contiguous() # TODO: how does this map to torch's semantics
    return out.assign(assigned)
  return _wrap_out

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,  # when are views unsafe, and do we care?
  "aten.remainder.Scalar_Tensor": lambda x,y: x%y,
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": inplace_fn("x")(lambda x,y: x.assign(x//y)),
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x*(2**y))),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x//(2**y))),
  # relu doesn't have an out form?
  "aten.relu": Tensor.relu,
  "aten.relu_": inplace_fn("x")(lambda x: x.assign(x.relu())),
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.mm": Tensor.matmul,
  "aten.mv": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.prod": Tensor.prod,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,
  "aten.std_mean.correction": Tensor.std_mean,
  "aten.var.correction": Tensor.var,
  "aten.var_mean.correction": Tensor.var_mean,
  "aten.scatter.value": Tensor.scatter,
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.gather": lambda self, dim, index: self.gather(dim, index.cast(dtypes.int)),
  "aten.where.self": Tensor.where, # NOTE: this is needed as well as the out type
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),
  "aten.random_": inplace_fn("self")(lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype))),
  "aten.random_.from": inplace_fn("self")(lambda self, from_, to:
    self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype))),
  "aten.uniform_": inplace_fn("self")(lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high))),
  "aten.normal_": inplace_fn("self")(lambda self, mean=0, std=1: self.assign(Tensor.normal(*self.shape, mean=mean, std=std))),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
  "aten.logical_or_": inplace_fn("x")(lambda x, y: x.assign(x | y)),
  "aten.multinomial": Tensor.multinomial,
  "aten.pad": Tensor.pad,
  "aten.reflection_pad2d": functools.partial(Tensor.pad, mode="reflect"),
  "aten.masked_fill_.Scalar": inplace_fn("self")(lambda self, mask, value: self.assign(self.masked_fill(mask, value))),
  "aten.masked_fill.Scalar": Tensor.masked_fill,
  "aten.masked_fill.Tensor": Tensor.masked_fill,
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
  "aten.fill_.Tensor": Tensor.full, # TODO: looks wrong
  "aten.flip": Tensor.flip,
  "aten.scatter_reduce.two": Tensor.scatter_reduce,
  "aten.squeeze_.dim": lambda self, dim: self.replace(self.squeeze(dim), allow_shape_mismatch=True),
  "aten.add.Tensor": lambda input,other,alpha=1: input+alpha*other,
  "aten.linspace": lambda start, stop, steps, dtype=None, **kwargs:
    Tensor.linspace(start, stop, steps, **({"dtype": _from_torch_dtype(dtype)} if dtype is not None else {})),
  "aten.topk": Tensor.topk,
  "aten::view.dtype": lambda self, dtype: self.bitcast(_from_torch_dtype(dtype)),
  "aten.constant_pad_nd": lambda self, padding, value=0.0: self.pad(padding, mode="constant", value=value),
  "aten.logsumexp": lambda self, axis, keepdim=False: self.logsumexp(axis[0], keepdim=keepdim),
  "aten.squeeze.dim": Tensor.squeeze,
  "aten.unsqueeze": Tensor.unsqueeze,
  "aten.roll": Tensor.roll,
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.repeat": Tensor.repeat,
  "aten.lerp.Tensor": Tensor.lerp,
  "aten.expand": Tensor.expand,
  "aten.t": Tensor.transpose,
  "aten.detach": Tensor.detach,
  "aten.max.dim": lambda self, dim, keepdim=False: (self.max(dim, keepdim), self.argmax(dim, keepdim).cast(dtype=dtypes.int64))
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
  def nf2(*args, **kwargs):
    return inplace_fn("out")(nf)(*args, **kwargs) if "out" in kwargs else nf(*args, **kwargs)
  return nf2

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  (_dispatch_log:=DispatchLog()).__enter__() # NOTE: must be kept alive

# NOTE: patch torch optimizer step to avoid continously growing the computation graph
def realize_optimizer_step(optimizer: torch.optim.Optimizer, *args, **kwargs):
  tinygrad_tensors = []
  for param_group in optimizer.param_groups:
    for param in param_group["params"]:
      if param is None: continue
      tinygrad_tensors.append(param.data)
  for state_dict in optimizer.state.values():
    for _, value in state_dict.items():
      if torch.is_tensor(value): tinygrad_tensors.append(value)
  real_tinygrad_tensors = [unwrap(x) for x in tinygrad_tensors if str(x.device) == "tiny"]
  if len(real_tinygrad_tensors): Tensor.realize(*real_tinygrad_tensors)

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init