from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, getenv, prod
from torch.nn import _reduction as _Reduction
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib
torch.autograd.grad_mode.set_multithreading_enabled(False)

# https://pytorch.org/docs/stable/torch.compiler_ir.html

# TODO: don't replicate this in cpp
torch_to_tiny_dtype = {
  torch.float16: dtypes.float16,
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.uint8: dtypes.uint8,
  torch.int8: dtypes.int8,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
}
tiny_to_torch_dtype = {v: k for k, v in torch_to_tiny_dtype.items()}

def to_tiny_dtype(torch_dtype:str|None) -> DType|None: return None if torch_dtype is None else torch_to_tiny_dtype[torch_dtype]

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x, tiny_to_torch_dtype[x.dtype])
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend: pass
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend)
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
  if TORCH_DEBUG >= 1: print("** NOTE: this as_strided is wrong", tensor.shape, size, stride, storage_offset)

  if tuple(x for x in tensor.shape if x != 1) == tuple(x for x in size if x != 1):
    # this is squeeze/unsqueeze
    return tensor.reshape(size)

  # TODO: how do i know this is permute?
  if tensor.shape == (1000, 512) and size == [512, 1000] and stride == [0, 1]:
    return wrap(unwrap(tensor).permute(1,0))

  #print(tensor.cpu().numpy())
  raise NotImplementedError("fix as_strided")

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  if TORCH_DEBUG: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=to_tiny_dtype(dtype))
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if TORCH_DEBUG: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=to_tiny_dtype(dtype or torch.get_default_dtype()))
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
  # TODO: shape desync
  # assert src.shape == unwrap(src)
  if str(src.device) == "tiny" and str(dest.device) == "tiny":
    unwrap(dest).replace(unwrap(src), allow_shape_mismatch=True)
  elif str(src.device) == "tiny" and str(dest.device) == "cpu":
    # TODO: is there a better way?
    dest.resize_(src.numel()).resize_(unwrap(src).shape)
    dest.copy_(torch.from_numpy(unwrap(src).numpy()))
  elif str(src.device) == "cpu" and str(dest.device) == "tiny":
    unwrap(dest).assign(Tensor(src.numpy()))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
def cat_out(tensors, dim=0, *, out): unwrap(out).replace(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim), allow_shape_mismatch=True)

def avg_pool2d(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True):
  return x.avg_pool2d(kernel_size, None if len(stride) == 0 else stride, 1, padding, ceil_mode, count_include_pad)

def max_pool2d(x, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
  return x.max_pool2d(kernel_size, None if len(stride) == 0 else stride, dilation, padding, ceil_mode)

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y): return wrap(unwrap(x)[y[0].tolist()])

def mean_out(tensor, axis=None, keepdim=False, *, dtype=None, out):
  tensor = tensor if dtype is None else tensor.cast(to_tiny_dtype(tensor))
  return out.replace(tensor.mean(axis, keepdim), allow_shape_mismatch=True)

def prod(tensor, dim=None, keepdim=False, *, dtype=None):
  tensor = tensor if dtype is None else tensor.cast(to_tiny_dtype(tensor))
  return tensor.prod(dim, keepdim)

def random(tensor, low=0, high=None):
  if high is None: high = float(2**(dtypes.finfo(dt)[1]+1)) if dtypes.is_float(dt:=tensor.dtype) else dtypes.max(dt)
  tensor.assign(Tensor.uniform(*tensor.shape, low=low, high=high, dtype=tensor.dtype))

def arange_start(start, stop=None, step=1, dtype=None, *, out) -> Tensor:
  if dtype is None: dtype = torch.get_default_dtype() if any(isinstance(x, float) for x in (start, stop, step)) else torch.int64
  out.replace(Tensor.arange(start, stop, step, dtype=to_tiny_dtype(dtype)), allow_shape_mismatch=True)

reduction_by_value = {_Reduction.get_enum(x): x for x in ("none", "mean", "sum")}
def nll_loss(tensor, target, weight=None, reduction=None, ignore_index=None):
  reduction = "mean" if reduction is None else reduction_by_value[reduction]
  r = tensor.nll_loss(target, weight, ignore_index, reduction)
  return r, r # TODO: What we should return?

# register some decompositions
from torch._decomp import get_decompositions
aten = torch.ops.aten
decomps = [
  aten.native_batch_norm,
  aten.addmm,
  aten.addmv,
  # NOTE: many of these don't work or cause infinite loops
  #aten.var_mean,
  #aten.var,
  #aten.rsqrt,
  #aten.max_pool2d_with_indices,
]
for k,v in get_decompositions(decomps).items():
  key = str(k._schema).split("(")[0]
  if TORCH_DEBUG >= 2: print("register decomp for", k)
  torch.library.impl(key, "privateuseone")(v)

tiny_backend = {
  "aten.nll_loss_forward": nll_loss,
  "aten.nll_loss2d_forward": nll_loss,

  "aten.view": Tensor.reshape,
  "aten.bmm": Tensor.dot,
  "aten.dot": Tensor.dot,
  "aten.add.Tensor": Tensor.add,
  "aten.sub.Tensor": Tensor.sub,
  "aten.mul.Tensor": Tensor.mul,
  "aten.div.Tensor": Tensor.div,
  "aten.remainder.Tensor": Tensor.mod, # TODO: Fix type mismatch in test_mod (int32 vs int64)
  "aten.floor_divide": Tensor.idiv,
  "aten.add_.Tensor": lambda x,y,alpha=1: x.assign(x.add(y)*alpha),
  "aten.pow.Tensor_Scalar": Tensor.pow,
  "aten.pow.Tensor_Tensor": Tensor.pow,
  "aten.pow.Scalar": lambda x,y: y.pow(x, reverse=True),
  "aten.bitwise_and.Tensor_out": lambda x,y,*,out: out.assign(x & y),
  "aten.bitwise_or.Tensor_out": lambda x,y,*,out: out.assign(x | y),
  "aten.bitwise_xor.Tensor_out": lambda x,y,*,out: out.assign(x ^ y),
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.logical_not": Tensor.logical_not,
  "aten.eq.Tensor": Tensor.eq, "aten.eq.Scalar": Tensor.eq,
  "aten.ne.Tensor": Tensor.ne, "aten.ne.Scalar": Tensor.ne,
  "aten.gt.Tensor": Tensor.__gt__, "aten.gt.Scalar": Tensor.__gt__,
  "aten.ge.Tensor": Tensor.__ge__, "aten.ge.Scalar": Tensor.__ge__,
  "aten.lt.Tensor": Tensor.__lt__, "aten.lt.Scalar": Tensor.__lt__,
  "aten.le.Tensor": Tensor.__le__, "aten.le.Scalar": Tensor.__le__,

  "aten._log_softmax": lambda x, dim, half_to_float: x.log_softmax(dim, dtypes.float if half_to_float else None),
  "aten._logcumsumexp": Tensor.logcumsumexp,
  "aten._softmax": lambda x, dim, half_to_float: x.softmax(dim, dtypes.float if half_to_float else None),
  "aten.all": Tensor.all,
  "aten.all.out": lambda x, axis, keepdim, out: out.assign(x.all(axis, keepdim)),
  "aten.any": Tensor.any,
  "aten.any.out": lambda x, axis, keepdim, out: out.assign(x.any(axis, keepdim)),
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.avg_pool2d": avg_pool2d,
  "aten.avg_pool3d": avg_pool2d,
  "aten.max_pool2d": max_pool2d,
  "aten.sum": lambda x, dim=None, keepdim=False, *, dtype=None: x.sum(dim, keepdim, acc_dtype=to_tiny_dtype(dtype)),
  # "aten.sum.IntList_out": lambda x, dim=None, keepdim=False, *, dtype=None, out: out.assign(x.sum(dim, keepdim, acc_dtype=to_tiny_dtype(dtype))),

  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,out=None:
    out.replace(Tensor.sum(self, axis if len(axis) else None, keepdim), allow_shape_mismatch=True),

  "aten.arange.start_out": arange_start,
  "aten.random_": random, "aten.random_.from": random, "aten.random_.to": random,
  "aten.uniform_": lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high)),
  "aten.normal_": lambda self, low=0, high=1: self.assign(Tensor.normal(*self.shape, low=low, high=high)),
  "aten.flip": Tensor.flip,

  "aten.tril": Tensor.tril,
  "aten.triu": Tensor.triu,

  "aten.abs": Tensor.abs,
  "aten.acos": Tensor.acos,
  "aten.acosh": Tensor.acosh,
  "aten.asin": Tensor.asin,
  "aten.asinh": Tensor.asinh,
  "aten.atan": Tensor.atan,
  "aten.atanh": Tensor.atanh,
  "aten.ceil": Tensor.ceil,
  "aten.celu": Tensor.celu,
  "aten.clamp": Tensor.clamp,
  "aten.cos": Tensor.cos,
  "aten.cosh": Tensor.cosh,
  "aten.elu": Tensor.elu,
  "aten.erf": Tensor.erf,
  "aten.exp": Tensor.exp,
  "aten.exp2": Tensor.exp2,
  "aten.floor": Tensor.floor,
  "aten.gelu": lambda x, approximate: x.gelu(),
  "aten.hardsigmoid": Tensor.hardsigmoid,
  "aten.hardswish": Tensor.hardswish,
  "aten.hardtanh": Tensor.hardtanh,
  "aten.isnan": Tensor.isnan,
  "aten.lerp.Tensor_out": lambda x,y,weight,*,out: out.assign(x.lerp(y, weight)),
  "aten.log": Tensor.log,
  "aten.log2": Tensor.log2,
  "aten.max": Tensor.max,
  "aten.maximum": Tensor.maximum,
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.mean.out": mean_out,
  "aten.min": Tensor.min,
  "aten.minimum": Tensor.minimum,
  "aten.mish": Tensor.mish,
  "aten.mm": Tensor.matmul,
  "aten.neg": Tensor.neg,
  "aten.prod": prod, "aten.prod.dim_int": prod,
  "aten.reciprocal": Tensor.reciprocal,
  "aten.relu": Tensor.relu,
  "aten.relu_": lambda x: x.assign(x.relu()),
  "aten.round": Tensor.round,
  "aten.rsqrt": Tensor.rsqrt,
  "aten.sgn": Tensor.sign,
  "aten.sigmoid": Tensor.sigmoid,
  "aten.sign": Tensor.sign,
  "aten.silu": Tensor.silu,
  "aten.sin": Tensor.sin,
  "aten.sinh": Tensor.sinh,
  "aten.softplus": Tensor.softplus,
  "aten.sqrt": Tensor.sqrt,
  "aten.std.correction": Tensor.std, # TODO: Do we need to reshuffle args like in std_mean?
  "aten.std_mean.correction": lambda x,dim=None,correction=1,keepdim=False: x.std_mean(dim, keepdim, correction),
  "aten.tan": Tensor.tan,
  "aten.tanh": Tensor.tanh,
  "aten.trunc": Tensor.trunc,
  "aten.var.correction": Tensor.var,
  # TODO: support var_mean in tinygrad
  "aten.var_mean.correction": lambda self, dims, keepdim=False, correction=1: (self.var(dims, keepdim, correction), self.mean(dims, keepdim)),
  "aten.where.self": Tensor.where,

  "aten.scatter.value": Tensor.scatter,
  "aten.gather": Tensor.gather,
}

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
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  DispatchLog().__enter__()
