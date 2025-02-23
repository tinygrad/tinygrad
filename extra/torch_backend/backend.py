from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG, getenv
import torch, pathlib
torch.autograd.grad_mode.set_multithreading_enabled(False)

# https://pytorch.org/docs/stable/torch.compiler_ir.html

# TODO: don't replicate this in cpp
torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.uint8: dtypes.uint8,
  torch.int8: dtypes.int8,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
}

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x)
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
def as_strided(tensor, size, stride, storage_offset=None):
  if size == [] and storage_offset is not None:
    # TODO: is this right?
    return wrap(unwrap(tensor).flatten()[storage_offset:storage_offset+1].reshape(()))
  # broadcast
  if len(tensor.shape) == 0: return wrap(unwrap(tensor).reshape((1,)*len(size)).expand(size))
  print("******* NOTE: this as_strided is wrong ***********\n", tensor.shape, size, stride, storage_offset)
  return wrap(Tensor.zeros(*size))
  raise NotImplementedError("fix as_strided")

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout, device, pin_memory=False):
  if DEBUG >= 2: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=torch_to_tiny_dtype[dtype])
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if DEBUG >= 2: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=torch_to_tiny_dtype[dtype])
  return wrap(ret)

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  #print(f"{input.shape=} {weight.shape=} {bias.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
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
def cat_out(tensors, dim=0, *, out): unwrap(out).replace(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim), allow_shape_mismatch=True)

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y): return wrap(unwrap(x)[y[0].tolist()])

tiny_backend = {
  "aten.view": Tensor.reshape,
  "aten.add.Tensor": Tensor.add,
  "aten.sub.Tensor": Tensor.sub,
  "aten.mul.Tensor": Tensor.mul,
  "aten.div.Tensor": Tensor.div,
  "aten.add_.Tensor": lambda x,y: x.assign(x.add(y)),
  "aten.pow.Tensor_Scalar": Tensor.pow,
  "aten.pow.Tensor_Tensor": Tensor.pow,
  "aten.pow.Scalar": lambda x,y: y.pow(x, reverse=True),
  "aten.bitwise_and.Tensor": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor": Tensor.xor,
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.eq.Tensor": Tensor.eq, "aten.eq.Scalar": Tensor.eq,
  "aten.ne.Tensor": Tensor.ne, "aten.ne.Scalar": Tensor.ne,
  "aten.gt.Tensor": Tensor.__gt__, "aten.gt.Scalar": Tensor.__gt__,
  "aten.lt.Tensor": Tensor.__lt__, "aten.lt.Scalar": Tensor.__lt__,

  "aten.all": Tensor.all,
  "aten.all.out": lambda x, axis, keepdim, out: out.assign(x.all(axis, keepdim)),
  "aten.any": Tensor.any,
  "aten.any.out": lambda x, axis, keepdim, out: out.assign(x.any(axis, keepdim)),
  "aten.argmin": Tensor.argmin,
  "aten.argmax": Tensor.argmax,
  "aten._softmax": lambda x, dim, half_to_float: x.softmax(dim, dtypes.float if half_to_float else None),
  "aten._log_softmax": lambda x, dim, half_to_float: x.log_softmax(dim, dtypes.float if half_to_float else None),
  "aten._logcumsumexp": Tensor.logcumsumexp,

  "aten.abs": Tensor.abs,
  "aten.acos": Tensor.acos,
  "aten.acosh": Tensor.acosh,
  "aten.asin": Tensor.asin,
  "aten.asinh": Tensor.asinh,
  "aten.atan": Tensor.atan,
  "aten.atanh": Tensor.atanh,
  "aten.ceil": Tensor.ceil,
  "aten.cos": Tensor.cos,
  "aten.cosh": Tensor.cosh,
  "aten.erf": Tensor.erf,
  "aten.exp": Tensor.exp,
  "aten.exp2": Tensor.exp2,
  "aten.floor": Tensor.floor,
  "aten.hardsigmoid": Tensor.hardsigmoid,
  "aten.hardtanh": Tensor.hardtanh,
  "aten.max": Tensor.max,
  "aten.mean": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.mm": Tensor.matmul,
  "aten.neg": Tensor.neg,
  "aten.relu": Tensor.relu,
  "aten.rsqrt": Tensor.rsqrt,
  "aten.sgn": Tensor.sign,
  "aten.sigmoid": Tensor.sigmoid,
  "aten.sign": Tensor.sign,
  "aten.sin": Tensor.sin,
  "aten.sinh": Tensor.sinh,
  "aten.sqrt": Tensor.sqrt,
  "aten.tan": Tensor.tan,
  "aten.tanh": Tensor.tanh,
  "aten.trunc": Tensor.trunc,
}

# there's earlier things to hook here
#"aten.add.out": lambda x,y,out: out.replace(x+y, allow_shape_mismatch=True),
#"aten.abs.out": lambda x,out: out.replace(x.abs(), allow_shape_mismatch=True),
#"aten.ceil.out": lambda x,out: out.replace(x.ceil(), allow_shape_mismatch=True),
#"aten.exp2.out": lambda x,out: out.replace(x.exp2(), allow_shape_mismatch=True),

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    #print(k, len(args), kwargs.keys())
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    return wrap(f(*args, **kwargs))
  return nf

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

if getenv("TORCH_DEBUG"):
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  DispatchLog().__enter__()
