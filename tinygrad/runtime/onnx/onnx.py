from __future__ import annotations
from typing import List, Dict, Union, Tuple
import importlib, functools
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import getenv, DEBUG, CI, OSX
from tinygrad.dtype import ConstType, DType
from onnx import AttributeProto, ModelProto, TensorProto, ValueInfoProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  def tensor_dtype_to_np_dtype(tensor_dtype:int) -> np.dtype: return TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

# TODO: yeah idk
# class TinyOnnx:
#   def __init__(self, onnx_model, training=False) -> None:
#     self.training = training

# ========== helpers
# Tensor -> python value cache for arguments
cache_misses = 0
@functools.lru_cache(None)
def _cached_to_python_const(t:Tensor): return t.data().tobytes() if t.dtype is dtypes.uint8 else t.tolist()
def to_python_const(t) -> Union[List[ConstType], List[bytes], Union[ConstType, bytes]]:
  if not isinstance(t, Tensor): return t
  global cache_misses
  ret = _cached_to_python_const(t)
  if (info := _cached_to_python_const.cache_info()).misses > cache_misses and DEBUG >= 3:
    print(f"Cache miss for {t}")
    cache_misses = info.misses
  return ret

# copied from helpers.py
def is_dtype_supported(dtype, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16: return False
  if dtype == dtypes.half: return not (CI and device in {"GPU", "LLVM", "CUDA"})
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  # if device in ["WEBGPU"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32] # lol
  return True

# ======= parsers
# src: onnx/mapping.py  https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping
# TODO: these float8 are kinda cursed. May run into subtle bugs passing them as float.....
# TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
# TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float
# TODO: check if dtypes.void can be used like this
DTYPE_MAP: Dict[int, DType] = {
  TensorProto.FLOAT:dtypes.float, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8, TensorProto.UINT16:dtypes.uint16,
  TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64, TensorProto.BOOL:dtypes.bool,
  TensorProto.FLOAT16:dtypes.float16, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32, TensorProto.UINT64:dtypes.uint64,
  TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
  TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float, TensorProto.UNDEFINED:dtypes.void}
def dtype_parse(onnx_dtype: int) -> DType:
  if (ret := DTYPE_MAP.get(onnx_dtype)) is None: raise RuntimeError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")
  return ret

def buffer_parse(inp: TensorProto) -> Tensor:
  dtype = dtype_parse(inp.data_type) if is_dtype_supported(DTYPE_MAP[inp.data_type]) else dtypes.float32
  if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
    return Tensor(dat, dtype=dtype, requires_grad=False).reshape(tuple(inp.dims))
  if len(inp.raw_data) > 0:
    # return Tensor(inp.raw_data, dtype=dtype, requires_grad=False).reshape(tuple(inp.dims)) # TODO REINTRODUCES REGRESSION AGAIN
    data = np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).astype(_to_np_dtype(dtype)).copy()
    return Tensor(data.reshape(tuple(inp.dims)), requires_grad=False)
  return Tensor(None, requires_grad=False)

# src: onnx/onnx_ml_pb2.pyi
# NOTE: this is not a complete list
# torch's parser at onnx2torch/onnx_node.py: `OnnxNode._parse_attribute_value()`
ATTRS_MAP = {AttributeProto.FLOAT: lambda a: float(a.f), AttributeProto.INT: lambda a: int(a.i), AttributeProto.STRING: lambda a: a.s.decode("utf-8"),
         AttributeProto.TENSOR: lambda a: buffer_parse(a.t), AttributeProto.FLOATS: lambda a: tuple(float(x) for x in a.floats),
         AttributeProto.INTS: lambda a: tuple(int(x) for x in a.ints), AttributeProto.STRINGS: lambda a: tuple(x.decode("utf-8") for x in a.strings)}
def attribute_parse(a: AttributeProto):
  if (ret := ATTRS_MAP.get(a.type, lambda _: None)(a)) is None: raise NotImplementedError(f"{a.type} not implemented")
  return ret

# ========== runner
ONNXLIMIT = getenv("ONNXLIMIT", -1)

# onnx_ops implemented methods
onnx_ops = importlib.import_module('tinygrad.runtime.onnx.onnx_ops')

# tensor methods
exact_tensor_methods = {"Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan", "Relu", "Sigmoid", "MatMul",
  "Floor", "Ceil", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh", "Tanh", "Softsign", "Asinh", "Acosh", "Atanh", "Elu", "Celu", "Xor",
  "Round", "Softmax"}

# tensor methods with different names
equivalent_tensor_methods = {"Less": "__lt__", "Greater": "__gt__", "LessOrEqual": "__le__", "GreaterOrEqual": "__ge__", "Equal": "__eq__",
  "LogSoftmax": "log_softmax", "Not": "logical_not", "Tile":"repeat", "Range": "arange", "NegativeLogLikelihoodLoss": "nll_loss"}

# tensor methods with different argument names
equivalent_tensor_methods_exceptions = {"Concat": ("cat", {"axis": "dim"}), "LeakyRelu": ("leakyrelu", {"alpha": "neg_slope"}),
  "Selu": ("selu", {"gamma": "scale"})}

# lambda methods lol
# lambda_methods = {"Identity": lambda x:x, "Sub": lambda x,y:x-y, }

def get_run_onnx(onnx_model: ModelProto):
  # get weights and biases
  tensors = {inp.name:buffer_parse(inp) for inp in onnx_model.graph.initializer}

  # preparse the attributes
  attributes = {num:{x.name:attribute_parse(x) for x in n.attribute} for num,n in enumerate(onnx_model.graph.node)}

  def run_onnx(inputs={}, debug=0, initialization_hook=None, op_hook=None):
    if initialization_hook: initialization_hook(tensors, attributes)
    debug = getenv("DEBUGONNX") or debug

    # get inputs
    def parse_input(model_input:ValueInfoProto):
      if model_input.name not in inputs: raise RuntimeError(f"no data for {model_input=}")
      # NOTE: when elem_type is 0 which maps to UNDEFINED DataType (void), the type_proto does not provide enough information to verify input shape
      if isinstance(inp := inputs[model_input.name], list):
        ret = [Tensor(i, requires_grad=False) for i in inp]
        if (dtype := dtype_parse(model_input.type.sequence_type.elem_type.tensor_type.elem_type)) is not dtypes.void:
          # the element_type of tensor_type of a sequence_type determines the dtype for all tensors in the sequence
          assert all(t.dtype is dtype for t in ret), f"parsed dtype {dtype}, input dtype {[t.dtype for t in ret]}"
        return ret
      inp = inp if isinstance(inp, Tensor) else Tensor(inp, requires_grad=False)
      if (dtype := dtype_parse(model_input.type.tensor_type.elem_type)) is not dtypes.void:
        assert dtype is inp.dtype, f"parsed dtype {dtype} input dtype {inp.dtype}"
        assert (shape:=tuple(d.dim_value for d in model_input.type.tensor_type.shape.dim))==inp.shape, f"parsed shape {shape} input shape{inp.shape}"
      return inp
    input_tensors = {model_input.name: parse_input(model_input) for model_input in onnx_model.graph.input if model_input.name not in tensors}
    intermediate_tensors: Dict[str,Tensor] = {}

    def fetch_tensor(x: str):
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != "": return input_tensors[x]
      return None

    # inputs we need to turn into a python const to compute
    to_python_const_inps: Dict[str, Tuple[int, ...]] = \
    {"Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
      "CumSum": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
      "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4)}

    for num,n in enumerate(onnx_model.graph.node):
      # prepare
      tensor_inp, opt = [fetch_tensor(x) for x in n.input], attributes[num]
      if debug >= 1: print(f"{num}: op \"{n.op_type}\" input shapes {[x.shape if isinstance(x, Tensor) else x for x in tensor_inp]} opt {opt}")
      # to python consts
      # TODO what if we just don't buffer parse some inps into Tensor to start with so no need waste time to python const
      if debug >= 3: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {t}" + (" -> *python const*" if i in to_python_const_inps.get(n.op_type,()) else "")
                                                      for i,(x,t) in enumerate(zip(n.input, tensor_inp))))
      inp = [to_python_const(x) if i in to_python_const_inps.get(n.op_type, ()) else x for i,x in enumerate(tensor_inp)]

      # tensor methods
      if n.op_type in exact_tensor_methods: ret = getattr(Tensor, n.op_type.lower())(*inp, **opt)
      elif n.op_type in equivalent_tensor_methods: ret = getattr(Tensor, equivalent_tensor_methods[n.op_type])(*inp, **opt)
      elif n.op_type in equivalent_tensor_methods_exceptions:
        rewrite = equivalent_tensor_methods_exceptions[n.op_type]
        ret = getattr(Tensor, rewrite[0])(*inp, **{rewrite[1].get(k, k):v for k,v in opt.items()})

      # NOTE some ops live here because they require access to some local variables
      # have to use n.output for cases when num_outputs is absent
      elif n.op_type == "Split":
        axis, outputs, split = opt.get("axis", 0), opt.get("num_outputs") or len(n.output), opt.get("split")
        # split is provided
        if len(inp) == 2: ret = inp[0].split(inp[1], axis)
        if len(inp) == 1 and split is not None: ret = inp[0].split(split, axis)
        # split has to be inferred
        size = inp[0].shape[axis]
        if len(inp) == 1 and split is None: ret = inp[0].split([size // outputs + (1 if i < size % outputs else 0) for i in range(outputs)], axis)

      # need to check onnx_model_version
      elif n.op_type == "Slice":
        # only onnx_model_version < 10 has opt, we just unload the opt into inp to match other versions
        if opt: inp.extend([list(v) for v in reversed(opt.values())]) # axes, ends, starts -> starts, ends, axes
        (data, starts, ends), axes, steps = inp[:3], inp[3] if len(inp) > 3 else list(range(inp[0].ndim)), inp[4] if len(inp) > 4 else [1]*inp[0].ndim
        slices = [slice(0,x,1) for x in data.shape]
        for i, axis in enumerate(axes): slices[axis] = slice(starts[i], ends[i], steps[i])
        ret = data[slices]

      # need to call backward on intermediate_tensors
      elif n.op_type == "Gradient":
        intermediate_tensors[opt["y"]].backward()
        ret = tuple([t.grad for t in inp])

      # onnx_ops.py
      elif hasattr(onnx_ops, n.op_type): ret = getattr(onnx_ops, n.op_type)(*inp, **opt)
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise NotImplementedError(f"op_type {n.op_type} not supported")

      if not isinstance(ret, tuple): ret = (ret, )
      assert len(n.output) <= len(ret), f"expected output size must be less than {len(ret)}, it's {n.output}"
      for i in range(len(n.output)): intermediate_tensors[n.output[i]] = ret[i]
      if debug >= 2: print("\toutputs:\n" + "\n".join(f"\t\t{n.output[i]} - {ret[i]}" for i in range(len(n.output))))

      if op_hook: op_hook(num, n, tuple(tensor_inp), opt, ret)

      if num == ONNXLIMIT: return {name:intermediate_tensors[name] for name in n.output}

    return {x.name:intermediate_tensors[x.name] for x in onnx_model.graph.output}
  return run_onnx
