from __future__ import annotations
from typing import List, Dict, Union, Callable, Any
import importlib, functools
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv, DEBUG
from tinygrad.dtype import DType, ConstType
from tinygrad.device import is_dtype_supported
from onnx import AttributeProto, ModelProto, TensorProto, TypeProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  def tensor_dtype_to_np_dtype(tensor_dtype:int) -> np.dtype: return TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

cache_misses = 0
@functools.lru_cache(None)
def _cached_to_python_const(t:Tensor):
  if t.dtype is dtypes.uint8: return t.data().tobytes()
  if 0 in t.shape: return []
  return t.tolist()

# Tensor -> python value cache for parameters
def to_python_const(t) -> Union[List[ConstType], List[bytes], Union[ConstType, bytes]]:
  if not isinstance(t, Tensor): return t
  global cache_misses
  ret = _cached_to_python_const(t)
  if (info := _cached_to_python_const.cache_info()).misses > cache_misses and DEBUG >= 3:
    print(f"Cache miss for {t}")
    cache_misses = info.misses
  return ret

# TODO: use real float16
# src: onnx/mapping.py
DTYPE_MAP: Dict[TensorProto.DataType | int, DType] = {
  TensorProto.FLOAT:dtypes.float32, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8,
  TensorProto.UINT16:dtypes.uint16, TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64,
  TensorProto.BOOL:dtypes.bool, TensorProto.FLOAT16:dtypes.float32, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32,
  TensorProto.UINT64:dtypes.uint64, TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float,
  TensorProto.FLOAT8E4M3FNUZ:dtypes.float, TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float
}
def dtype_parse(onnx_dtype: TensorProto.DataType | int) -> DType:
  if onnx_dtype not in DTYPE_MAP: raise NotImplementedError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")
  return DTYPE_MAP[onnx_dtype] if is_dtype_supported(DTYPE_MAP[onnx_dtype]) else dtypes.float

# src: onnx/onnx_ml_pb2.pyi
ATTRIBUTE_MAP: Dict[AttributeProto.AttributeType, Callable[[AttributeProto], Any]] = {
  AttributeProto.FLOAT: lambda a: float(a.f), AttributeProto.INT: lambda a: int(a.i),
  AttributeProto.STRING: lambda a: a.s.decode("utf-8"), AttributeProto.TENSOR: lambda a: buffer_parse(a.t),
  AttributeProto.FLOATS: lambda a: tuple(float(x) for x in a.floats), AttributeProto.INTS: lambda a: tuple(int(x) for x in a.ints),
  AttributeProto.STRINGS: lambda a: tuple(x.decode("utf-8") for x in a.strings)
}
def attribute_parse(onnx_attribute: AttributeProto):
  if onnx_attribute.type not in ATTRIBUTE_MAP:
    raise NotImplementedError(f"attribute with type {AttributeProto.AttributeType.Name(onnx_attribute.type)} is not supported")
  return ATTRIBUTE_MAP[onnx_attribute.type](onnx_attribute)

def buffer_parse(inp: TensorProto) -> Tensor:
  if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
    return Tensor(dat, dtype=dtype_parse(inp.data_type), requires_grad=False).reshape(tuple(inp.dims))
  if len(inp.raw_data) > 0:
    return Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).copy().reshape(tuple(inp.dims)),
                  dtype=dtype_parse(inp.data_type), requires_grad=False)
  raise NotImplementedError(f"buffer with data type {TensorProto.DataType.Name(inp.data_type)} is not supported")

onnx_ops = importlib.import_module('extra.onnx_ops')
ONNXLIMIT = getenv("ONNXLIMIT", -1)

def get_run_onnx(onnx_model: ModelProto):
  def type_parse(type_proto: TypeProto):
    ret = []
    while True:
      attr = type_proto.WhichOneof('value')
      if attr == 'tensor_type':
        if "dim_value" not in type_proto.tensor_type.shape.dim.__dir__(): return () # variable type, unable to determine shape
        elif not ret:
          return tuple([x.dim_value for x in type_proto.tensor_type.shape.dim])
        else:
          ret.extend([(x.dim_value,) for x in type_proto.tensor_type.shape.dim])
          return tuple(ret)
      elif attr == 'sequence_type':
        type_proto = getattr(type_proto, attr).elem_type
        ret.append(1)
      elif attr == 'optional_type': type_proto = getattr(type_proto, attr).elem_type
      elif attr == 'map_type': raise NotImplementedError(f"map_type is not implemented: {type_proto}")
      elif attr == 'opaque_type': raise NotImplementedError(f"opaque_type is not implemented: {type_proto}")
      elif attr == 'sparse_tensor_type': raise NotImplementedError(f"sparse_tensor_type is not implemented: {type_proto}")
      else: raise AttributeError(f"unknown attr: {attr}, {type_proto}")

  # initialization data
  model_parameters = {inp.name:buffer_parse(inp) for inp in onnx_model.graph.initializer}
  model_attributes = {num:{x.name:attribute_parse(x) for x in n.attribute} for num,n in enumerate(onnx_model.graph.node)}

  # model specs
  is_onnx_preview_training = any(n.HasField("domain") and n.domain == "ai.onnx.preview.training" for n in onnx_model.graph.node)
  onnx_model_version = onnx_model.opset_import[0].version

  # mapping from onnx ops to tensor.py ops
  tensor_methods = {
    op:op.lower() for op in ("Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
    "Relu", "Sigmoid", "MatMul", "Floor", "Ceil", "IsInf", "IsNaN", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh", "Tanh",
    "Softsign", "Asinh", "Acosh", "Atanh",  "Elu", "Celu", "Selu", "Xor", "Round", "Erf")
  }

  def run_onnx(inputs={}, debug=0):
    debug = getenv("DEBUGONNX") or debug
    input_tensors: Dict[str,Tensor|List[Tensor]] = {}
    intermediate_tensors: Dict[str,Tensor] = {}

    # get inputs
    for model_input in onnx_model.graph.input:
      name = model_input.name
      if name in model_parameters: continue
      shape = type_parse(model_input.type)
      if name in inputs:
        if isinstance(inputs[name], Tensor):
          input_tensors[name] = inputs[name]
        elif isinstance(inputs[name], list):
          input_tensors[name] = [Tensor(i, requires_grad=False) for i in inputs[name]]
        # TODO: this is just to make training tests pass, need a principled way to handle training vs non-training
        elif is_onnx_preview_training:
          input_tensors[name] = Tensor(inputs[name], requires_grad=True)
        else:
          input_tensors[name] = Tensor(inputs[name], requires_grad=False)
        if shape: # if only input_tensor is not variable type
          ts = input_tensors[name]
          input_shape = ts.shape if isinstance(ts, Tensor) else (1, *[i.shape for i in ts])
          assert input_shape == shape, f"wrong shape for input {name}, {input_shape} isn't {shape}"
      else:
        raise RuntimeError(f"no data for {name} with shape {shape}")

    def fetch_tensor(x: str):
      if x in model_parameters: return model_parameters[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != "": return input_tensors[x]
      return None

    for num,n in enumerate(onnx_model.graph.node):
      inp = [fetch_tensor(x) for x in n.input]
      opt = model_attributes[num]

      if debug >= 1: print(f"{num}: op \"{n.op_type}\" input shapes {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")
      if debug >= 3: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {t}" for i,(x,t) in enumerate(zip(n.input, inp))))

      if n.op_type in tensor_methods:
        ret = getattr(Tensor, tensor_methods[n.op_type])(*inp, **opt)

      # NOTE some ops live here because they require access to some local variables
      elif n.op_type == "Split":
        axis, n_outputs  = opt.get('axis', 0), opt.get('num_outputs') or len(n.output)
        sz = inp[0].shape[axis]
        sizes = to_python_const(inp[1]) if len(inp) == 2 else [sz // n_outputs + (1 if i < sz % n_outputs else 0) for i in range(n_outputs)]
        ret = inp[0].split(sizes, axis)
      elif n.op_type == "Gradient":
        assert len(opt["xs"]) == len(inp), f"len(opt['xs']):{len(opt['xs'])}, len(inp):{len(inp)} output and input has to match"
        y = opt["y"]
        intermediate_tensors[y].backward()
        ret = tuple([t.grad for t in inp])

      # onnx_ops.py
      elif hasattr(onnx_ops, n.op_type):
        fxn = getattr(onnx_ops, n.op_type)
        if isinstance(fxn, dict):
          for k in sorted(fxn.keys()):
            if k <= onnx_model_version:
              real_fxn = fxn[k]
        else:
          real_fxn = fxn
        ret = real_fxn(*inp, **opt)
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise NotImplementedError(f"op_type {n.op_type} not supported")

      # finalization after running the op
      if not isinstance(ret, tuple): ret = (ret, )
      if len(n.output) > len(ret): raise RuntimeError(f"expected output size must be less than {len(ret)}, it's {n.output}")
      for i in range(len(n.output)): intermediate_tensors[n.output[i]] = ret[i]
      if debug >= 2: print("\toutputs:\n" + "\n".join(f"\t\t{n.output[i]} - {ret[i]}" for i in range(len(n.output))))

      if num == ONNXLIMIT: return {name:intermediate_tensors[name] for name in n.output}
    return {x.name:intermediate_tensors[x.name] for x in onnx_model.graph.output}
  return run_onnx
