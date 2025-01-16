from typing import Callable, Any, Sequence
import importlib, functools, dataclasses, enum
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, DEBUG, all_same
from tinygrad.dtype import DType, ConstType, dtypes
from tinygrad.device import is_dtype_supported
def get_run_onnx():

# ***** protobuf parsing ******
from onnx import AttributeProto, ModelProto, TensorProto, TypeProto, helper
import numpy as np

def dtype_parse(onnx_dtype: int) -> DType:
  supported: dict[int, DType] = {
    TensorProto.FLOAT:dtypes.float32, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8,
    TensorProto.UINT16:dtypes.uint16, TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64,
    TensorProto.BOOL:dtypes.bool, TensorProto.FLOAT16:dtypes.float32, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32,
    TensorProto.UINT64:dtypes.uint64, TensorProto.BFLOAT16:dtypes.bfloat16,
  }
  unsupported = {
    TensorProto.UNDEFINED, TensorProto.STRING, TensorProto.COMPLEX64, TensorProto.COMPLEX128, TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2, TensorProto.FLOAT8E5M2FNUZ, TensorProto.UINT4, TensorProto.INT4
  }
  if onnx_dtype in unsupported: raise NotImplementedError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")
  return supported[onnx_dtype] if is_dtype_supported(supported[onnx_dtype]) else dtypes.float

def attribute_parse(onnx_attribute: AttributeProto):
  supported: dict[AttributeProto.AttributeType, Callable[[AttributeProto], Any]] = {
    AttributeProto.FLOAT: lambda a: float(a.f), AttributeProto.INT: lambda a: int(a.i),
    AttributeProto.STRING: lambda a: a.s.decode("utf-8"), AttributeProto.TENSOR: lambda a: buffer_parse(a.t),
    AttributeProto.FLOATS: lambda a: tuple(float(x) for x in a.floats), AttributeProto.INTS: lambda a: tuple(int(x) for x in a.ints),
    AttributeProto.STRINGS: lambda a: tuple(x.decode("utf-8") for x in a.strings)
  }
  unsupported = {
    AttributeProto.UNDEFINED, AttributeProto.GRAPH, AttributeProto.SPARSE_TENSOR, AttributeProto.TYPE_PROTO, AttributeProto.TENSORS,
    AttributeProto.GRAPHS, AttributeProto.SPARSE_TENSORS, AttributeProto.TYPE_PROTOS
  }
  if onnx_attribute.type in unsupported:
    raise NotImplementedError(f"attribute with type {AttributeProto.AttributeType.Name(onnx_attribute.type)} is not supported")
  return supported[onnx_attribute.type](onnx_attribute)

def buffer_parse(onnx_tensor: TensorProto) -> Tensor:
  if onnx_tensor.string_data: raise NotImplementedError("Parsing for buffer with string data is not implemented.")
  dtype, shape = dtype_parse(onnx_tensor.data_type), tuple(onnx_tensor.dims)
  if data := list(onnx_tensor.float_data) or list(onnx_tensor.int32_data) or list(onnx_tensor.int64_data) or list(onnx_tensor.double_data) or \
             list(onnx_tensor.uint64_data):
    if len(data) == 1: return Tensor(data[0], dtype=dtype).reshape(shape)
    return Tensor(data, dtype=dtype).reshape(shape).realize()
  if onnx_tensor.HasField("raw_data"):
    np_buffer = np.frombuffer(onnx_tensor.raw_data, dtype=helper.tensor_dtype_to_np_dtype(onnx_tensor.data_type)).copy().reshape(shape)
    if np_buffer.size == 1: return Tensor(np_buffer.item(), dtype=dtype).reshape(shape)
    return Tensor(np_buffer, dtype=dtype)
  return Tensor(None)

def type_parse(onnx_type: TypeProto):
  if is_optional := onnx_type.HasField("optional_type"): onnx_type = onnx_type.optional_type.elem_type
  if onnx_type.HasField("sequence_type"): value_type, tensor_type = ValueType.SEQUENCE, onnx_type.sequence_type.elem_type.tensor_type
  elif onnx_type.HasField("tensor_type"): value_type, tensor_type = ValueType.TENSOR, onnx_type.tensor_type
  else: raise NotImplementedError(f"Unsupported {onnx_type=}")
  shape, dtype = tuple(d.dim_param or d.dim_value for d in tensor_type.shape.dim), dtype_parse(tensor_type.elem_type)
  return OnnxValue(value_type, is_optional, shape, dtype)

# ***** onnx spec *****
class ValueType(enum.Enum):
  # supported
  TENSOR = enum.auto(); SEQUENCE = enum.auto()
  # unsupported
  MAP = enum.auto(); OPTIONAL = enum.auto(); SPARSE_TENSOR = enum.auto(); OPAQUE = enum.auto()

@dataclasses.dataclass
class OnnxValue:
  type: ValueType
  is_optional: bool
  shape: tuple[str|int]
  dtype: DType

@dataclasses.dataclass
class OnnxNode:
  num: int
  op: str
  inputs: tuple[str]
  outputs: tuple[str]
  opts: dict[str, Any]
  def __repr__(self): return f"{self.num}: op '{self.op}' inputs={self.inputs} opts={self.opts} outputs={self.outputs}"

# ***** python const *****
cache_misses = 0
@functools.lru_cache(None)
def _cached_to_python_const(t:Tensor):
  if t.dtype is dtypes.uint8: return t.data().tobytes()
  if 0 in t.shape: return []
  return t.tolist()

# Tensor -> python value cache for parameters
def to_python_const(t) -> list[ConstType]|ConstType|bytes:
  if not isinstance(t, Tensor): return t
  global cache_misses
  ret = _cached_to_python_const(t)
  if (info := _cached_to_python_const.cache_info()).misses > cache_misses and DEBUG >= 3:
    print(f"Cache miss for {t}")
    cache_misses = info.misses
  return ret

# ***** runner ******
debug = int(getenv("DEBUGONNX", "0"))
limit = int(getenv("ONNXLIMIT", "-1"))
class OnnxSession:
  # these values are expected to be python consts
  required_input_python_consts: dict[str, tuple[int, ...]] = {
    "Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
    "CumSum": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
    "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4),
    **{"Reduce"+r: (1,) for r in ("Max", "Min", "Sum", "Mean", "SumSquare", "Prod", "L1", "L2", "LogSum", "LogSumExp")},
    **{optim: (1,) for optim in ("Adam", "Adagrad", "Momentum")}
  }
  # TODO: move extra.onnx_ops here so we don't have to deal with annoying circular import
  # TODO: clean up opset stuff after moving extra.onnx_ops here
  onnx_ops_module = importlib.import_module('extra.onnx_ops')
  onnx_ops = {
    **{op: getattr(Tensor, op.lower()) for op in ("Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan",
    "Asin", "Acos", "Atan", "Relu", "Sigmoid", "MatMul", "Floor", "Ceil", "IsInf", "IsNaN", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh",
    "Tanh", "Softsign", "Asinh", "Acosh", "Atanh",  "Elu", "Celu", "Selu", "Xor", "Round", "Erf", "Mod")},
  }

  def __init__(self, model: ModelProto):
    # parse model protobuf
    self.is_training = any(n.HasField("domain") and n.domain == "ai.onnx.preview.training" for n in model.graph.node)
    Tensor.no_grad = False if self.is_training else True
    Tensor.training = True if self.is_training else False
    self.values = {x.name:buffer_parse(x) for x in model.graph.initializer}
    self.input_spec = {x.name:type_parse(x.type) for x in model.graph.input if x.name not in self.values}
    self.output_spec = {x.name:type_parse(x.type) for x in model.graph.output}
    self.nodes = tuple(OnnxNode(num, n.op_type, tuple(n.input), tuple(n.output), {x.name:attribute_parse(x) for x in n.attribute})
                       for num,n in enumerate(model.graph.node))
    self.opset_version = model.opset_import[0].version
    self.variable_dims = {}

  def _parse_input(self, name: str, value: Any, spec: OnnxValue):
    if spec.is_optional and value is None: return None
    if spec.type is ValueType.SEQUENCE:
      if not isinstance(value, Sequence): raise RuntimeError(f"{name} received {value}, expected a sequence type")
      sequence = [Tensor(v, dtype=spec.dtype, requires_grad=self.is_training) if not isinstance(v, Tensor) else v for v in value]
      if not all_same(tuple(t.shape for t in sequence)): raise RuntimeError(f"Shapes for {name} sequence must be homogeneous")
      # TODO: check for var dim
      # TODO: need true float16 for dtype checking
      return sequence
    if spec.type is ValueType.TENSOR:
      tensor = Tensor(value, dtype=spec.dtype, requires_grad=self.is_training) if not isinstance(value, Tensor) else value
      # TODO: need true float16 for dtype checking
      for dim, (onnx_dim, user_dim_input) in enumerate(zip(spec.shape, tensor.shape, strict=True)):
        if isinstance(onnx_dim, str):
          onnx_dim = self.variable_dims[onnx_dim] if onnx_dim in self.variable_dims else self.variable_dims.setdefault(onnx_dim, user_dim_input)
        if user_dim_input != onnx_dim:
          raise RuntimeError(f"{name} has mismatch on {dim=}. Expected {onnx_dim}, received {user_dim_input}.")
      return tensor
    raise NotImplementedError(f"{name} was not parsed properly")

  def _dispatch_op(self, op, inps, opts):
    if op in self.onnx_ops: return self.onnx_ops[op](*inps, **opts)
    if hasattr(self.onnx_ops_module, op):
      fxn = getattr(self.onnx_ops_module, op)
      if isinstance(fxn, dict):
        for k in sorted(fxn.keys()):
          if k <= self.opset_version:
            real_fxn = fxn[k]
      else: real_fxn = fxn
      return real_fxn(*inps, **opts)
    raise NotImplementedError(f"{op=} not supported")

  def run(self, inputs={}, debug=debug, limit=limit):
    for name, input_spec in self.input_spec.items():
      if name not in inputs: raise RuntimeError(f"Please provide input data for {name}")
      self.values[name] = self._parse_input(name, inputs[name], input_spec)

    for node in self.nodes:
      inps = [self.values.get(name) for name in node.inputs]
      opts, op = node.opts, node.op

      # provide additional opts
      if op == "Split" and 'num_outputs' not in opts: opts['num_outputs'] = len(node.outputs)
      if op == "Gradient": opts['intermediate_tensors'] = self.values

      required_consts = self.required_input_python_consts.get(op, ())
      inps = [to_python_const(t) if i in required_consts else t for i,t in enumerate(inps)]

      if debug >= 1: print(node)
      if debug >= 2: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {i!r}" for x,i in zip(node.inputs, inps)))
      ret = self._dispatch_op(op, inps, opts)
      ret = ret if isinstance(ret, tuple) else (ret,)
      self.values.update(dict(zip(node.outputs, ret[:len(node.outputs)], strict=True)))

      if node.num == limit: return {name:self.values[name] for name in node.outputs}
    return {name:self.values[name] for name in self.output_spec}