from __future__ import annotations
from typing import List, Dict, Union
import importlib
from functools import lru_cache
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv, DEBUG
from tinygrad.dtype import ConstType, DType
from tinygrad.device import is_dtype_supported
from onnx import AttributeProto, ModelProto, TensorProto, TypeProto, ValueInfoProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  def tensor_dtype_to_np_dtype(tensor_dtype:int) -> np.dtype: return TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

cache_misses = 0
@lru_cache(None)
def _cached_to_python_const(t:Tensor, tobytes): return t.data().tobytes() if tobytes else t.tolist()

# Tensor -> python value cache for parameters
def to_python_const(t, tobytes=False) -> Union[List[ConstType], List[bytes], Union[ConstType, bytes]]:
  if not isinstance(t, Tensor): return t
  global cache_misses
  ret = _cached_to_python_const(t, tobytes)
  if (info := _cached_to_python_const.cache_info()).misses > cache_misses and DEBUG >= 3:
    print(f"Cache miss for {t}, {tobytes=}")
    cache_misses = info.misses
  return ret

# TODO: use real float16
# src: onnx/mapping.py
DTYPE_MAP: Dict[int, DType] = { TensorProto.FLOAT:dtypes.float32, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8,
  TensorProto.UINT16:dtypes.uint16, TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64,
  TensorProto.BOOL:dtypes.bool, TensorProto.FLOAT16:dtypes.float32, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32,
  TensorProto.UINT64:dtypes.uint64, TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float,
  TensorProto.FLOAT8E4M3FNUZ:dtypes.float, TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float}
def dtype_parse(onnx_dtype: TensorProto.DataType) -> DType:
  if onnx_dtype not in DTYPE_MAP: raise NotImplementedError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")
  return DTYPE_MAP[onnx_dtype] if is_dtype_supported(DTYPE_MAP[onnx_dtype]) else dtypes.float

# src: onnx/onnx_ml_pb2.pyi
ATTRIBUTE_MAP = {AttributeProto.FLOAT: lambda a: float(a.f), AttributeProto.INT: lambda a: int(a.i),
  AttributeProto.STRING: lambda a: a.s.decode("utf-8"), AttributeProto.TENSOR: lambda a: buffer_parse(a.t),
  AttributeProto.FLOATS: lambda a: tuple(float(x) for x in a.floats), AttributeProto.INTS: lambda a: tuple(int(x) for x in a.ints),
  AttributeProto.STRINGS: lambda a: tuple(x.decode("utf-8") for x in a.strings)}
def attribute_parse(onnx_attribute: AttributeProto):
  if onnx_attribute.type not in ATTRIBUTE_MAP:
    raise NotImplementedError(f"attribute with type {AttributeProto.AttributeType.Name(onnx_attribute.type)} is not supported")
  return ATTRIBUTE_MAP[onnx_attribute.type](onnx_attribute)

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
  tensors: Dict[str, Tensor] = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    tensors[inp.name] = buffer_parse(inp)

  # preparse the attributes
  attribute_dict = {}
  domain = ""
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = {x.name:attribute_parse(x) for x in n.attribute}
    if n.domain: domain = n.domain

  onnx_model_version = onnx_model.opset_import[0].version

  def run_onnx(inputs={}, debug=0):
    debug = getenv("DEBUGONNX") or debug
    input_tensors: Dict[str,Tensor|List[Tensor]] = {}
    intermediate_tensors: Dict[str,Tensor] = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    # get inputs
    for model_input in onnx_model.graph.input:
      name = model_input.name
      if name in tensors: continue
      shape = type_parse(model_input.type)
      if name in inputs:
        if isinstance(inputs[name], Tensor):
          input_tensors[name] = inputs[name]
        elif isinstance(inputs[name], list):
          input_tensors[name] = [Tensor(i, requires_grad=False) for i in inputs[name]]
        # TODO: this is just to make training tests pass, need a principled way to handle training vs non-training
        elif domain == "ai.onnx.preview.training":
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
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != "": return input_tensors[x]
      return None

    for num,n in enumerate(onnx_model.graph.node):
      inp: List[Tensor] = []
      if debug >= 3: print("inputs:")
      for x in n.input:
        t = fetch_tensor(x)
        if debug >= 3: print(f"\t{x} - {t}")
        inp.append(t)
      opt: Dict = attribute_dict[num]
      if debug >= 1: print(f"{num}: op {n.op_type} shape {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")

      # NOTE some ops live here because they require access to some local variables
      if n.op_type in onnx_ops.tensor_methods:
        ret = getattr(Tensor, n.op_type.lower())(*inp, **opt)

      elif n.op_type == "Split":
        axis, n_outputs  = opt.get('axis', 0), opt.get('num_outputs') or len(n.output)
        sz = inp[0].shape[axis]
        sizes = to_python_const(inp[1]) if len(inp) == 2 else [sz // n_outputs + (1 if i < sz % n_outputs else 0) for i in range(n_outputs)]
        ret = inp[0].split(sizes, axis)
      elif n.op_type == "Slice":
        inp += [list(v) for v in reversed(opt.values())] # axes, ends, starts -> starts, ends, axes
        starts, ends, axes, steps = inp[1], inp[2], inp[3] if len(inp)>=4 else list(range(inp[0].ndim)), inp[4] if len(inp)>=5 else [1]*inp[0].ndim
        starts, ends, axes, steps = (to_python_const(x) for x in (starts, ends, axes, steps))
        slices = [slice(0,x,1) for x in inp[0].shape]
        for i, axis in enumerate(axes): slices[axis] = slice(starts[i], ends[i], steps[i])
        ret = inp[0][tuple(slices)]
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

      if not isinstance(ret, tuple): ret = (ret, )
      assert len(n.output) <= len(ret), f"expected output size must be less than {len(ret)}, it's {n.output}"
      if debug >= 2: print([x.shape if isinstance(x, Tensor) else None for x in ret])
      if debug >= 2: print("outputs:")
      for i in range(len(n.output)):
        if debug >= 2: print(f"\t{n.output[i]} - {ret[i]}")
        intermediate_tensors[n.output[i]] = ret[i]
      if num == ONNXLIMIT:
        output_tensor_names = n.output
        break

    return {outp:intermediate_tensors[outp] for outp in output_tensor_names}
  return run_onnx
