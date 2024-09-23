from __future__ import annotations
from typing import List, Dict, Union
import importlib
from functools import lru_cache
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import getenv, DEBUG, CI, OSX
from tinygrad.dtype import ConstType, DType
from onnx import AttributeProto, ModelProto, TensorProto, TypeProto
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

# copied from helpers.py
def is_dtype_supported(dtype, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16: return False
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  if dtype == dtypes.half: return not (CI and device in {"GPU", "LLVM", "CUDA"})
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

# src: onnx/mapping.py  https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping
# not supported: STRING = 8 COMPLEX64 = 14, COMPLEX128 = 15, UINT4 = 21, INT4 = 22
# TODO: use dtypes.float16 for FLOAT16
DTYPE_MAP: Dict[TensorProto.DataType, DType] = {
  TensorProto.FLOAT:dtypes.float, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8, TensorProto.UINT16:dtypes.uint16,
  TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64, TensorProto.BOOL:dtypes.bool,
  TensorProto.FLOAT16:dtypes.float, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32, TensorProto.UINT64:dtypes.uint64,
  TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
  TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float
}

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

  def buffer_parse(inp: TensorProto) -> Tensor:
    if inp.data_type not in DTYPE_MAP:
      raise NotImplementedError(f"data type not supported {inp.name} {inp.dims} {inp.data_type}")
    dtype = DTYPE_MAP[inp.data_type] if is_dtype_supported(DTYPE_MAP[inp.data_type]) else dtypes.float32
    if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
      return Tensor(dat, dtype=dtype, requires_grad=False).reshape(tuple(inp.dims))
    if len(inp.raw_data) > 0:
      data = np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).astype(_to_np_dtype(dtype)).copy()
      return Tensor(data.reshape(tuple(inp.dims)), requires_grad=False)
    return Tensor(None, requires_grad=False)

  def attribute_parse(a: AttributeProto) -> float | int | str | Tensor | tuple[float] | tuple[int]:
    # TODO: this is not complete, see onnx/onnx_ml_pb2.pyi for a complete list
    if a.type == AttributeProto.FLOAT: return float(a.f)
    elif a.type == AttributeProto.INT: return int(a.i)
    elif a.type == AttributeProto.STRING: return a.s.decode("utf-8")
    elif a.type == AttributeProto.TENSOR: return buffer_parse(a.t) # TENSOR
    elif a.type == AttributeProto.FLOATS: return tuple(float(x) for x in a.floats)
    elif a.type == AttributeProto.INTS: return tuple(int(x) for x in a.ints)
    elif a.type == AttributeProto.STRINGS: return tuple(x.decode("utf-8") for x in a.strings)
    elif a.type == AttributeProto.GRAPH: raise NotImplementedError(f"graph not implemented: {a.g}\n likely an OP requiring control flow")
    else: raise RuntimeError(f"can't parse {a.type} {a}")

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
        elif domain == "ai.onnx.preview.training": # not sure if in real use the domain is "ai.onnx.preview.training"
          input_tensors[name] = Tensor(inputs[name], requires_grad=True) # TODO there isn't a good way to parse which inp requires_grad, some are manually turned off in optimizer ops
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
      # have to use n.output for cases when num_outputs is absent
      if n.op_type in onnx_ops.tensor_methods:
        ret = getattr(Tensor, n.op_type.lower())(*inp, **opt)
      elif n.op_type == "Split":
        axis = opt.get("axis", 0)
        split = None if len(inp) == 1 else to_python_const(inp[1])
        if split is None:
          split = [inp[0].shape[axis] // len(n.output)] * len(n.output)
          for i in range(inp[0].shape[axis] % len(n.output)):
            split[i] += 1
        i, ret = 0, []
        arg = [None] * inp[0].ndim
        for s in split:
          arg[axis] = (i,i+s)
          ret.append(inp[0].shrink(arg=tuple(arg)))
          i = i+s
        ret = tuple(ret)

      # need to check onnx_model_version
      elif n.op_type == "Slice":
        if onnx_model_version < 10:
          axes, ends, starts, steps = list(opt.get("axes", range(inp[0].ndim))), list(opt["ends"]), list(opt["starts"]), [1]*inp[0].ndim
        else:
          starts, ends = inp[1:3]
          axes = list(range(inp[0].ndim)) if len(inp) <= 3 else to_python_const(inp[3].cast(dtypes.int32))
          steps = inp[4].cast(dtypes.int32).tolist() if len(inp) > 4 else [1]*inp[0].ndim
          starts, ends = to_python_const(starts), to_python_const(ends)
        arg = [(0,x,1) for x in inp[0].shape]
        for i, axis in enumerate(axes):
          axis = int(axis) + inp[0].ndim if axis < 0 else int(axis)
          if starts[i] < 0: starts[i] += inp[0].shape[axis]
          if ends[i] < 0: ends[i] += inp[0].shape[axis]
          starts[i], ends[i] = max(0, min(starts[i], inp[0].shape[axis])), max(0, min(ends[i], inp[0].shape[axis]))
          if starts[i] > ends[i] and steps[i] >= 0: steps[i] = -steps[i]
          arg[axis] = (starts[i], ends[i], steps[i])
        new_shape = tuple((s, e) if st > 0 else (e+1, s+1) for s, e, st in arg)
        if any(s==e for s,e in new_shape): ret = inp[0].shrink(new_shape)
        else: ret = inp[0][tuple([slice(s,e,st) for s,e,st in arg])]

      # need to call backward on intermediate_tensors
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
