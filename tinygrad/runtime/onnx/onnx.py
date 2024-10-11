from __future__ import annotations
from typing import List, Dict, Union, Tuple, cast
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
def _cached_to_python_const(t:Tensor): return t.data().tobytes() if t.dtype is dtypes.uint8 else t.tolist()

# Tensor -> python value cache for parameters
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

# src: onnx/mapping.py  https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping
# not supported: STRING = 8 COMPLEX64 = 14, COMPLEX128 = 15, UINT4 = 21, INT4 = 22
# TODO: use dtypes.float16 for FLOAT16
DTYPE_MAP: Dict[int, DType] = {
  TensorProto.FLOAT:dtypes.float, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8, TensorProto.UINT16:dtypes.uint16,
  TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64, TensorProto.BOOL:dtypes.bool,
  TensorProto.FLOAT16:dtypes.float, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32, TensorProto.UINT64:dtypes.uint64,
  TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
  TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float
}

onnx_ops = importlib.import_module('tinygrad.runtime.onnx.onnx_ops')

ONNXLIMIT = getenv("ONNXLIMIT", -1)

def get_run_onnx(onnx_model: ModelProto):
  def type_parse(type_proto: TypeProto):
    ret: Union[List[int], List[Union[int, Tuple[int]]]] = []
    while True:
      attr = type_proto.WhichOneof('value')
      if attr == 'tensor_type':
        if not hasattr(type_proto.tensor_type.shape.dim, 'dim_value'): return () # variable type, unable to determine shape
        elif not ret: # tensor
          return tuple([x.dim_value for x in type_proto.tensor_type.shape.dim])
        else: # list of tensors
          ret += [(x.dim_value,) for x in type_proto.tensor_type.shape.dim]
          return tuple(ret)
      elif attr == 'sequence_type':
        type_proto = getattr(type_proto, 'sequence_type').elem_type
        ret.append(1)
      elif attr == 'optional_type': type_proto = getattr(type_proto, attr).elem_type
      else: NotImplementedError(f"{type_proto=} is not implemented")

  def buffer_parse(inp: TensorProto) -> Tensor:
    if inp.data_type not in DTYPE_MAP: raise NotImplementedError(f"data type not supported {inp.name} {inp.dims} {inp.data_type}")
    dtype = DTYPE_MAP[inp.data_type] if is_dtype_supported(DTYPE_MAP[inp.data_type]) else dtypes.float32
    if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
      return Tensor(dat, dtype=dtype, requires_grad=False).reshape(tuple(inp.dims))
    if len(inp.raw_data) > 0:
      # return Tensor(inp.raw_data, dtype=dtype, requires_grad=False).reshape(tuple(inp.dims)) # TODO REINTRODUCES REGRESSION AGAIN
      data = np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).astype(_to_np_dtype(dtype)).copy()
      return Tensor(data.reshape(tuple(inp.dims)), requires_grad=False)
    return Tensor(None, requires_grad=False)

  def attribute_parse(a: AttributeProto):
    # TODO: this is not complete, see onnx/onnx_ml_pb2.pyi for a complete list
    if a.type == AttributeProto.FLOAT: return float(a.f)
    elif a.type == AttributeProto.INT: return int(a.i)
    elif a.type == AttributeProto.STRING: return a.s.decode("utf-8")
    elif a.type == AttributeProto.TENSOR: return buffer_parse(a.t) # TENSOR
    elif a.type == AttributeProto.FLOATS: return tuple(float(x) for x in a.floats)
    elif a.type == AttributeProto.INTS: return tuple(int(x) for x in a.ints)
    elif a.type == AttributeProto.STRINGS: return tuple(x.decode("utf-8") for x in a.strings)
    elif a.type == AttributeProto.GRAPH: raise NotImplementedError(f"graph not implemented: {a.g}")
    else: raise RuntimeError(f"can't parse {a.type} {a}")

  tensors: Dict[str, Tensor] = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer: tensors[inp.name] = buffer_parse(inp)

  # preparse the attributes
  attribute_dict = {}
  domain = ""
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = {x.name:attribute_parse(x) for x in n.attribute}
    if n.domain: domain = n.domain

  def run_onnx(inputs={}, debug=0):
    debug = getenv("DEBUGONNX") or debug
    input_tensors: Dict[str,Union[Tensor, List[Tensor]]] = {}
    intermediate_tensors: Dict[str,Tensor] = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    # get inputs
    for model_input in onnx_model.graph.input:
      name = model_input.name
      if name in tensors: continue
      shape = type_parse(model_input.type)
      if name in inputs:
        if isinstance(inputs[name], Tensor): input_tensors[name] = inputs[name]
        elif isinstance(inputs[name], list): input_tensors[name] = [Tensor(i, requires_grad=False) for i in inputs[name]]
        # not sure if in real use the domain is "ai.onnx.preview.training"
        # TODO there isn't a good way to parse which inp requires_grad, some are manually turned off in optimizer ops
        elif domain == "ai.onnx.preview.training": input_tensors[name] = Tensor(inputs[name], requires_grad=True)
        else: input_tensors[name] = Tensor(inputs[name], requires_grad=False)
        if shape: # if only input_tensor is not variable type
          ts: Union[Tensor, List[Tensor]] = input_tensors[name]
          if isinstance(ts, Tensor): input_shape: Union[Tuple[int, ...], Tuple[Union[int, Tuple[int, ...],]]] = cast(Tuple[int, ...], ts.shape)
          # sequence type of list of tensors
          elif isinstance(ts, list): input_shape = (1,) + tuple(i.shape for i in ts)  # type: ignore
          else: assert False
          assert input_shape == shape, f"wrong shape for input {name}, {input_shape} isn't {shape}"
      else: raise RuntimeError(f"no data for {name} with shape {shape}")

    def fetch_tensor(x: str):
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != "": return input_tensors[x]
      return None

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

    # TODO what if we just don't buffer parse some inps or attributes so no need waste time to python const
    # inputs we need to turn into a python const
    to_python_const_inps: Dict[str, Tuple[int, ...]] = \
    {"Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
      "CumSum": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
      "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4)}

    for num,n in enumerate(onnx_model.graph.node):
      inp = []
      if debug >= 3: print("inputs:")
      for x in n.input:
        t = fetch_tensor(x)
        if debug >= 3: print(f"\t{x} - {t}")
        inp.append(t)
      opt: Dict = attribute_dict[num]
      if debug >= 1: print(f"{num}: op {n.op_type} shape {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")

      # to_python_consts
      inp = [to_python_const(x) if i in to_python_const_inps.get(n.op_type, ()) else x for i,x in enumerate(inp)]

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
        arg = [slice(0,x,1) for x in data.shape]
        for i, axis in enumerate(axes): arg[axis] = slice(starts[i], ends[i], steps[i])
        ret = data[arg]

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
