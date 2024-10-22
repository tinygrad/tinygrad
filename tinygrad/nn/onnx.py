from __future__ import annotations
from typing import Dict, Tuple, Sequence, Callable, Any
import importlib, functools
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv, CI, OSX
from tinygrad.dtype import DType
from onnx import AttributeProto, ModelProto, TensorProto, ValueInfoProto
# TODO try to remove this np stuff later
import numpy as np
from tinygrad.tensor import _to_np_dtype
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  def tensor_dtype_to_np_dtype(tensor_dtype:int) -> np.dtype: return TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

# ========== helpers
# Tensor -> python value cache for arguments
@functools.lru_cache(None)
def to_python_const(t:Tensor): return t if not isinstance(t, Tensor) else t.data().tobytes() if t.dtype is dtypes.uint8 else t.tolist()

# copied from helpers.py
def supported_device_dtypes(dtype, device):
  if dtype is dtypes.bfloat16: return dtypes.default_float
  if dtype is dtypes.half and (CI and device in {"GPU", "LLVM", "CUDA"}): return dtypes.default_float
  if dtype is dtypes.float64 and (device == "METAL" or (OSX and device == "GPU")): return dtypes.default_float
  return dtype

# ======= parsers
# src: onnx/mapping.py  https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping
DTYPE_MAP: Dict[int, DType] = {
  TensorProto.FLOAT:dtypes.float, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8, TensorProto.UINT16:dtypes.uint16,
  TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64, TensorProto.BOOL:dtypes.bool,
  TensorProto.FLOAT16:dtypes.float16, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32, TensorProto.UINT64:dtypes.uint64,
  TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
  TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float}
def parse_dtype(onnx_dtype: int) -> DType:
  if onnx_dtype in DTYPE_MAP: return supported_device_dtypes(DTYPE_MAP[onnx_dtype], Device.DEFAULT)
  raise NotImplementedError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")

def parse_buffer(inp: TensorProto) -> Tensor:
  if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
    return Tensor(dat, dtype=parse_dtype(inp.data_type), requires_grad=False).reshape(tuple(inp.dims))
  if len(inp.raw_data) > 0:
    data = np.frombuffer(inp.raw_data, dtype=(dt:=tensor_dtype_to_np_dtype(inp.data_type))).astype(_to_np_dtype(dt)).copy()
    return Tensor(data.reshape(tuple(inp.dims)), requires_grad=False)
  raise NotImplementedError(f"buffer with data type {TensorProto.DataType.Name(inp.data_type)} is not supported")

# src: onnx/onnx_ml_pb2.pyi
# NOTE: this is not a complete list
# torch's parser at onnx2torch/onnx_node.py: `OnnxNode._parse_attribute_value()`
ATTRS_MAP = {AttributeProto.FLOAT: lambda a: float(a.f), AttributeProto.INT: lambda a: int(a.i), AttributeProto.STRING: lambda a: a.s.decode("utf-8"),
         AttributeProto.TENSOR: lambda a: parse_buffer(a.t), AttributeProto.FLOATS: lambda a: tuple(float(x) for x in a.floats),
         AttributeProto.INTS: lambda a: tuple(int(x) for x in a.ints), AttributeProto.STRINGS: lambda a: tuple(x.decode("utf-8") for x in a.strings)}
def parse_attribute(a: AttributeProto):
  if a.type in ATTRS_MAP: return ATTRS_MAP[a.type](a)
  raise NotImplementedError(f"attribute with type {AttributeProto.AttributeType.Name(a.type)} is not supported")

# ========== runner
ONNXLIMIT = getenv("ONNXLIMIT", -1)

# onnx_ops implemented methods
onnx_ops = importlib.import_module('tinygrad.nn.onnx_ops')

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

# simple lambda methods lol
# used to trigger things like __add__ and __iadd__
lambda_methods: Dict[str, Callable[..., Any]] = {"Identity": lambda x:x, "Add": lambda x,y,*_,**__: x+y, "Sub": lambda x,y,*_:x-y}

def get_run_onnx(onnx_model: ModelProto):
  # initialization data
  model_parameters = {inp.name:parse_buffer(inp) for inp in onnx_model.graph.initializer}
  model_attributes = {num:{x.name:parse_attribute(x) for x in n.attribute} for num,n in enumerate(onnx_model.graph.node)}

  def run_onnx(inputs=None, debug=0):
    """
    Run the ONNX model with the provided inputs.

    `debug` parameter is used to control the logging verbosity for onnx.
    `debug` can be used with `DEBUGONNX` environment variable or passed in as an argument
    verbosity levels for `debug`:
      - 0: No debug output (default).
      - 1: Prints each op with input shapes.
      - 2: Prints intermediate outputs with names for each op.
      - 3: Prints the input for each op along with whether or not they are turned into a python const.
      - 4: TODO Prints the performance info for each op
      - 5: Runs correctness verification using `torch` for initialization, input, and output data (must have `torch` and `onnx2torch` installed)
    """
    debug, inputs = getenv("DEBUGONNX") or debug, inputs or {}
    if debug >= 5:
      from extra.onnx_verifier import verify_initialization, verify_op
      verify_initialization(onnx_model, inputs, model_parameters, model_attributes)

    # TODO: we can also infer output data types and verify that too. Torch does this, not sure we should
    # src: https://onnx.ai/onnx/repo-docs/IR.html#input-output-data-types
    # we're doing a get_input_metadata like thing when we prep onnx inputs and then we check inputs again using a get_input_metadata like thing ....
    # dynamically load inputs to correct dtype and validate shape when possible
    def parse_input(model_input:ValueInfoProto):
      if model_input.name not in inputs: raise RuntimeError(f"no data for {model_input=}")
      inp, type_proto = inputs[model_input.name], model_input.type
      if type_proto.HasField("map_type"): raise NotImplementedError(f"model input {model_input.name} has map type")
      if type_proto.HasField("optional_type"):
        if inp is None: return Tensor(None)
        type_proto = type_proto.optional_type.elem_type
      if type_proto.HasField("sequence_type"):
        if not isinstance(inp, Sequence): raise RuntimeError(f"model input has to be a sequence type {model_input.name}: {inp}")
        # the element_type of tensor_type of a sequence_type determines the dtype for all tensors in the sequence
        dtype = parse_dtype(type_proto.sequence_type.elem_type.tensor_type.elem_type)
        ret = [Tensor(i, dtype=dtype, requires_grad=False) if not isinstance(i, Tensor) else i for i in inp]
        # TODO: compile2.py is in conflict with dtype verification for input
        # either we compile2.py test half or we don't verify dtype for input, orrrr maybe add a strict parameter to enable tighter checking
        # if not all(t.dtype is dtype for t in ret): raise RuntimeError(f"{model_input.name}: parsed dtype {dtype} input {ret}")
        return ret
      assert type_proto.HasField("tensor_type"), f"{model_input=}"
      dtype = parse_dtype(type_proto.tensor_type.elem_type)
      inp = Tensor(inp, dtype=dtype, requires_grad=False) if not isinstance(inp, Tensor) else inp
      # if dtype is not inp.dtype: raise RuntimeError(f"{model_input.name}: has wrong input dtype, parsed dtype {dtype} input dtype {inp.dtype}")
      # if dim_value is missing, it's a variable dim_value, e.g. dim {dim_param: "N"}, so we skip validation for those
      for i,d in enumerate(type_proto.tensor_type.shape.dim):
        if not d.dim_param and inp.shape[i] != d.dim_value:
          raise RuntimeError(f"{model_input.name}: tensor proto shape {type_proto.tensor_type.shape} input shape {inp.shape}")
      return inp

    input_tensors = {model_input.name: parse_input(model_input) for model_input in onnx_model.graph.input if model_input.name not in model_parameters}
    intermediate_tensors: Dict[str,Tensor] = {}

    def fetch_tensor(x: str):
      if x in model_parameters: return model_parameters[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != "": return input_tensors[x]
      return None

    # inputs we need to turn into a python const to compute
    to_python_const_inps: Dict[str, Tuple[int, ...]] = \
    {"Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
      "CumSum": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
      "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4)}

    for num,n in enumerate(onnx_model.graph.node):
      # preperation
      tensor_inp, opt = [fetch_tensor(x) for x in n.input], model_attributes[num]
      if debug >= 1: print(f"{num}: op \"{n.op_type}\" input shapes {[x.shape if isinstance(x, Tensor) else x for x in tensor_inp]} opt {opt}")
      # to python consts
      if debug >= 3: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {t}" + (" -> *python const*" if i in to_python_const_inps.get(n.op_type,()) else "")
                                                      for i,(x,t) in enumerate(zip(n.input, tensor_inp))))
      inp = [to_python_const(x) if i in to_python_const_inps.get(n.op_type, ()) else x for i,x in enumerate(tensor_inp)]

      # running the op
      # tensor methods
      if debug >= 4: ...
      if n.op_type in exact_tensor_methods: ret = getattr(Tensor, n.op_type.lower())(*inp, **opt)
      elif n.op_type in equivalent_tensor_methods: ret = getattr(Tensor, equivalent_tensor_methods[n.op_type])(*inp, **opt)
      elif n.op_type in equivalent_tensor_methods_exceptions:
        rewrite = equivalent_tensor_methods_exceptions[n.op_type]
        ret = getattr(Tensor, rewrite[0])(*inp, **{rewrite[1].get(k, k):v for k,v in opt.items()})
      elif n.op_type in lambda_methods: ret = lambda_methods[n.op_type](*inp, **opt)
      # NOTE some ops live here because they require access to local variables
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
      else: raise NotImplementedError(f"op_type {n.op_type} not supported")
      if debug >= 4: ...

      # finalization after the op finishes running
      if not isinstance(ret, tuple): ret = (ret,)
      if len(n.output) > len(ret): raise RuntimeError(f"expected output size must be less than {len(ret)}, it's {n.output}")
      for i in range(len(n.output)): intermediate_tensors[n.output[i]] = ret[i]
      if debug >= 2: print("\toutputs:\n" + "\n".join(f"\t\t{n.output[i]} - {ret[i]}" for i in range(len(n.output))))
      if debug >= 5: verify_op(num, n, tuple(tensor_inp), opt, ret)

      if num == ONNXLIMIT: return {name:intermediate_tensors[name] for name in n.output}
    return {x.name:intermediate_tensors[x.name] for x in onnx_model.graph.output}
  return run_onnx
