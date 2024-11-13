from __future__ import annotations
from typing import Dict, Tuple, Sequence, Callable, Union, Optional, List, cast, Literal, Any
import functools, io, math, inspect
from tinygrad.tensor import Tensor, Device, _broadcast_shape, ConstType
from tinygrad.helpers import getenv, CI, OSX, prod, flatten, make_tuple
from tinygrad.dtype import dtypes, DType
from tinygrad.device import is_dtype_supported
from onnx import AttributeProto, ModelProto, TensorProto, ValueInfoProto

# TODO try to remove this np stuff later
import numpy as np
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  def tensor_dtype_to_np_dtype(tensor_dtype:int) -> np.dtype: return TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

# ========== helpers
# Tensor -> python value cache for arguments
@functools.lru_cache(None)
def to_python_const(t:Any):
  if not isinstance(t,Tensor): return t
  if t.dtype is dtypes.uint8: return t.data().tobytes()
  return [] if 0 in t.shape else t.tolist()

# ======= parsers
# src: onnx/mapping.py  https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping
DTYPE_MAP: Dict[int, DType] = {
  TensorProto.FLOAT:dtypes.float, TensorProto.UINT8:dtypes.uint8, TensorProto.INT8:dtypes.int8, TensorProto.UINT16:dtypes.uint16,
  TensorProto.INT16:dtypes.int16, TensorProto.INT32:dtypes.int32, TensorProto.INT64:dtypes.int64, TensorProto.BOOL:dtypes.bool,
  TensorProto.FLOAT16:dtypes.float16, TensorProto.DOUBLE:dtypes.double, TensorProto.UINT32:dtypes.uint32, TensorProto.UINT64:dtypes.uint64,
  TensorProto.BFLOAT16:dtypes.bfloat16, TensorProto.FLOAT8E4M3FN:dtypes.float, TensorProto.FLOAT8E4M3FNUZ:dtypes.float,
  TensorProto.FLOAT8E5M2:dtypes.float, TensorProto.FLOAT8E5M2FNUZ:dtypes.float}
def parse_dtype(onnx_dtype: int) -> DType:
  if onnx_dtype in DTYPE_MAP: return DTYPE_MAP[onnx_dtype] if is_dtype_supported(DTYPE_MAP[onnx_dtype], Device.DEFAULT) else dtypes.float
  raise NotImplementedError(f"onnx dtype {TensorProto.DataType.Name(onnx_dtype)} is not supported")

def parse_buffer(inp: TensorProto) -> Tensor:
  if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
    return Tensor(dat, dtype=parse_dtype(inp.data_type), requires_grad=False).reshape(tuple(inp.dims))
  if len(inp.raw_data) > 0:
    return Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).copy().reshape(tuple(inp.dims)),
                  dtype=parse_dtype(inp.data_type), requires_grad=False)
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
      - 2: Prints intermediate outputs for each op.
      - 3: Prints the input for each op along with whether or not they are turned into a python const.
      - 4: Prints the details of `transform_arguments` and `dispatch` (this shows what is actually being ran to compute output)
    NOTE: debug level 5 greatly hinders performance!
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
      return None # TODO idk is returning none here the best way

    # inputs we need to turn into a python const to compute
    to_python_const_inps: Dict[str, Tuple[int, ...]] = \
    {"Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
      "CumSum": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
      "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4),
      **{"Reduce"+r: (1,) for r in ("Max", "Min", "Sum", "Mean", "SumSquare", "Prod", "L1", "L2", "LogSum", "LogSumExp")}}

    for num,n in enumerate(onnx_model.graph.node):
      # preperation
      tensor_inp, opt = [fetch_tensor(x) for x in n.input], model_attributes[num]
      if debug >= 1: print(f"{num}: op \"{n.op_type}\" input shapes {[x.shape if isinstance(x, Tensor) else x for x in tensor_inp]} opt {opt}")
      # to python consts
      if debug >= 3: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {t}" + (" -> *python const*" if i in to_python_const_inps.get(n.op_type,()) else "")
                                                      for i,(x,t) in enumerate(zip(n.input, tensor_inp))))
      # TODO: maybe combine handle_arguments and to_python_const
      inp = [to_python_const(x) if i in to_python_const_inps.get(n.op_type, ()) else x for i,x in enumerate(tensor_inp)]

      # running the op
      if debug >= 4: print(f"\texecution:\n\t\tbefore `transform_arguments`: {inp=}, {opt=}")
      inp, opt = transform_arguments(n.op_type, inp, opt, n=n, intermediate_tensors=intermediate_tensors)
      if debug >= 4: print(f"\t\tafter `transform_arguments`: {inp=}, {opt=}")
      fxn, fxn_name = dispatch(n.op_type)
      if debug >= 4: print(f"\t\tcalling: {fxn_name}({', '.join(f'{k}={v}' for k,v in inspect.signature(fxn).bind(*inp, **opt).arguments.items())})")
      ret = fxn(*inp, **opt)

      # finalization after the op finishes running
      if not isinstance(ret, tuple): ret = (ret,)
      if len(n.output) > len(ret): raise RuntimeError(f"expected output size must be less than {len(ret)}, it's {n.output}")
      for i in range(len(n.output)): intermediate_tensors[n.output[i]] = ret[i]
      if debug >= 2: print("\toutputs:\n" + "\n".join(f"\t\t{n.output[i]} - {ret[i]}" for i in range(len(n.output))))
      if debug >= 5: verify_op(num, n, tuple(tensor_inp), opt, ret)

      if num == ONNXLIMIT: return {name:intermediate_tensors[name] for name in n.output}
    return {x.name:intermediate_tensors[x.name] for x in onnx_model.graph.output}
  return run_onnx

# =========== ops lol
# this is such a cluster fk
# I think I'm trying too hard with abstractions here
# the tradeoff is linecount + maybe readablilty vs maintainablilty + good errors
# actually maybe readability also suffers, crap.

# Argument types are defined in here: https://github.com/onnx/onnx/blob/main/docs/Operators.md
# NOTE: There are numerous tests that have argument specifications which differ from Onnx's docs

# TODO TODO TODO !!!! maybe
# TODO: maybe map this better so the return is just one dict `args: Dict[str, Any]` with 'self': Tensor, ..., probably much clearer, and no need
# helper functions for op_handler. but again, tradeoff is more lines :D

# transforms the arguments so that it can be easily dispatched
def transform_arguments(op:str, inps:List, opts:Dict, **kwargs):
  # helper functions
  # NOTE: isinstance(arg, Tensor) to get aronud __bool__ being banned on tensor
  def arg_select(*args: Any): return next((arg for arg in args if isinstance(arg, Tensor) or arg), args[-1])
  def list_get(l, i, default=None): return l[i] if i < len(l) else default

  # padding helpers
  # (x1_begin, x2_begin, ..., x1_end, x2_end, ...) -> (..., x2_start, x2_end, x1_start, x1_end)
  def _onnx_pads_to_pad2d_pads(pads): return flatten(reversed(list((pB, pE) for pB, pE in zip(pads, pads[len(pads)//2:]))))
  # (H_pad, W_pad) -> (U_pad, L_pad, D_pad, R_pad) aka (x1_begin, x2_begin, ..., x1_end, x2_end, ...)
  def _auto_pad(pads, auto_pad: Literal["SAME_UPPER", "SAME_LOWER"]):
    return [pads[i]//2 for i in range(len(pads))] + [pads[i]-pads[i]//2 for i in range(len(pads))] if auto_pad == "SAME_UPPER" else \
            [pads[i]-pads[i]//2 for i in range(len(pads))] + [pads[i]//2 for i in range(len(pads))]
  # pool
  def resolve_pool_pads(inps, opts, **_):
    p_,k_,d_,s_ = opts.get('pads', 0), opts.get('kernel_shape') or inps[1].shape[2:], opts.get('dilations',1), opts.get('strides',1),
    i_,(s_,d_,p_) = inps[0].shape[-len(k_):], (make_tuple(x, len(k_)*2) for x in (s_, d_, p_))
    if (auto_pad:=opts.get('auto_pad', "NOTSET"))=="NOTSET": return inps,{**opts, "pads":_onnx_pads_to_pad2d_pads(p_ if len(p_)==len(k_)*2 else p_*2)}
    o_ = [((i - (1 if auto_pad in ("SAME_UPPER", "SAME_LOWER") else k)) // s + 1) for i,k,s in zip(i_, k_, s_)]
    return inps, {**opts, "pads": _onnx_pads_to_pad2d_pads(_auto_pad([(o-1)*s+k-i for o,i,k,s in zip(o_, i_, k_, s_)], auto_pad))}
  conv_opts = {'pads': 'padding', 'dilations': 'dilation', 'strides': 'stride', 'group': 'groups'}

  # reduce
  def reduce(inps,opts): return [inps[0]], \
    {'axis': arg_select(list_get(inps,1), opts.get('axes'), ([] if opts['noop_with_empty_axes'] else None)), 'keepdim': opts['keepdims']}

  # multi-line transformations
  def _slice(inps, opts, **_):
    # only onnx_model_version < 10 has opt, we just unload the opt into inp to match other versions
    inps += [list(v) for v in reversed(opts.values())] # axes, ends, starts -> starts, ends, axes
    starts, ends, axes, steps = inps[1], inps[2], list_get(inps, 3, list(range(inps[0].ndim))), list_get(inps, 4, [1]*inps[0].ndim)
    slices = [slice(0,x,1) for x in inps[0].shape]
    for i, axis in enumerate(axes): slices[axis] = slice(starts[i], ends[i], steps[i])
    return [inps[0]], {"indices": tuple(slices)}
  def split(inps, opts, n, **_):
    sz, n_outputs = inps[0].shape[opts.get('axis', 0)], opts.get('num_outputs') or len(n.output)
    return [inps[0]], {**opts, 'split': arg_select(list_get(inps,1), [sz // n_outputs + (1 if i < sz % n_outputs else 0) for i in range(n_outputs)])}
  def pad(inps, opts, **_):
    x, pads, value, mode = inps[0], opts.get('pads') or inps[1], opts.get('value') or list_get(inps,2) or 0, opts.get("mode", "constant")
    axes, real_pads = list_get(inps, 3, list(range(x.ndim))), [0] * (x.ndim*2)
    for i,axis in enumerate(axes): real_pads[axis%x.ndim], real_pads[axis%x.ndim+x.ndim] = pads[i], pads[i+len(axes)]
    return [x], {"padding":_onnx_pads_to_pad2d_pads(real_pads), "value":value, "mode":{"edge":"replicate", "wrap":"circular"}.get(mode, mode)}

  # helper functions for op_handler
  def set_defaults(inps, opts, defaults: Dict, **_): return inps, {**defaults, **opts}
  def rewrite_opt_names(inps, opts, mapping: Dict, **_): return inps, {mapping.get(k, k): v for k, v in opts.items()}
  def remove_opt_entries(inps, opts, opt_names: Sequence[str], **_): return inps, {k:v for k,v in opts.items() if k not in opt_names}

  op_handler: Dict[str, Callable] = {
    **{op: functools.partial(set_defaults, defaults=d) for op, d in {"HardSigmoid": {"alpha": 0.2, "beta": 0.5}}.items()},
    **{op: functools.partial(rewrite_opt_names, mapping=amap) for op, amap in
       {"Concat": {"axis": "dim"}, "LeakyRelu": {"alpha": "neg_slope"}, "Selu": {"gamma": "scale"}}.items()},
    # reduce ops
    **{"Reduce"+op:lambda inps, opts, **_: reduce(*set_defaults(inps, opts, {"noop_with_empty_axes":0, "keepdims":1}))
       for op in ("Max", "Min", "Sum", "Mean", "SumSquare", "Prod", "L1", "L2", "LogSum", "LogSumExp")},
    **{op: lambda inps, _, **__: (inps, {"axis": tuple(range(2, inps[0].ndim)), "keepdim": True}) for op in ("GlobalAveragePool", "GlobalMaxPool")},
    **{op: lambda inps, opts, **_: resolve_pool_pads(inps,opts) for op in {"AveragePool", "MaxPool"}},
    "Conv": lambda inps, opts, **_: remove_opt_entries(*rewrite_opt_names(*resolve_pool_pads(inps,opts), conv_opts), ('kernel_shape', 'auto_pad')),
    "Split": lambda inps, opts, n, **_: remove_opt_entries(*rewrite_opt_names(*split(inps,opts,n), {'axis':'dim', 'split':'sizes'}), ('num_outputs')),
    "Gradient": lambda inps, opts, intermediate_tensor, **_: (inps, {"y": intermediate_tensor[opts["y"]]}),
    "Slice": lambda inps, opts, **_: _slice(inps, opts), "Einsum": lambda inps, opts, **_: ([opts['equation']] + inps, {}),
    # NOTE: cast and castlike currently doesn't support staturate for float8
    "Cast": lambda inps, opts, **_: (inps, {'dtype': parse_dtype(opts['to'])}),"CastLike": lambda inps, _,**__: ([inps[0]], {'dtype': inps[1].dtype}),
    "EyeLike": lambda inps, opts, **_: (inps, {**opts, 'dtype': parse_dtype(opts['dtype']) if 'dtype' in opts else inps[0].dtype}),
    "Clip": lambda inps, _, **__: ([inps[0], arg_select(list_get(inps,1), opts.get('min'), dtypes.min(inps[0].dtype)),
                                    arg_select(list_get(inps,2), opts.get('max'), dtypes.max(inps[0].dtype))], {}),
    "Shape": lambda _, opts, **__: ([], {"data":inps[0].shape[slice(opts.get("start", None), opts.get("end", None))], "dtype":dtypes.int64},),
    "Flatten": lambda inps, opts, **_: (inps, {"shape": (prod(inps[0].shape[0:opts.get("axis",1)]), -1)}),
    "Reshape": lambda inps, opts, **_: ([inps[0]],
                                        {"shape": [int(x) or (0 if opts.get('allowzero') else inps[0].size(i)) for i, x in enumerate(inps[1])]}),
    "Transpose": lambda inps, opts, **_: (inps, {"order": opts.get("perm") or list(range(inps[0].ndim))[::-1]}),
    "Pad": lambda inps, opts, **_: pad(inps, opts),
    "PRelu": lambda inps, _, **__: (inps, {"channel_dim": None}),
    }
  return op_handler.get(op, lambda inps, opts, **_: (inps, opts))(inps, opts, **kwargs)

# dispatches the op to Tensor.py methods
def dispatch(op:str):
  # tensor methods
  tensor_methods = {"Less": "__lt__", "Greater": "__gt__", "LessOrEqual": "__le__", "GreaterOrEqual": "__ge__", "Equal": "__eq__", "CastLike": "cast",
    "LogSoftmax": "log_softmax", "Not": "logical_not", "Tile": "repeat", "Range": "arange", "NegativeLogLikelihoodLoss": "nll_loss", "Concat": "cat",
    "ReduceMax": "max", "ReduceMin": "min", "ReduceSum": "sum", "ReduceMean": "mean", "ReduceProd": "prod", "GlobalAveragePool": "mean",
    "GlobalMaxPool": "max", "Conv": "conv2d", "Flatten": "reshape", "Transpose": "permute", "Pad": "pad2d", "Slice": "__getitem__",
    # "SpaceToDepth": "rearrange", "DepthToSpace": "rearrange",
    **{n:n.lower() for n in ("Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan", "Sinh", "Cosh", "Tanh",
    "Asinh", "Acosh", "Atanh", "Relu", "PRelu", "Elu", "Celu", "IsNaN", "IsInf",
    "Sigmoid", "MatMul", "Floor", "Ceil", "Softplus", "HardSwish", "Where", "Mul", "Softsign",
    "Xor", "Round", "Softmax", "Erf", "LeakyRelu", "Selu", "HardSigmoid", "Einsum", "Cast", "Split", "Clip", "Reshape")}}

  # easy lambdas
  # TODO: some of these easy lambdas can go in Tensor.py
  # TODO: hmmm maybe lambdas like this is also not the right idea lol, we're literally losing proper typing for a few lines
  lambda_methods: Dict[str, Callable[..., Tensor]] = {"Identity": lambda x:x, "Add": lambda x,y,**_: x+y, "Sub": lambda x,y:x-y,
    "Div": lambda x,y:(x/y).cast(x.dtype), "ArrayFeatureExtractor": lambda x,indices: x[..., indices],
    "ReduceSumSquare": lambda x,axis,keepdim: x.square().sum(axis, keepdim), "ReduceL1": lambda x,axis,keepdim: x.abs().sum(axis, keepdim),
    "ReduceL2": lambda x,axis,keepdim: x.square().sum(axis, keepdim).sqrt(), "ReduceLogSum": lambda x,axis,keepdim: x.sum(axis, keepdim).log(),
    "ReduceLogSumExp": lambda x,axis,keepdim: x.exp().sum(axis, keepdim).log(), "And": lambda x,y: (x==y).where(x,False),
    "Or": lambda x,y: (x==y).where(x,True), "Binarizer": lambda x, threshold: (x>threshold).float(),
    "Mean": lambda *data: functools.reduce(Tensor.add, data) / len(data), "Min": lambda *data: functools.reduce(Tensor.minimum, data),
    "Max": lambda *data: functools.reduce(Tensor.maximum, data), "Sum": lambda *data: functools.reduce(Tensor.add, data), }

  #######################
  # implemented methods #
  #######################
  def Gradient(x, y):
    y.backward()
    return tuple([t.grad for t in x])

  # TODO maybe don't cast hack things
  # TODO maybe implement meshgrid utility
  # TODO maybe write a helper function for patterns like
  # axes, real_pads  = axes or list(range(x.ndim)), [0] * (x.ndim*2)
  # for i,axis in enumerate(axes): real_pads[axis%x.ndim], real_pads[axis%x.ndim+x.ndim] = pads[i], pads[i+len(axes)]
  # **************** Free Ops ****************

  def Squeeze(data: Tensor, axes): return functools.reduce(lambda d, dim: d.squeeze(dim), sorted(axes, reverse=True), data)
  def Unsqueeze(data: Tensor, axes): return functools.reduce(lambda d, dim: d.unsqueeze(dim), sorted(axes), data)

  # **************** Simple Ops ****************

  # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_div.py

  def Constant(value:Optional[Tensor]=None, value_float=None, value_floats=None, value_int=None,value_ints=None,value_string=None,value_strings=None):
    if value is not None: return value
    if value_float is not None: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
    if value_floats is not None: return Tensor(value_floats, dtype=dtypes.float32, requires_grad=False)
    if value_int is not None: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
    if value_ints is not None: return Tensor(value_ints, dtype=dtypes.int64, requires_grad=False)
    if value_string is not None or value_strings is not None: raise NotImplementedError('value_string or value_strings not implemented')
  def ConstantOfShape(shape:List[ConstType], value:Tensor): return Tensor.ones(*shape, dtype=value.dtype) * (value if shape != [0] else 1)

  def Gelu(x:Tensor, approximate=None): return x.gelu() if approximate == "tanh" else 0.5 * x * (1 + (x/math.sqrt(2)).erf())
  def ThresholdedRelu(X: Tensor, alpha=1.0): return (X > alpha).where(X, 0)

  def OptionalHasElement(x: Optional[Tensor]=None): return Tensor(x is not None and x.numel() > 0)
  def OptionalGetElement(x: Optional[Tensor]=None): return x if x is not None else Tensor([])

  def Shape(data, dtype): return Tensor(data=data, dtype=dtype)
  def Size(data: Union[Tensor, List]): return prod(data if isinstance(data, list) else data.shape)
  def Expand(x: Tensor, shape:List): return x.expand(_broadcast_shape(x.shape, tuple(shape)))
  def Shrink(x: Tensor, bias=0.0, lambd=0.5): return (x < -lambd)*(x+bias) + (x > lambd)*(x-bias)

  def Asin(x): return Atan(x / (1 - x * x).sqrt())
  def Acos(x: Tensor):
    negate = (x < 0)
    x = x.abs()
    ret = ((((-0.0187293 * x) + 0.0742610)*x - 0.2121144) * x + 1.5707288) * (1.0 - x).sqrt()
    ret = ret - 2 * negate * ret
    return negate * math.pi + ret
  def Atan(y: Tensor):
    t1 = y.abs()
    t3 = (1 > t1).where(t1, t1.reciprocal())
    t4 = t3 * t3
    t0 = ((((-0.013480470 * t4 + 0.057477314) * t4 - 0.121239071) * t4 + 0.195635925) * t4 - 0.332994597) * t4 + 0.999995630
    t3 = t0 * t3
    t3 = (t1 > 1).where(1.570796327 - t3, t3)
    return y.sign() * t3

  def Trilu(x: Tensor, k:int=0, upper=1): return x.triu(k) if upper else x.tril(k)

  def ArgMax(x: Tensor, axis=0, keepdims=1, select_last_index=0):
    if select_last_index: return ((x.shape[axis]-1) - x.flip(axis).argmax(axis, keepdim=keepdims)).cast(dtypes.int64)
    return x.argmax(axis, keepdim=keepdims).cast(dtypes.int64)
  def ArgMin(x, axis=0, keepdims=1, select_last_index=0): return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

  # **************** Complex Ops ****************

  def Gemm(A: Tensor, B: Tensor, C: Optional[Tensor] = None, alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0):
    ret = alpha * (A.transpose(transA) @ B.transpose(transB))
    if C is not None: ret = ret + beta * (C if broadcast == 0 else C.reshape([-1 if i < len(C.shape) else 1 for i in range(ret.ndim)][::-1]))
    return ret

  def CumSum(X:Tensor, axis:int, exclusive=0, reverse=0):
    if reverse: X = X.flip(axis)
    if exclusive: X = X.pad(tuple((1,0) if i == axis else None for i in range(X.ndim)))\
                        .shrink(tuple((0,X.shape[axis]) if i == axis else None for i in range(X.ndim)))
    return X.cumsum(axis).flip(axis) if reverse else X.cumsum(axis)

  # TODO: this is copied from tinygrad/nn/__init__.py
  # spatial is from opset 7 and has since been removed
  def BatchNormalization(X: Tensor, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1, is_test=0):
    if training_mode:
      batch_mean = X.mean(axis=(reduce_axes:=tuple(x for x in range(X.ndim) if x != 1)))
      y = (X - batch_mean.detach().reshape(1, -1, *([1]*(X.ndim-2))))  # d(var)/d(mean) = 0
      batch_var = (y*y).mean(axis=reduce_axes)
      running_mean, running_var = input_mean * momentum + batch_mean * (1 - momentum), input_var * momentum + batch_var * (1 - momentum)
      return X.batchnorm(scale, B, batch_mean, batch_var.add(epsilon).rsqrt()), running_mean, running_var
    return X.batchnorm(scale, B, input_mean, (input_var + epsilon).rsqrt())

  def InstanceNormalization(x: Tensor, scale: Tensor, bias: Tensor, epsilon=1e-05):
    axis = tuple(range(2, x.ndim))
    mean = x.mean(axis=axis, keepdim=True)
    invstd = ((x - mean) ** 2).mean(axis=axis, keepdim=True).add(epsilon).rsqrt()
    return (x - mean) * scale.reshape(-1, 1, 1) * invstd + bias.reshape(-1, 1, 1)

  def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
    assert stash_type == 1, "only float32 is supported"
    axis = tuple(i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    mean = x.mean(axis=axis, keepdim=True)
    return x.layernorm(axis, epsilon) * scale + bias, mean, ((x - mean).square().mean(axis=axis, keepdim=True) + epsilon).rsqrt()

  def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
    return (x.rearrange('b (g c) h w -> b g (c h w)',g=num_groups).layernorm(eps=epsilon) * scale.unsqueeze(-1) + bias.unsqueeze(-1)).reshape(x.shape)

  # **************** Ops with Padding ****************
  # helpers
  # (x1_begin, x2_begin, ..., x1_end, x2_end, ...) -> (..., x2_start, x2_end, x1_start, x1_end)
  def _onnx_pads_to_pad2d_pads(pads): return flatten(reversed(list((pb, pe) for pb, pe in zip(pads, pads[len(pads)//2:]))))

  # (H_pad, W_pad) -> (U_pad, L_pad, D_pad, R_pad) aka (x1_begin, x2_begin, ..., x1_end, x2_end, ...)
  def _auto_pad(pads, auto_pad: Literal["SAME_UPPER", "SAME_LOWER"]):
    return [pads[i]//2 for i in range(len(pads))] + [pads[i]-pads[i]//2 for i in range(len(pads))] if auto_pad == "SAME_UPPER" else \
            [pads[i]-pads[i]//2 for i in range(len(pads))] + [pads[i]//2 for i in range(len(pads))]

  def AveragePool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=False, count_include_pad=False, dilations=1, pads=0, strides=1):
    ret = X.pad2d(pads).avg_pool2d(kernel_shape, strides, dilations, ceil_mode=ceil_mode)
    return ret if count_include_pad else ret / X.ones_like().pad2d(pads).avg_pool2d(kernel_shape, strides, dilations, ceil_mode=ceil_mode)

  def MaxPool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=False, dilations=1, pads=0, storage_order=0, strides=1):
    ret = X.pad2d(pads, float('-inf')).max_pool2d(kernel_shape, strides, dilations, ceil_mode=ceil_mode)
    indices = ((ret.reshape(-1, 1) == X.reshape(1, -1)) * Tensor.arange(X.numel(), dtype=dtypes.int64).unsqueeze(0)).sum(1).reshape(ret.shape)
    return ret.cast(X.dtype), indices.transpose(-2, -1) if storage_order else indices

  # TODO: basically implement scatter
  def MaxUnpool(xT: Tensor, xI: Tensor, outshape: Optional[Tensor]=None, kernel_shape=None, pads=(0,0,0,0), strides=None):
    out_sh = [(ks//2)*2 + st * inps for inps, st, ks in zip(xI.shape, strides, kernel_shape)]
    ret = ((xI.reshape(-1, 1) == Tensor.arange(prod(out_sh))) * xT.reshape(-1, 1)).sum(0).reshape(1, 1, *out_sh)
    if outshape is not None and outshape != ret.shape: pads = _auto_pad([outshape[-2] - ret.shape[-2], outshape[-1] - ret.shape[-1]], "SAME_UPPER")
    return ret.pad2d(_onnx_pads_to_pad2d_pads(pads))

  # TODO: their reference implementation and their documentation have different information
  # ref: https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv_transpose.py
  # doc: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
  # the current implementation makes sense to geohotstan and passes tests, but differs from both ref and doc
  def ConvTranspose(X: Tensor, W: Tensor, B:Optional[Tensor]=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None,
                    output_shape=None, output_padding=0, strides=1):
    input_shape, kernel_shape = X.shape[2:], (kernel_shape or W.shape[2:])
    strides, dilations, output_padding = (make_tuple(x, len(input_shape)) for x in (strides, dilations, output_padding))
    if output_shape is not None: # we pad according to output_shape
      X = X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=0, output_padding=0)
      return X.pad((None, None, *((0, out-xs) for out, xs in zip(output_shape, X.shape[2:]))))  # TODO: unsure about this
    # NOTE the pads either from args or auto_pad have the format [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    # this is asymmetrical padding and conv_transpose2d does not support it natively
    # padding for conv_transpose2d effectively "shrinks" the padding that goes into conv2d, so we can get around this asymmetry by shrinking it after
    if pads is None: # we generate asymmetrical pads
      output_shape = [X.shape[i+2] * strides[i] for i in range(len(strides))]
      pads = [strides[i]*(input_shape[i]-1) + output_padding[i] + ((kernel_shape[i]-1)*dilations[i]+1)-output_shape[i] for i in range(len(input_shape))] # noqa: E501
      pads = [0,0] * len(input_shape) if auto_pad == "NOTSET" else _auto_pad(pads, auto_pad)
    X = X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=0, output_padding=output_padding)
    return X.pad2d(_onnx_pads_to_pad2d_pads([-p for p in pads])) # neg it since we shrink
    # return X if pads is None else X.shrink((None, None, *((pl, X.size(i+2)-pr) for i,(pl,pr) in enumerate(zip(pads, pads[len(pads)//2:])))))

  def DepthToSpace(X:Tensor, blocksize:int, mode:str="DCR"):
    return X.rearrange("b (c h1 w1) h w -> b c (h h1) (w w1)" if mode=="CRD" else "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=blocksize, w1=blocksize)

  def SpaceToDepth(X:Tensor, blocksize:int): return X.rearrange("b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=blocksize, w1=blocksize)

  # Reimplemented here because you need legacy RNG for passing ONNX tests.
  def Dropout(data: Tensor, ratio=0.5, training_mode=False, seed=None):
    if not training_mode: return data, Tensor.ones(data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
    mask = Tensor(np.random.RandomState(seed).random(cast(Tuple[int,...], data.shape)) >= ratio, requires_grad=False, device=data.device)
    return data * mask * (1/(1.0 - ratio)), mask

  def LRN(x: Tensor, size, alpha=1e-4, beta=0.75, bias=1.0):
    pooled_x = (x**2).rearrange('b c h w -> b 1 c (h w)').pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1)
    return x / (pooled_x.reshape(x.shape) * alpha + bias).pow(beta)

  def MeanVarianceNormalization(x: Tensor, axis=(0, 2, 3)): return (x - x.mean(axis, keepdim=True)) / (x.std(axis, keepdim=True, correction=0) + 1e-9)

  def SoftmaxCrossEntropyLoss(scores: Tensor, labels: Tensor, weights=None, ignore_index=None, reduction="mean"):
    log_probs = scores.log_softmax(1)
    return log_probs.nll_loss(labels, weights, ignore_index, reduction), log_probs

  # TODO: is fuse_arange stuff working for this?
  def Gather(x: Tensor, indices: Tensor, axis=0):
    if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
      x_sh = list(x.shape)
      ret_shape = x_sh[:axis] + list(indices.shape) + x_sh[axis+1:]
      if indices.ndim > 1: indices = indices.flatten()
      python_indices = cast(Union[List[int], int], to_python_const(indices))
      normalized_python_indices = [python_indices] if not isinstance(python_indices, list) else [x_sh[axis]+x if x<0 else x for x in python_indices]
      args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(x_sh)] for i in normalized_python_indices]
      return x.shrink(arg=tuple(args[0])).cat(*[x.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
    # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
    return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]
  def GatherElements(x: Tensor, indices: Tensor, axis):
    indices = (indices < 0).where(x.shape[axis], 0) + indices
    return x.gather(axis, indices)

  def Resize(X:Tensor, roi=None, scales=None, sizes=None, antialias=0, axes=None, coordinate_transformation_mode='half_pixel',
              cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy='stretch',
              mode='nearest', nearest_mode='round_prefer_floor'):
    def _apply_nearest_mode(index: Tensor, input_dim, mode: str):
      if mode == "round_prefer_floor": index = (index - 0.5).ceil()
      elif mode == "round_prefer_ceil": index = (index + 0.5).floor()
      elif mode in ["floor", "ceil"]: index = getattr(index, mode)()
      else: raise ValueError(f"invalid {nearest_mode=}")
      return index.cast(dtypes.int32).clip(0, input_dim-1)
    def _apply_transformation(index: Tensor, input_dim, scale_dim, roi_dim, sizes_frac, mode):
      # TODO: needs more testing, not confident in this
      # NOTE: their reference implementation differ from the implementation in their reference docs
      # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py
      # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
      output_dim = scale_dim * input_dim
      if mode == "half_pixel": index = (index + 0.5) / scale_dim - 0.5
      elif mode == "align_corners": index = index * (input_dim - 1) / (output_dim - 1) if output_dim != 1 else Tensor([0])
      elif mode == "asymmetric": index = index / scale_dim
      elif mode == "pytorch_half_pixel": index = (index + 0.5) / scale_dim - 0.5 if output_dim != 1 else Tensor([-0.5])
      elif mode == "half_pixel_symmetric": index = input_dim / 2 * (1 - int(output_dim) / sizes_frac) + (index + 0.5) / scale_dim - 0.5
      elif mode == "tf_crop_and_resize": index = roi_dim[0] * (input_dim - 1) + index * ((roi_dim[1] - roi_dim[0]) * (input_dim - 1) / (output_dim - 1)) # noqa: E501
      else: raise ValueError(f"invalid {coordinate_transformation_mode=}")
      return index.clip(0, input_dim-1)

    scales, sizes = (None if scales is None else scales[-2:]), (None if sizes is None else sizes[-2:])
    # we pre permute the axes and permute back after resize
    axes, input_shape, = (axes or list(range(X.ndim))), X.shape[2:],
    perm = [a for a in range(len(X.shape)) if a not in axes] + list(axes)
    X = X.permute(*perm)

    if sizes is not None:
      if keep_aspect_ratio_policy in ["not_larger", "not_smaller"]:
        scale_fxn = min if keep_aspect_ratio_policy == "not_larger" else max
        scales = scale_fxn([sizes[i] / input_shape[i] for i in range(X.ndim-2) if i+2 in axes])
        sizes = [int((scales * input_shape[i]) + 0.5) if i+2 in axes else input_shape[i] for i in range(X.ndim-2)]
      else: scales = [sizes[-2] / X.size(-2), sizes[-1] / X.size(-1)]
    else: sizes = [int(sc*sh) for sc, sh in zip(scales, input_shape)]
    scales = [scales] * 2 if not isinstance(scales, list) else scales
    roi = [[st, ed] for st, ed in zip(roi, roi[len(roi)//2:])] if isinstance(roi, list) else [None] * (X.ndim-2)

    # NOTE: this transformation makes it so that we can't just call Tensor.interpolate
    # in Tensor.interpolate, we use aranged indexes without any transformation
    indexes = []
    for shape, size, scale, region in zip(input_shape, sizes, scales, roi):
      indexes.append(_apply_transformation(Tensor.arange(size), shape,scale, region, shape * scale, coordinate_transformation_mode))

    if mode == "nearest":
      indexes = [_apply_nearest_mode(index, shape, nearest_mode) for (index, shape) in zip(indexes, input_shape)]
      # meshgrid
      X = X[(..., *[idx.reshape(*(-1 if i == dim else 1 for i in range(len(sizes)))).expand(sizes) for dim, idx in enumerate(indexes)])]
    if mode == "linear":
      expand = list(X.shape)
      for i in range(-len(sizes), 0):
        reshape, index = [1] * X.ndim, indexes[i]
        reshape[i] = expand[i] = sizes[i]
        low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor(), index.ceil(), index - index.floor())]
        X = X.gather(i, low).lerp(X.gather(i, high), perc)
    if mode == "cubic": raise NotImplementedError("cubic interpolation is not implemented")
    return X.permute(*[perm.index(i) for i in range(len(perm))]) if perm else X
  def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)

  def CenterCropPad(t: Tensor, shape, axes=None):
    shrink_arg = [None] * t.ndim
    pad_arg = [None] * t.ndim
    for s, x in zip(shape, axes or range(t.ndim)):
      tx = t.shape[x]
      if s < tx: shrink_arg[x] = (tx//2 - (s+1)//2, tx//2 + s//2)
      elif s > tx: pad_arg[x] = ((s-tx)//2, (s-tx+1)//2)
    return t.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

  def OneHot(indices: Tensor, depth, values, axis=-1):
    # Scalar or Rank 1 tensor containing exactly one element
    depth, indices = depth[0] if isinstance(depth, list) else depth, (indices < 0).where(indices+depth, indices),
    if axis < 0: axis += indices.ndim + 1
    return (indices[:,None] == Tensor.arange(int(depth)).reshape((int(depth),) + (1,)*(indices.ndim-axis))).where(values[1], values[0])

  def Compress(inp: Tensor, condition, axis=None):
    if axis is None:
      inp = inp.flatten()
      axis = 0
    if axis < 0: axis += inp.ndim
    con = Tensor(np.arange(len(condition))[condition]) # TODO no boolean indexing in Tensor, pretty sure it's possible maybe?
    return inp[tuple(con if i == axis else slice(None) for i in range(inp.ndim))]

  def EyeLike(x: Tensor, dtype: Optional[int]=None, k=0):
    ret = Tensor.eye(cast(int, min(x.shape)), dtype=dtype)
    return ret if x.size(0) == x.size(1) else ret.pad(tuple(None if d == ret.size(0) else (k, d-ret.size(0)-k) for d in x.shape))

  def DequantizeLinear(x: Tensor, x_scale: Tensor, x_zero_point: Union[Tensor, int] = 0, axis=1, block_size=0):
    if axis < 0: axis += x.ndim
    if not isinstance(x_zero_point, Tensor): x_zero_point = Tensor(x_zero_point)
    if block_size: x_zer, x_sc = x_zero_point.repeat_interleave(block_size, axis), x_scale.repeat_interleave(block_size, axis)
    else:
      shape = (*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim))
      x_sc, x_zer = x_scale.reshape(shape), x_zero_point.reshape(shape)
    return ((x.float() - x_zer) * x_sc).cast(x_scale.dtype)

  # copied from https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_image_decoder.py
  # TODO maybe uint8 stuff may work?
  def ImageDecoder(encoded_stream: bytes, pixel_format="RGB"):
    try: import PIL.Image
    except ImportError as e: raise ImportError("Pillow must be installed to use the reference implementation of the ImageDecoder operator") from e
    img = PIL.Image.open(io.BytesIO(encoded_stream))
    if pixel_format == "BGR": return Tensor(np.array(img))[:, :, ::-1]
    if pixel_format == "RGB": return Tensor(np.array(img))
    if pixel_format == "Grayscale": return Tensor(np.array(img.convert("L"))).unsqueeze(-1) # (H, W) to (H, W, 1)
    raise ValueError(f"pixel_format={pixel_format!r} is not supported.")

  # TODO: can this be cleaned up? This can use linspace and meshgrid but idk about line save
  def AffineGrid(theta: Tensor, size, align_corners=0):
    _, _, *data_sz = size
    size_zeros, original_grid = Tensor.zeros(data_sz), Tensor.ones(data_sz)
    stackable = [original_grid]
    for dim, dim_sz in enumerate(data_sz):
      a = Tensor.arange(-1, 1.0001, 2/(dim_sz-1)) if align_corners == 1 else Tensor.arange(-1+1/dim_sz, 1, 2/dim_sz)
      if dim == 0: stackable = [a.reshape(dim_sz, *[1]*(len(data_sz)-1)) + size_zeros, *stackable]
      elif dim == 1: stackable = [a.reshape(1, dim_sz, *[1]*(len(data_sz)-2)) + size_zeros, *stackable]
      else: stackable = [a.reshape(1, dim_sz) + size_zeros, *stackable]
    original_grid = Tensor.stack(*stackable, dim=len(data_sz))
    transformed_grid = theta.matmul(original_grid.reshape(-1, len(data_sz)+1).transpose()).transpose(1, 2)
    return transformed_grid.reshape(size[0], *data_sz, theta.size(1))

  # **************** com.microsoft Ops ****************

  def SkipLayerNormalization(x:Tensor, skip:Tensor, gamma, beta:Optional[Tensor]=None, bias:Optional[Tensor]=None, epsilon=1e-12):
    if epsilon is None: epsilon=1e-12
    x = x + skip + bias
    return x.layernorm(eps=epsilon) * gamma + beta, None, None, x

  def FastGelu(x:Tensor, bias:Optional[Tensor]=None):
    # this is tanh approximated
    return (x + bias).gelu()

  # TODO: how to simplify these haha, I don't actually understand ML, IM A FRAUD
  def EmbedLayerNormalization(input_ids: Tensor, segment_ids: Tensor, word_embedding:Tensor,
                              position_embedding:Tensor, segment_embedding:Tensor, gamma=None, beta=None,
                              mask:Optional[Tensor]=None, position_ids:Optional[Tensor]=None, epsilon=1e-12, mask_index_type=None):
    # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
    assert (segment_ids is None) is (segment_embedding is None)
    assert (mask is None) is (mask_index_type is None)
    assert mask is None, "functionality not supported yet"  # TODO
    input_shape = input_ids.shape
    seq_length = input_shape[1]
    compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
    vocab_size, max_position_embeddings, type_vocab_size = word_embedding.shape[0], position_embedding.shape[0], (segment_embedding.shape[0]
                                                                                                                  if compute_seg_emb else None)

    def embedding(x, vocab_size, weight) -> Tensor:
      vocab_counter = Tensor.arange(vocab_size, dtype=x.dtype, requires_grad=False).reshape(1, 1, vocab_size).expand(*x.shape, vocab_size)
      return (vocab_counter == x.unsqueeze(2).expand(*x.shape, vocab_size)) @ weight

    # bert embedding layer
    if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
    wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
    pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
    seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

    embedding_sum = wrd_embedding_res + pos_embedding_res
    if seg_embedding_res is not None: embedding_sum = embedding_sum + seg_embedding_res
    out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
    return out, None, embedding_sum

  # TODO I gotta learn this
  def Attention(x:Tensor, weights, bias:Tensor, mask_index:Optional[Tensor]=None, past:Optional[Tensor]=None,
                relative_position_bias:Optional[Tensor]=None, past_sequence_length:Optional[Tensor]=None, do_rotary=None, mask_filter_value=None,
                num_heads=None, past_present_share_buffer=None, qkv_hidden_sizes=None, scale=None, unidirectional=None):
    # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
    assert num_heads is not None  # required
    assert (qkv_hidden_sizes is None and past is not None) or (qkv_hidden_sizes is not None)
    assert relative_position_bias is do_rotary is past_sequence_length is mask_filter_value is past_present_share_buffer is scale is None, \
    "functionality not supported yet"  # TODO strange params
    hidden_size, v_hidden_size = qkv_hidden_sizes[1:] if qkv_hidden_sizes is not None else 2*(weights.size(1) // 3,)

    if unidirectional:  # gpt-style
      assert hidden_size == v_hidden_size
      xqkv = x.linear(weights, bias)
      xq, xk, xv = [xqkv.shrink([None, None, (i*hidden_size, (i+1)*hidden_size)]) for i in range(3)]
    else:  # bert-style
      wq, wk, wv = weights[:,:hidden_size], weights[:,hidden_size:hidden_size+v_hidden_size], weights[:,hidden_size+v_hidden_size:]
      bq, bk, bv = (bias[:hidden_size], bias[hidden_size:hidden_size+v_hidden_size], bias[hidden_size+v_hidden_size]) if bias is not None else None
      xq, xk, xv = [x.linear(w, b) for w, b in zip((wq, wk, wv), (bq, bk, bv))]
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], num_heads, -1).transpose(1, 2) for x in (xq, xk, xv)]

    present = None
    if past is not None:
      xk, xv = Tensor.cat(past[0], xk, dim=-2), Tensor.cat(past[1], xv, dim=-2)
      present = Tensor.cat(xk.unsqueeze(0), xv.unsqueeze(0))

    def attn(query, key, value, attn_mask):
      query_length, key_length = query.shape[-2], key.shape[-2]
      cdim = max(query_length, key_length) + 1
      attn_weights = query @ key.transpose(-1, -2) / math.sqrt(value.shape[-1])
      # This is where Tensor.scaled_dot_product_attention differs:
      causal_mask = Tensor.ones((cdim, cdim), requires_grad=False, dtype=dtypes.bool).tril(0)[key_length - query_length : key_length, :key_length]
      masked = Tensor.where(causal_mask, attn_weights, -math.inf)
      if attn_mask is not None: masked = masked + attn_mask
      return masked.softmax(-1) @ value

    bsz, _, seq_len, _ = xq.shape
    out = attn(xq, xk, xv, mask_index).transpose(1, 2).reshape(bsz, seq_len, -1)
    return out, present

  #############
  # dispatch! #
  #############
  # op name rewrite
  if op in tensor_methods: return (fxn := getattr(Tensor, tensor_methods[op])), fxn.__qualname__
  # lambda
  if op in lambda_methods: return lambda_methods[op], 'lambda_methods.' + op
  # implemented
  if op in locals(): return locals()[op], "implemented." + op

  raise NotImplementedError(f"Operation '{op}' is not implemented.")
