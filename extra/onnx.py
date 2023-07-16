from __future__ import annotations
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
import importlib
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, getenv, DEBUG, dtypes
from typing import List,Dict
from onnx.onnx_pb import AttributeProto, ModelProto, TensorProto, TypeProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  tensor_dtype_to_np_dtype = lambda x: TENSOR_TYPE_TO_NP_TYPE[x]

# global numpy cache for parameters
numpy_cache = {}
def safe_numpy(t) -> np.ndarray:
  if not isinstance(t, Tensor): return t
  global numpy_cache
  if t not in numpy_cache:
    if DEBUG >= 1:
      print("numpy cache miss", t)
    numpy_cache[t] = t.numpy()
  return numpy_cache[t]

onnx_ops = importlib.import_module('extra.onnx_ops')

ONNXLIMIT = getenv("ONNXLIMIT", -1)

def get_run_onnx(onnx_model: ModelProto):
  def type_parse(type_proto: TypeProto):
    while True: # NEED BETTER PARSER :D
      attr = type_proto.WhichOneof('value')
      if attr == 'tensor_type': return tuple(x.dim_value for x in getattr(type_proto, attr).shape.dim)
      elif attr == 'sequence_type': 
        return []
        type_proto = getattr(type_proto, attr).elem_type
        attr = type_proto.WhichOneof('value')
        t_shape = [x.dim_value for x in getattr(type_proto, attr).shape.dim]
        return (1, *t_shape)
      elif attr == 'map_type': raise NotImplementedError(f"map_type is not implemented: {type_proto}")
      elif attr == 'opaque_type': raise NotImplementedError(f"opaque_type is not implemented: {type_proto}")
      elif attr == 'sparse_tensor_type': raise NotImplementedError(f"sparse_tensor_type is not implemented: {type_proto}")
      elif attr == 'optional_type': type_proto = getattr(type_proto, attr).elem_type
      else: raise Exception(f"unknown attr: {attr}, {type_proto}")
        
  def buffer_parse(inp: TensorProto) -> Tensor:
    if inp.data_type in (1,10,6,7):
      # TODO: this is shared with below
      if len(inp.float_data) > 0:
        ret = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
      elif len(inp.int64_data) > 0:
        ret = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
      elif len(inp.int32_data) > 0:
        ret = Tensor(np.array(inp.int32_data, dtype=np.int32).reshape(inp.dims), requires_grad=False)
      else:
        ret = Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).reshape(inp.dims).astype(np.float32).copy(), requires_grad=False)
    else:
      raise Exception(f"bad data type {inp.name} {inp.dims} {inp.data_type}")
    return ret

  def attribute_parse(a: AttributeProto) -> float | int | str | Tensor | tuple[float] | tuple[int]:
    # TODO: this is not complete, see onnx/onnx_ml_pb2.pyi for a complete list
    if a.type == AttributeProto.FLOAT: return float(a.f)
    elif a.type == AttributeProto.INT: return int(a.i)
    elif a.type == AttributeProto.STRING: return a.s.decode("utf-8")
    elif a.type == AttributeProto.TENSOR: return buffer_parse(a.t) # TENSOR
    elif a.type == AttributeProto.FLOATS: return tuple(float(x) for x in a.floats)
    elif a.type == AttributeProto.INTS: return tuple(int(x) for x in a.ints)
    elif a.type == AttributeProto.STRINGS: return tuple(x.decode("utf-8") for x in a.strings)
    elif a.type == AttributeProto.GRAPH: raise Exception(f"graph not implemented: {a.g}")
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a: RepeatedCompositeFieldContainer[AttributeProto]): return {x.name:attribute_parse(x) for x in a}

  tensors: Dict[str, Tensor] = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    if len(inp.raw_data) > 0:
      tensors[inp.name] = buffer_parse(inp)
    elif len(inp.float_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    elif len(inp.int64_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    else:
      print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
      print(inp)
      raise Exception("no data")
    if DEBUG >= 1:
      print("realize", inp.name)
    tensors[inp.name].realize()

  # preparse the attributes
  attribute_dict = {}
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = attribute_to_dict(n.attribute)
  
  onnx_model_version = onnx_model.opset_import[0].version

  def run_onnx(inputs={}, debug=False):
    if getenv("DEBUGONNX"): debug = True
    input_tensors: Dict[str,Tensor] = {}
    intermediate_tensors: Dict[str,Tensor] = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    requires_grad = False
    for opset in onnx_model.opset_import: 
      if opset.domain == "ai.onnx.preview.training": requires_grad = True # TODO TEST WITH REAL ONNX MODELS CUZ I HAVE NO IDEA IF THIS WORKS IN PRACTICE

    # get inputs
    for inp in onnx_model.graph.input:
      if inp.name in tensors: continue
      shape = type_parse(inp.type)
      if len(shape) >= 1 and shape[0] == 0 and shape != (0,): shape = tuple([1]+list(shape[1:]))   # 1 batch size
      if inp.name in inputs:
        if isinstance(inputs[inp.name], Tensor):
          input_tensors[inp.name] = inputs[inp.name]
        elif isinstance(inputs[inp.name], list):
          input_tensors[inp.name] = [Tensor(i, requires_grad=False) for i in inputs[inp.name]]
        elif requires_grad:
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=True)
        else:
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=False)
        input_shape = input_tensors[inp.name].shape if input_tensors[inp.name].__class__ is Tensor else []
        assert input_shape == shape or shape == [], f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
        for _,v in input_tensors.items(): v.realize() if v.__class__ is Tensor else ...
      else:
        raise Exception(f"no data for {inp.name} with shape {shape}")

    def fetch_tensor(x: str):
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != str(): return input_tensors[x]
      return None

    for num,n in enumerate(onnx_model.graph.node):
      inp: List[Tensor] = []
      if debug: print("inputs:")
      for x in n.input:
        t = fetch_tensor(x)
        if debug: print(f"\t{x} - {t}")
        if debug: print(f"{t.numpy() if isinstance(t, Tensor) else t}")
        inp.append(t)
      opt = attribute_dict[num]
      if debug: print(f"{num}: op {n.op_type} shape {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")
      # free ones
      if n.op_type == "Relu": ret = inp[0].relu()
      elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
      elif n.op_type == "Tanh": ret = inp[0].tanh()
      elif n.op_type == "MatMul": ret = inp[0].matmul(inp[1])
      # one liners
      elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt.get('alpha', 1.0))
      elif n.op_type == "Concat": ret = inp[0].cat(*inp[1:], dim=opt['axis'])
      elif n.op_type == "Transpose": ret = inp[0].permute(order=opt.get('perm', list(range(len(inp[0].shape))[::-1])))
      elif n.op_type == "Squeeze": 
        axes = opt['axes'] if 'axes' in opt else safe_numpy(inp[1])
        axes = [int(x) if x >= 0 else int(x+inp[0].ndim) for x in axes]
        ret = inp[0].reshape([s for i,s in enumerate(inp[0].shape) if i not in axes])
      elif n.op_type == "Div":
        # in openpilot, due to SHUFFLE_PAD_OPS issues, we are spending an extra kernel
        ret = inp[0].div(inp[1]) if inp[0].dtype == dtypes.float else inp[0].div(inp[1]).floor()
      elif n.op_type == "Constant":
        if 'value' in opt: ret = opt['value'] # tensor
        elif 'value_float' in opt: ret = Tensor(np.array(opt['value_float'], dtype=np.float32), requires_grad=False)
        elif 'value_int' in opt: ret = Tensor(np.array(opt['value_int'], dtype=np.int64), requires_grad=False)
        elif 'value_floats' in opt: ret = Tensor(np.array(opt['value_floats'], dtype=np.float32), requires_grad=False)
        elif 'value_ints' in opt: ret = Tensor(np.array(opt['value_ints'], dtype=np.int64), requires_grad=False)
        else: raise NotImplementedError(f'Constant not implemented for {opt}')
      elif n.op_type == "Reshape": ret = inp[0].reshape([int(x) if x != 0 else inp[0].shape[i] for i,x in enumerate(safe_numpy(inp[1]))])
      elif n.op_type in ["Add", "Sub", "Mul", "Pow"]:
        if all(isinstance(x, Tensor) for x in inp) and (len(inp[0].shape) != len(inp[1].shape)) and (prod(inp[0].shape) == prod(inp[1].shape)):
          inp[1] = inp[1].reshape(inp[0].shape)
        # TODO: is this right?
        if 'broadcast' in opt: inp[1] = inp[1].reshape([-1 if i == opt['broadcast'] else 1 for i in range(len(inp[0].shape))])
        if n.op_type == "Add": ret = inp[0] + inp[1]
        if n.op_type == "Sub": ret = inp[0] - inp[1]
        if n.op_type == "Mul": ret = (inp[0] * inp[1]).cast(inp[0].dtype)
        if n.op_type == "Pow": ret = (inp[0] ** inp[1]).cast(inp[0].dtype)
      elif n.op_type == "Split":
        if 'axis' not in opt: opt['axis'] = 0
        if 'num_outputs' in opt or len(inp) == 1: 
          opt['split'] = [inp[0].shape[opt['axis']] // len(n.output)] * len(n.output)
          for i in range(inp[0].shape[opt['axis']] % len(n.output)):
            opt['split'][i] += 1
        if 'split' not in opt: opt['split'] = [int(x) for x in safe_numpy(inp[1])]  # split can be a tensor
        i = 0
        arg = [(0,x) for x in inp[0].shape]
        for o,s in zip(n.output, opt['split']):
          arg[opt['axis']] = (i,i+s)
          intermediate_tensors[o] = inp[0].slice(arg=arg)
          i = i+s
        continue
      elif n.op_type == "Slice":
        if onnx_model_version < 10:
          axes = list(opt["axes"])
          ends = list(opt["ends"])
          starts = list(opt["starts"])
          steps = [1]*inp[0].ndim
        else:
          starts, ends = inp[1:3]
          axes = safe_numpy(Tensor.arange(inp[0].ndim, dtype=dtypes.int32) if len(inp) <= 3 else inp[3]).tolist()
          steps = safe_numpy(inp[4]) if len(inp) > 4 else [1]*inp[0].ndim
          starts, ends = safe_numpy(starts.ceil().cast(dtypes.int32)).tolist(), safe_numpy(ends.ceil().cast(dtypes.int32)).tolist()
        arg = [(0,x,1) for x in inp[0].shape]
        shrink_args = [(0,x) for x in inp[0].shape]
        only_shrink = False # HACK BUT SOME TESTS [s:e:st], st > 1 and s == e. otherwise __getitem__ Tensor.reshape() has to allow 0 in newshape 
        for i, axis in enumerate(axes):
          axis = int(axis) + inp[0].ndim if axis < 0 else int(axis)
          starts[i], ends[i] = starts[i] + inp[0].shape[axis] if starts[i] < 0 else starts[i], ends[i] + inp[0].shape[axis] if ends[i] < 0 else ends[i]
          starts[i], ends[i] = max(0, min(starts[i], inp[0].shape[axis])), max(0, min(ends[i], inp[0].shape[axis]))
          shrink_args[axis] = (starts[i], ends[i])
          if starts[i] == ends[i]: 
            only_shrink = True 
            continue
          if starts[i] > ends[i] and steps[i] >= 0: steps[i] = -steps[i]
          arg[axis] = (starts[i], ends[i], steps[i])
        ret = inp[0].shrink(tuple(shrink_args)) if only_shrink else inp[0].__getitem__(tuple([slice(s,e,st) for s,e,st in arg]))
      elif n.op_type == "Shrink":
        bias = opt['bias'] if 'bias' in opt else 0
        ret = (inp[0] < -opt['lambd'])*(inp[0]+bias) + (inp[0] > opt['lambd'])*(inp[0]-bias)
      elif n.op_type == "Gradient": # TODO NO IDEA IF THIS IS CORRECT LOL
        assert len(opt["xs"]) == len(inp), "output and input has to match lol"
        y = opt["y"]
        intermediate_tensors[y].backward()
        ret = tuple([t.grad for t in inp])
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
        raise Exception(f"op_type {n.op_type} not supported")
      if not isinstance(ret, tuple): ret = (ret, )
      # else: print(ret); print(len(ret))
      assert len(n.output) <= len(ret), f"expected output size must be less than {len(ret)}, it's {n.output}"
      if debug: print([x.shape if isinstance(x, Tensor) else None for x in ret])
      if debug: print("outputs:")
      for i in range(len(n.output)): 
        if debug: print(f"\t{n.output[i]} - {ret[i]}")
        if debug: print(f"{ret[i].numpy() if isinstance(ret[i], Tensor) else type(ret[i])}")
        intermediate_tensors[n.output[i]] = ret[i]
      if num == ONNXLIMIT:
        output_tensor_names = n.output
        break

    return {outp:intermediate_tensors[outp] for outp in output_tensor_names}
  return run_onnx
