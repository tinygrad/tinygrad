# this uses external models and torch's onnx runner to validate tinygrad's onnx
# we validate both initialization data and both input and output per op
# I think this is useful as a debug tool to see which exact op it started to mismatch if external_model_benchmark raises error
# would've been really helpful to debug float16 inaccuracy
from os import getenv
import onnx
from external_model_benchmark import MODELS
from tinygrad import Tensor
from tinygrad.helpers import fetch
from tinygrad.runtime.onnx.onnx import get_run_onnx
from onnx.helper import tensor_dtype_to_np_dtype
from onnx2torch import convert
from onnx2torch.onnx_graph import OnnxGraph, OnnxTensor
import torch
import numpy as np

def verify(model_name):
  fn = fetch(MODELS[model_name])
  onnx_model = onnx.load(fn)
  excluded = {inp.name for inp in onnx_model.graph.initializer}
  input_shapes = {inp.name:tuple(x.dim_value if x.dim_value != 0 else 1 for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input
                  if inp.name not in excluded}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input if inp.name not in excluded}
  np_inputs = {k:torch.randn(shp).numpy().astype(input_types[k]) for k,shp in input_shapes.items()}
  assert len(input_shapes) < 30, f"too many input shapes {len(input_shapes)}"

  # torch
  torch_model = convert(onnx_model, attach_onnx_mapping=True)
  # NOTE: torch by default sets training as True, this means Batchnorm runs training code path, so we set it False
  torch_model.train(False)

  # NOTE: intermediate input tensors cannot be checked properly since nn.modules.Module is not functional so its attributes and input
  # do not map to onnx's inputs and attributes. Checking the modules named_buffers or named_parameters doesn't really help since they don't have
  # proper names and sometimes do not match to onnx's input.
  # in onnx2torch/converter.py:
  #   `torch_module, onnx_mapping = converter(onnx_node, onnx_graph)`
  # this line of code dispatches the onnx op to some nn.module and mangles with the inputs and attributes.
  # NOTE: even by turning attach_onnx_mapping to True `torch_model = convert(onnx_model, attach_onnx_mapping=True)`, the attach_onnx_mapping is only
  # the mapping to the post-mangled nn.moduels.Module input. We only verify that input using the mapped name.

  # torch_onnx_graph and torch_intermediate_tensors are used to verify tinygrad
  torch_onnx_graph = OnnxGraph(onnx_model.graph)
  # we don't use names for this because OnnxGraph.generate_node_name sometimes generates different names
  torch_intermediate_tensors = []

  # get torch intermediate inputs and outputs
  def hook(module:torch.nn.modules.Module, inputs, outputs):
    inputs = tuple(o.detach().cpu().numpy() for o in inputs)
    input_tensors = dict(zip(module.onnx_mapping.inputs, inputs))
    outputs = tuple(o.detach().cpu().numpy() for o in (outputs if isinstance(outputs, tuple) else (outputs,)))
    output_tensors = dict(zip(module.onnx_mapping.outputs, outputs))
    torch_intermediate_tensors.append((input_tensors, output_tensors))
  for _, layer in torch_model.named_children(): layer.register_forward_hook(hook)

  # run torch
  torch_global_inputs = [torch.tensor(x) for x in np_inputs.values()]
  torch_out = torch_model(*torch_global_inputs)

  # tinygrad
  # initialization verify
  def tinygrad_init_verify(tensors, attributes):
    assert len(torch_intermediate_tensors) == sum(1 for _ in onnx_model.graph.node), "tinygrad op count different from torch"

    print("Verifying initializer tensors (weights and biases)")
    assert len(tensors) == len(torch_onnx_graph.initializers)
    for name, torch_onnx_tensor in torch_onnx_graph.initializers.items():
      tinygrad_onnx_tensor = tensors[name]
      np.testing.assert_allclose(torch_onnx_tensor.to_numpy(), tinygrad_onnx_tensor.numpy())
      print(f"\tverified {name}")
    print("Initializers (weights and biases) verified!")

    print("Verifying Attributes")
    for i, (name, node) in enumerate(torch_onnx_graph.nodes.items()):
      # torch's onnx parser returns lists when floats, ints, or strings, we return tuple
      # see onnx2torch/onnx_node.py: `OnnxNode._parse_attribute_value()`
      tinygrad_attributes = attributes[i]
      for k,v in node.attributes.items():
        if isinstance(v, list): v = tuple(v)
        if isinstance(v, OnnxTensor): v = tuple(v.to_numpy().tolist())
        tinygrad_value = tinygrad_attributes[k]
        if isinstance(tinygrad_value, Tensor): tinygrad_value = tuple(tinygrad_value.tolist())
        assert v == tinygrad_value, f"{name}, {k}, {v}, {tinygrad_value}"
      print(f"\tverified {name}")
    print("Attributes Verified")

  rtol, atol = 2e-3, 2e-3  # tolerance for fp16 models
  # verify inputs and outputs per op
  def tinygrad_intermediate_result_verify(num, node_proto:onnx.NodeProto, tiny_inp, tiny_opt, tiny_out):
    # unpack
    torch_inp, torch_out = torch_intermediate_tensors[num]
    tiny_inps, tiny_outs = dict(zip(node_proto.input, tiny_inp)), dict(zip(node_proto.output, tiny_out))

    # start validation
    print("\tvalidation:")
    # validate inputs
    for input_name, tinygrad_input_tensor in tiny_inps.items():
      if (torch_input_tensor := torch_inp.get(input_name)) is not None:
        try:
          np.testing.assert_allclose(tinygrad_input_tensor.numpy(), torch_input_tensor, atol=atol, rtol=rtol,
                                    err_msg=f"{node_proto.op_type=} {input_name=}", strict=True)
          print(f"\t\tinput {input_name} validated")
        except AssertionError as e:
          print(f"\t\tinput ERROR: {input_name} validation failed")
          print("\t\t\t" + "\n\t\t\t".join(str(e).split("\n")))
          raise e
      else: print(f"\t\tinput WARNING: {input_name} does not match any torch inputs")

    # validate outputs
    for output_name, tinygrad_output_tensor in tiny_outs.items():
      if (torch_output_tensor := torch_out.get(output_name)) is not None:
        try:
          np.testing.assert_allclose(tinygrad_output_tensor.numpy(), torch_output_tensor, atol=atol, rtol=rtol,
                                    err_msg=f"{node_proto.op_type=} {output_name=}", strict=True)
          print(f"\t\toutput {output_name} validated")
        except AssertionError as e:
          print(f"\t\toutput ERROR: {output_name} validation failed")
          print("\t\t\t" + "\n\t\t\t".join(str(e).split("\n")[1:]))
          raise e
      else: print(f"\t\toutput WARNING: {output_name} does not match any torch outputs")

  tiny_model = get_run_onnx(onnx_model)
  tiny_out = tiny_model(inputs=np_inputs, debug=3, initialization_hook=tinygrad_init_verify, op_hook=tinygrad_intermediate_result_verify)

  np.testing.assert_allclose(list(tiny_out.values())[0].numpy(), torch_out.detach().numpy(), atol=atol, rtol=rtol,)
  print("Final output validated!")

if __name__ == "__main__":
  model_name = getenv("MODEL")
  assert model_name in MODELS, f"please specify a model with MODEL={tuple(MODELS.keys())}"
  verify(model_name)
