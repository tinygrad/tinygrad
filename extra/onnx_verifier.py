import onnx
from tinygrad import Tensor
from onnx2torch import convert
from onnx2torch.onnx_graph import OnnxGraph, OnnxTensor
import torch
import numpy as np

onnx_graph, intermediate_tensors = None, []

def verify_initialization(onnx_model:onnx.ModelProto, inputs, model_parameters, model_attributes, train=False):
  global onnx_graph, intermediate_tensors
  torch_model = convert(onnx_model, attach_onnx_mapping=True)
  torch_model.train(train)
  onnx_graph = OnnxGraph(onnx_model.graph)

  def hook(module:torch.nn.modules.Module, inputs, outputs):
    inputs = tuple(o.detach().cpu().numpy() for o in inputs)
    outputs = tuple(o.detach().cpu().numpy() for o in (outputs if isinstance(outputs, tuple) else (outputs,)))
    input_tensors, output_tensors  = dict(zip(module.onnx_mapping.inputs, inputs)), dict(zip(module.onnx_mapping.outputs, outputs))
    intermediate_tensors.append((input_tensors, output_tensors))
  for _, layer in torch_model.named_children(): layer.register_forward_hook(hook)

  # run torch
  try: _ = torch_model(*(torch.tensor(x) for x in inputs.values()))
  except Exception as e: raise RuntimeError(f"debug=5 is not available, torch failed to run {onnx_model}") from e

  # run initialization verification
  assert len(intermediate_tensors) == sum(1 for _ in onnx_model.graph.node), "tinygrad op count different from torch"
  print("Verifying initialization parameter tensors")
  assert len(model_parameters) == len(onnx_graph.initializers)
  for name, torch_onnx_tensor in onnx_graph.initializers.items():
    tinygrad_onnx_tensor = model_parameters[name]
    np.testing.assert_allclose(torch_onnx_tensor.to_numpy(), tinygrad_onnx_tensor.numpy())
    print(f"\tverified {name}")
  print("Initialization parameters verified!")

  print("Verifying model attributes")
  for i, (name, node) in enumerate(onnx_graph.nodes.items()):
    # torch's onnx parser returns lists when floats, ints, or strings, we return tuple
    # see onnx2torch/onnx_node.py: `OnnxNode._parse_attribute_value()`
    tinygrad_attributes = model_attributes[i]
    for k,v in node.attributes.items():
      if isinstance(v, list): v = tuple(v)
      if isinstance(v, OnnxTensor): v = tuple(v.to_numpy().tolist())
      tinygrad_value = tinygrad_attributes[k]
      if isinstance(tinygrad_value, Tensor): tinygrad_value = tuple(tinygrad_value.numpy().tolist())
      assert v == tinygrad_value, f"{name}, {k}, {v}, {tinygrad_value}"
    print(f"\tverified {name}")
  print("Model attributes verified!")

# TODO make rtol and atol change dynamically depending on input datatype
def verify_op(num, node_proto:onnx.NodeProto, tiny_inp, tiny_opt, tiny_out, rtol=2e-3, atol=2e-3):
  # unpack
  torch_inp, torch_out = intermediate_tensors[num]
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
        # raise e
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
        # raise e
    else: print(f"\t\toutput WARNING: {output_name} does not match any torch outputs")
