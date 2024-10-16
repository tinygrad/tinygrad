# this uses external models to test onnx by verifying the output per OP
# I think this is useful as a debug tool to see which exact OP it started to mismatch if external_model_benchmark raises error
from os import getenv
import onnx
from external_model_benchmark import MODELS
from tinygrad import Tensor
from tinygrad.helpers import fetch
from tinygrad.runtime.onnx.onnx import get_run_onnx
from onnx.helper import tensor_dtype_to_np_dtype
from onnx2torch import convert
import torch
import numpy as np

def benchmark(model_name):
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
  intermediate_tensors = []
  def get_hook(name):
    def hook(module, input_, output):
      input_ = tuple(i.detach().cpu().numpy() for i in input_)
      output = output if isinstance(output, tuple) else (output,)
      output = tuple(o.detach().cpu().numpy() for o in output)
      intermediate_tensors.append((name, input_, output))
    return hook
  for name, layer in torch_model.named_children():
    layer.register_forward_hook(get_hook(name))
  torch_inputs = [torch.tensor(x) for x in np_inputs.values()]
  torch_out = torch_model(*torch_inputs)

  # tinygrad
  # TODO: Batchnorm gives wrong output?????????
  rtol, atol = 2e-3, 2e-3  # tolerance for fp16 models
  def tinygrad_hook(num, op_type, input_, output):
    torch_op_type, _, torch_output = intermediate_tensors[num]
    print(op_type, torch_op_type)
    # check output
    assert len(torch_output) == len(output), f"{num}:{torch_op_type} outputs length mismatched for tiny {len(output)} torch {len(torch_output)}"
    for i, (torch_output, tiny_output) in enumerate(zip(torch_output, output)):
      np.testing.assert_allclose(tiny_output.numpy(), torch_output, atol=atol, rtol=rtol,
                                err_msg=f"{num}:{torch_op_type} output {i} mismatched")
      print(f"output validated for {i}")

  tiny_model = get_run_onnx(onnx_model)
  tiny_out = tiny_model(inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}, hook=tinygrad_hook)

  np.testing.assert_allclose(list(tiny_out.values())[0].numpy(), torch_out.detach().numpy(), atol=atol, rtol=rtol,)

if __name__ == "__main__":
  model_name = getenv("MODEL")
  assert model_name in MODELS, f"please specify a model with MODEL={tuple(MODELS.keys())}"
  benchmark(model_name)
