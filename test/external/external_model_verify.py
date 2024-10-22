# this uses external models and torch's onnx runner to validate tinygrad's onnx
# we validate both initialization data and both input and output per op
# I think this is useful as a debug tool to see which exact op it started to mismatch if external_model_benchmark raises error
# would've been really helpful to debug float16 inaccuracy
from os import getenv
import onnx
from external_model_benchmark import MODELS
from tinygrad.helpers import fetch
from tinygrad.nn.onnx import get_run_onnx
from onnx.helper import tensor_dtype_to_np_dtype
import torch

def verify(model_name):
  fn = fetch(MODELS[model_name])
  onnx_model = onnx.load(fn)
  excluded = {inp.name for inp in onnx_model.graph.initializer}
  input_shapes = {inp.name:tuple(x.dim_value if x.dim_value != 0 else 1 for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input
                  if inp.name not in excluded}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input if inp.name not in excluded}
  np_inputs = {k:torch.randn(shp).numpy().astype(input_types[k]) for k,shp in input_shapes.items()}
  assert len(input_shapes) < 30, f"too many input shapes {len(input_shapes)}"

  tiny_model = get_run_onnx(onnx_model)
  _ = tiny_model(inputs=np_inputs, debug=5)

if __name__ == "__main__":
  model_name = getenv("MODEL")
  broken_on_torch = ("squeezenet", "commavq")
  available_models = tuple(m for m in MODELS.keys() if m not in broken_on_torch)
  assert model_name in MODELS, f"please specify a model with MODEL={available_models}"
  verify(model_name)
