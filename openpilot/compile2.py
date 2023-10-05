import sys
import onnx
import io
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor
import numpy as np
from tinygrad.helpers import dtypes, GlobalCounters
from tinygrad.realize import run_schedule

np.random.seed(1337)
def get_random_input_tensors(input_shapes):
  # this 16 is a random scale factor
  inputs = {k:Tensor.randn(*shp, requires_grad=False)*8 for k,shp in input_shapes.items()}
  np_inputs = {k:v.realize().numpy() for k,v in inputs.items()}
  return inputs, np_inputs

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  dat = fetch(sys.argv[1])
  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  inputs, np_inputs = get_random_input_tensors(input_shapes)

  ret = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  schedule = ret.lazydata.schedule()
  print(f"{len(schedule)} kernels scheduled")

  GlobalCounters.reset()
  run_schedule(schedule)

