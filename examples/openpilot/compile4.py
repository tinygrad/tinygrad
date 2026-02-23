import sys
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.helpers import fetch, Timing
#from tinygrad.engine.allocations import transform_to_call

# IMAGE_PITCH_ALIGN=256 CL=1 IMAGE=2 DEBUG=2 python3 examples/openpilot/compile4.py

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/cf6376aa9a090f0da26c280ef69eabf9bbdd51d1faac9ed392919c3db69be916"

# resnet 18
#OPENPILOT_MODEL = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx"

if __name__ == "__main__":
  with Timing("load graph: "):
    run_onnx = OnnxRunner(fetch(OPENPILOT_MODEL))
    input_shapes = {name: tuple(x if isinstance(x, int) else 1 for x in spec.shape) for name, spec in run_onnx.graph_inputs.items()}
    input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}

  def build():
    Tensor.manual_seed(100)
    inputs = {k:Tensor.randn(*shp, dtype=input_types[k]) for k,shp in sorted(input_shapes.items())}
    return next(iter(run_onnx(inputs).values()))

  #out = build()
  #ret, buffer_map = transform_to_call(out.uop)

  with Timing("early: "):
    out = build().realize()

  with Timing("prerealize: "):
    out = build()
    ((out.int()*0)^(out.int()*0)).realize()

  with Timing("realize: "):
    out.realize()

  #with Timing("realize 1: "): build().realize()
  #with Timing("realize 2: "): build().realize()
  #with Timing("realize 3: "): build().realize()
