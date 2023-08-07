import csv
import pathlib
import time
import onnx
import torch
torch.set_num_threads(1)
from onnx2torch import convert
from extra.utils import download_file
from extra.onnx import get_run_onnx
from tinygrad.helpers import OSX
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device

MODELS = {
  "resnet50": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx",
  "openpilot": "https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx",
  "efficientnet": "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
  "shufflenet": "https://github.com/onnx/models/raw/main/vision/classification/shufflenet/model/shufflenet-9.onnx",

  # broken in torch MPS
  #"zfnet": "https://github.com/onnx/models/raw/main/vision/classification/zfnet-512/model/zfnet512-9.onnx",
  # TypeError: BatchNormalization() got an unexpected keyword argument 'is_test'
  #"densenet": "https://github.com/onnx/models/raw/main/vision/classification/densenet-121/model/densenet-3.onnx",
  # AssertionError: only onnx version >= 10 supported for slice
  #"bert": "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
  # really slow
  #"resnet18": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v2-7.onnx",
}

CSV = {}
open_csv = None

def benchmark(mnm, nm, fxn):
  tms = []
  for _ in range(3):
    st = time.perf_counter_ns()
    ret = fxn()
    tms.append(time.perf_counter_ns() - st)
  print(f"{m:15s} {nm:25s} {min(tms)*1e-6:7.2f} ms")
  CSV[nm] = min(tms)*1e-6
  return min(tms), ret

#BASE = pathlib.Path(__file__).parent.parent.parent / "weights" / "onnx"
BASE = pathlib.Path("/tmp/onnx")
def benchmark_model(m):
  global open_csv, CSV
  CSV = {"model": m}

  fn = BASE / MODELS[m].split("/")[-1]
  download_file(MODELS[m], fn)
  onnx_model = onnx.load(fn)

  excluded = {inp.name for inp in onnx_model.graph.initializer}
  input_shapes = {inp.name:tuple(x.dim_value if x.dim_value != 0 else 1 for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input if inp.name not in excluded}
  np_inputs = {k:torch.randn(shp).numpy() for k,shp in input_shapes.items()}
  assert len(input_shapes) < 20

  for device in ["METAL" if OSX else "GPU", "CLANG"]:
    Device.DEFAULT = device
    inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
    tinygrad_model = get_run_onnx(onnx_model)
    benchmark(m, f"tinygrad_{device.lower()}_jitless", lambda: {k:v.numpy() for k,v in tinygrad_model(inputs).items()})

    from tinygrad.jit import TinyJit
    tinygrad_jitted_model = TinyJit(lambda **kwargs: {k:v.realize() for k,v in tinygrad_model(kwargs).items()})
    for _ in range(3): {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}
    benchmark(m, f"tinygrad_{device.lower()}_jit", lambda: {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()})
    del inputs, tinygrad_model, tinygrad_jitted_model

  torch_model = convert(onnx_model)
  torch_inputs = [torch.tensor(x) for x in np_inputs.values()]
  benchmark(m, "torch_cpu", lambda: torch_model(*torch_inputs))

  torch_device = "mps" if OSX else "cuda"
  torch_mps_model = torch_model.to(torch_device)
  torch_mps_inputs = [x.to(torch_device) for x in torch_inputs]
  benchmark(m, f"torch_{torch_device}", lambda: torch_mps_model(*torch_mps_inputs))

  if open_csv is None:
    open_csv = csv.DictWriter(open('onnx_inference_speed.csv', 'w', newline=''), fieldnames=list(CSV.keys()))
    open_csv.writeheader()
  open_csv.writerow(CSV)

if __name__ == "__main__":
  for m in MODELS: benchmark_model(m)
