import os
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "OPT" not in os.environ: os.environ["OPT"] = "99"
os.environ["PREREALIZE"] = "0"

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

import sys
import onnx
import io
from typing import Tuple, List
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.graph import print_tree
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, partition, GlobalCounters, Context, DEBUG, getenv, ImageDType
from tinygrad.realize import run_schedule
from tinygrad.ops import LoadOps, Device, ScheduleItem
from tinygrad.features.image import fix_schedule_for_images
Device.DEFAULT = "GPU"

def get_schedule(fn:str) -> Tuple[List[ScheduleItem], List[ScheduleItem]]:
  Tensor.no_grad = True
  Tensor.training = False

  # load the model
  dat = fetch(fn)
  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}

  # run the model
  inputs = {k:Tensor.empty(*shp) for k,shp in input_shapes.items()}
  ret: Tensor = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  schedule = ret.lazydata.schedule()

  # filter schedule that don't depend on the inputs
  input_lb = [x.lazydata.base for x in inputs.values()]
  depends = set(input_lb)
  for si in schedule:
    if any(b in depends for b in si.inputs):
      depends.add(si.out)

  # run all kernels that don't depend on the inputs
  # NOTE: there's two extra kernels due to fusions that now happen since the weights aren't realized
  schedule, schedule_independent = partition(schedule, lambda si: si.out in depends)
  print(f"{len(schedule)} schedule items depend on the input, {len(schedule_independent)} don't")

  # confirm no loadops in the (non independent) schedule except for the ones that load the input buffers
  assert all(si.ast.op not in LoadOps or si.out in input_lb for si in schedule), "has loadops, can't compile to Thneed"
  return schedule, schedule_independent

def lb_to_numbers(schedule):
  nschedule = []
  nlb = {}
  for op,out,buffers in schedule:
    for lb in (out,)+buffers:
      if lb not in nlb:
        nlb[lb] = len(nlb)
    nschedule.append((op, nlb[out], tuple(nlb[x] for x in buffers)))
  return nschedule

if __name__ == "__main__":
  schedule, schedule_independent = get_schedule(sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL)
  run_schedule(schedule_independent, disable_logging=True)
  schedule = fix_schedule_for_images(schedule)

  image_count = 0
  for si in schedule:
    if isinstance(si.out.dtype, ImageDType):
      image_count += 1

  print(f"**** running real kernels {image_count}/{len(schedule)} images ****")
  with Context(DEBUG=2, BEAM=getenv("LATEBEAM")):
    GlobalCounters.reset()
    run_schedule(schedule)

