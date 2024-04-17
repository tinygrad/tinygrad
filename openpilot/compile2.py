#!/usr/bin/env python3
import os, sys, io, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

import onnx
from typing import Tuple, List, Optional, Dict
from extra.onnx import get_run_onnx
from tinygrad import Tensor, Device, GlobalCounters, dtypes
from tinygrad.dtype import ImageDType
from tinygrad.helpers import partition, Context, fetch, getenv, DEBUG
from tinygrad.engine.realize import run_schedule, memory_planner
from tinygrad.engine.schedule import create_schedule
from tinygrad.ops import LoadOps, ScheduleItem
Device.DEFAULT = "GPU"

def get_schedule(onnx_data) -> Tuple[List[ScheduleItem], List[ScheduleItem]]:
  Tensor.no_grad = True
  Tensor.training = False

  # load the model
  onnx_model = onnx.load(io.BytesIO(onnx_data))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}

  # run the model
  inputs = {k:Tensor.empty(*shp) for k,shp in input_shapes.items()}
  ret: Tensor = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  schedule = create_schedule([ret.lazydata])

  # filter schedule that don't depend on the inputs
  input_lb = [x.lazydata.base.buffer for x in inputs.values()]
  depends = set(input_lb)
  for si in schedule:
    if any(b in depends for b in si.inputs):
      for out in si.outputs: depends.add(out)

  # run all kernels that don't depend on the inputs
  # NOTE: there's two extra kernels due to fusions that now happen since the weights aren't realized
  schedule, schedule_independent = partition(schedule, lambda si: any(out in depends for out in si.outputs))
  print(f"{len(schedule)} schedule items depend on the input, {len(schedule_independent)} don't")

  # confirm no loadops in the (non independent) schedule except for the ones that load the input buffers
  assert all(si.ast[0].op not in LoadOps or out in input_lb for si in schedule for out in si.outputs), "has loadops, can't compile to Thneed"
  return schedule, schedule_independent, inputs

def test_vs_onnx(onnx_data, schedule:Optional[List[ScheduleItem]], inputs:Dict[str, Tensor]):
  import onnx
  #import pyopencl as cl
  #from extra.thneed import Thneed
  import numpy as np
  onnx_model = onnx.load(io.BytesIO(onnx_data))

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  Tensor.manual_seed(1337)
  new_inputs = {k:Tensor.randn(*shp, requires_grad=False)*8 for k,shp in input_shapes.items()}
  new_np_inputs = {k:v.realize().numpy() for k,v in new_inputs.items()}

  if getenv("ORT"):
    # test with onnxruntime
    import onnxruntime as ort
    onnx_session = ort.InferenceSession(onnx_data)
    onnx_output = onnx_session.run([onnx_model.graph.output[0].name], {k:v.astype(np.float16) for k,v in new_np_inputs.items()})
    new_torch_out = onnx_output[0]
    print("got ort outputs")
  else:
    # test with torch
    from test.models.test_onnx import run_onnx_torch
    new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()
    print("got torch outputs")

  # if you don't have a schedule
  if schedule is None:
    run_onnx = get_run_onnx(onnx_model)
    new_tinygrad_out = next(iter(run_onnx(new_inputs).values())).cast(dtypes.float32).numpy()
    np.testing.assert_allclose(new_torch_out, new_tinygrad_out, atol=1e-4, rtol=1e-2)
    print("classic self-test passed!")
    return

  # set inputs
  for k,v in inputs.items(): v.lazydata.base.realized.copyin(new_np_inputs[k].data)

  # run code (all buffers have been allocated)
  GlobalCounters.reset()
  output = schedule[-1].outputs[0]
  run_schedule(schedule)

  new_tinygrad_out = np.frombuffer(output.as_buffer(), dtype=output.dtype.np)
  np.testing.assert_allclose(new_torch_out.reshape(new_tinygrad_out.shape), new_tinygrad_out, atol=1e-4, rtol=1e-2)
  print("semi-thneed self-test passed!")

if __name__ == "__main__":
  onnx_data = fetch(sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL).read_bytes()

  # quick test for ONNX issues
  #thneed_test_onnx(onnx_data, None)
  #exit(0)

  schedule, schedule_independent, inputs = get_schedule(onnx_data)
  schedule, schedule_input = partition(schedule, lambda x: x.ast[0].op not in LoadOps)
  print(f"{len(schedule_input)} inputs")

  run_schedule(schedule_independent)
  run_schedule(schedule_input)
  schedule = memory_planner(schedule)
  with Context(DEBUG=max(DEBUG.value, 2), BEAM=getenv("LATEBEAM")):
    image_count = sum(isinstance(out.dtype, ImageDType) for si in schedule for out in si.outputs)
    print(f"**** running real kernels {image_count}/{len(schedule)} images ****")

    GlobalCounters.reset()
    run_schedule(schedule[:])

  print("kernel count:", len(schedule))
  assert len(schedule) <= getenv("ALLOWED_KERNEL_COUNT", 0) or getenv("ALLOWED_KERNEL_COUNT", 0) == 0, "too many kernels!"

  # TODO: thneed is broken
  #output_fn = sys.argv[2] if len(sys.argv) >= 3 else "/tmp/output.thneed"
  #schedule_to_thneed(schedule, output_fn)

  FLOAT16 = getenv("FLOAT16", 0)
  if FLOAT16 == 0:
    try:
      test_vs_onnx(onnx_data, schedule, inputs)
    except ModuleNotFoundError as e:
      print(f"TEST NOT HAPPENING {e}")


