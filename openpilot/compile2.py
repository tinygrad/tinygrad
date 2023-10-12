import os
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"

import sys
import onnx
import io
from typing import Tuple, List
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.graph import print_tree
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, partition, GlobalCounters, Context, DEBUG, getenv
from tinygrad.realize import run_schedule
from tinygrad.ops import LoadOps, Device, ScheduleItem
Device.DEFAULT = "GPU"

"""
def get_random_input_tensors(input_shapes):
  # this 8 is a random scale factor
  inputs = {k:(Tensor.randn(*shp, requires_grad=False)*8).realize() for k,shp in input_shapes.items()}
  np_inputs = {k:v.numpy() for k,v in inputs.items()}
  return inputs, np_inputs
#inputs, np_inputs = get_random_input_tensors(input_shapes)
"""

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
  schedule, schedule_independent = get_schedule(sys.argv[1])

  run_schedule(schedule_independent)

  if getenv("GREEDY"):
    from tinygrad.codegen.linearizer import Linearizer
    from tinygrad.codegen.search import bufs_from_lin, time_linearizer, get_linearizer_actions
    for i,si in enumerate(schedule):
      if si.ast.op in LoadOps: continue
      linhc = Linearizer(si.ast)
      rawbufs = bufs_from_lin(linhc)
      linhc.hand_coded_optimizations()
      tmhc = time_linearizer(linhc, rawbufs)

      lin = Linearizer(si.ast)
      lin.required_optimizations()
      while 1:
        acted_lins = get_linearizer_actions(lin)
        timed_lins = {k:time_linearizer(v, rawbufs) for k,v in acted_lins.items()}
        opts = sorted(timed_lins.items(), key=lambda x: x[1])
        if opts[0][0] == 0: break   # we are done
        lin = acted_lins[opts[0][0]]
        if DEBUG >= 1: print(f"{opts[0][1]*1e6:10.2f} us from {len(opts):3d} actions", lin.colored_shape())
      tm = time_linearizer(lin, rawbufs)
      print(f"opt kernel {i:3d}: {tmhc*1e6:10.2f} -> {tm*1e6:10.2f}")

  #schedule = schedule[0:72]  # first model should be 8.85 ms
  print("**** running real kernels ****")
  with Context(DEBUG=2):
    GlobalCounters.reset()
    run_schedule(schedule)
