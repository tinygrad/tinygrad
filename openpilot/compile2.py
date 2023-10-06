import sys
import onnx
import io
from extra.utils import fetch, print_tree
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, partition, GlobalCounters, Context
from tinygrad.realize import run_schedule
from tinygrad.ops import LoadOps

"""
def get_random_input_tensors(input_shapes):
  # this 8 is a random scale factor
  inputs = {k:(Tensor.randn(*shp, requires_grad=False)*8).realize() for k,shp in input_shapes.items()}
  np_inputs = {k:v.numpy() for k,v in inputs.items()}
  return inputs, np_inputs
#inputs, np_inputs = get_random_input_tensors(input_shapes)
"""

def get_schedule(fn:str):
  Tensor.no_grad = True
  Tensor.training = False

  # load the model
  dat = fetch(fn)
  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}

  # run the model
  inputs = {k:Tensor.empty(*shp) for k,shp in input_shapes.items()}
  ret = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  schedule = ret.lazydata.schedule()

  # filter schedule that don't depend on the inputs
  input_lb = [x.lazydata.base for x in inputs.values()]
  depends = set(input_lb)
  for op,out,buffers in schedule:
    if any(b in depends for b in buffers):
      depends.add(out)

  # run all kernels that don't depend on the inputs
  # NOTE: there's two extra kernels due to fusions that now happen since the weights aren't realized
  schedule, schedule_independent = partition(schedule, lambda x: x[1] in depends)
  print(f"{len(schedule)} schedule items depend on the input, {len(schedule_independent)} don't")

  # confirm no loadops in the (non independent) schedule except for the ones that load the input buffers
  assert all(op.op not in LoadOps or out in input_lb for op,out,_ in schedule), "has loadops, can't compile to Thneed"
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
  #schedule_no_empty = [x for x in schedule if x[0].op not in LoadOps]
  #print(lb_to_numbers(schedule_no_empty)[11])
  #exit(0)

  #for op,out,buffers in lb_to_numbers(schedule): print(out,buffers)
  #exit(0)

  run_schedule(schedule_independent)
  schedule = schedule[0:72]  # first model should be 8.85 ms
  print("**** running real kernels ****")
  with Context(DEBUG=2):
    GlobalCounters.reset()
    run_schedule(schedule)
