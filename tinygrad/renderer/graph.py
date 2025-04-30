# render the kernel graph for use outside tinygrad
from tinygrad.tensor import Tensor
from tinygrad.ops import UOp, Variable
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_webgpu import WebGPUJavaScriptRenderer
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule
from typing import Callable, Sequence

class GraphRenderer(Renderer):
  def __init__(self, kernel_graph:UOp):
    schedule, var_vals, _ = create_schedule_with_vars(UOp.sink(kernel_graph))
    for si, ei in lower_schedule(memory_planner(schedule)):
      continue

def export_webgpu(model:Callable, inputs:Sequence, js_outfile:str|None=None, state_dict:dict[str,Tensor]|None=None,
                  model_name="model", save_weights=True, fix_contiguous=True) -> tuple[str, dict[str, Tensor]]:
  """
  Exports a javascript WebGPU implementation of a model together with its `state_dict`.
  """
  return