# render the kernel graph for execution outside tinygrad
from tinygrad.tensor import Tensor
from tinygrad.ops import UOp, Variable
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_webgpu import JavaScriptRenderer
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule
from tinygrad.nn.state import get_parameters
from typing import Callable, Sequence

class GraphRenderer(Renderer):
  def __init__(self, model:Callable, inputs:Sequence, device:str):
    # Inputs
    # JS model input args will have same relative positions as Tensors/Variables within above inputs
    #   only Tensor and Variable args will be captured as inputs, the rest is baked into the model if relevant
    #   Tensor input becomes JS typed array of closest type, e.g. Uint8Array, Int32Array
    #   Variable input becomes JS number type
    # TODO: don't use all positional args? instead use Object w/ names for Variable args, or for everything?
    # TODO: don't realize input Tensors?
    in_args = [(x.realize() if isinstance(x, Tensor) else x) for x in inputs if isinstance(x, (Tensor, Variable))]
  
    # Outputs
    # JS model output is an array of typed arrays whose order matches get_parameters output
    # TODO: if only one output, return a typed array instead of array of typed arrays?
    # TODO: assert no realizes happen within model, only kernelize is allowed
    output = [t.kernelize() for t in get_parameters(model(*inputs))]

    schedule, var_vals, _ = create_schedule_with_vars(UOp.sink(*[t.lazydata for t in output]))
    for si, ei in lower_schedule(memory_planner(schedule)):
      continue

def export_webgpu(model:Callable, inputs:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Renders the model's kernel graph into JavaScript and exports the model's state.
  """
  graph = GraphRenderer(model, inputs, "WEBGPU")
  return