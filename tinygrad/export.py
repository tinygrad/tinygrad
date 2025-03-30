from typing import Callable, Sequence, Optional
import types
from tinygrad import Tensor, TinyJit, UOp, Device
from tinygrad.device import Buffer
from tinygrad.engine.jit import CapturedJit, _prepare_jit_inputs, GraphRunner
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save

"""
Things we do here for model export. TODO: Should TinyJit/CapturedJit do any of these?
  1. Ensure model weights are on the correct device and contiguous for export
  2. Wrap the function to be exported with TinyJit and call it the correct number of times
  3. Validate the jitted function's return types
  4. Find the state_dict if one is not provided, map state_dict names to buffers in CapturedJit
  5. Track input indices of symbolic vars, required for using positional args in JS (could switch to kwargs)
"""
# Common logic for any model export
def export_init(model: Callable, inputs: Sequence, state_dict:dict[str,Tensor]={}, fix_contiguous=True) -> \
  tuple[CapturedJit, dict[Buffer, int], dict[UOp, int], dict[Buffer, str]]:

  if isinstance(model, types.MethodType): weights_holder = model.__self__
  elif hasattr(model, "__call__") and not isinstance(model, types.FunctionType): weights_holder = model
  else: weights_holder = None

  # TODO: get rid of this _state_dict / contiguous stuff
  # for the WebGPU efficientnet, torch_load loads non-contiguous tensors, which when not safe saved/loaded, gives incorrect inference
  # TODO: investigate contiguity in torch_load, and in safe_save/safe_load cycle (which enforces contiguity)
  _state_dict = get_state_dict(weights_holder)
  if _state_dict and fix_contiguous:
    for k,v in _state_dict.items():
      _state_dict[k] = v.contiguous().to(Device.DEFAULT).realize()
    load_state_dict(weights_holder, _state_dict)
  if not state_dict: state_dict = _state_dict

  @TinyJit
  def run(*args) -> list[Tensor]:
    out:list[Tensor]|tuple[Tensor] = returned if isinstance((returned := model(*args)), (list, tuple)) else [returned]
    assert all(isinstance(x, Tensor) for x in out), "must return a Tensor, or a list or tuple of Tensors"
    return [t.realize() for t in out]

  # TODO: improve automatic handling of JIT_BATCH_SIZE and GRAPH_ONE_KERNEL; tune JIT_BATCH_SIZE?
  for _ in range(3): run(*inputs) # Generate GraphRunner(s) in CapturedJit
  assert (cj:=run.captured) is not None and not cj._first_run
  # put input bufs back in ExecItems
  bufs = _prepare_jit_inputs(tuple(inputs), {})[0]
  for (j,i),idx in cj._input_replace.items(): cj._jit_cache[j].bufs[i] = bufs[idx]
  for ji in cj._jit_cache:
    if isinstance(ji.prg, GraphRunner):
      for (j,i),idx in ji.prg.input_replace.items(): ji.prg.jit_cache[j].bufs[i] = bufs[idx]
  # TODO: track input (+output?) metadata elsewhere, like CapturedJit or a runner class?
  reusable_inputs = [arg for arg in inputs if isinstance(arg, (Tensor, UOp))]
  # mapping bufs/vars to absolute indices of input args is needed if we want to have the same ordering when calling model from JS
  in_bufs: dict[Buffer, int] = {arg.lazydata.base.realized: i for i, arg in enumerate(reusable_inputs) if isinstance(arg, Tensor)}
  in_vars: dict[UOp, int] = {arg.unbind()[0]: i for i, arg in enumerate(reusable_inputs) if isinstance(arg, UOp)}

  weight_names = {v.lazydata.base.realized: k for k,v in state_dict.items()}
  weight_names.update({buf: weight_names.get(buf, cj.buf_names[buf]) for buf in cj.weight_bufs})

  return cj, in_bufs, in_vars, weight_names

def export_webgpu(model:Callable, inputs:Sequence, js_outfile:Optional[str]=None, state_dict:dict[str,Tensor]={},
                  model_name="model", save_weights=True, fix_contiguous=True) -> tuple[str, dict[str, Tensor]]:
  """
  Exports a javascript WebGPU implementation of a model together with its `state_dict`.
  """
  from tinygrad.runtime.graph.webgpu import render_js

  captured_jit, in_bufs, in_vars, weight_names = export_init(model, inputs, state_dict, fix_contiguous)

  js_code = render_js(captured_jit, in_bufs, in_vars, weight_names, model_name, save_weights)

  # ensure a complete state_dict is exported; for example, if no/incomplete state_dict is provided. Random seeds are missing from state_dict
  full_state_dict = {weight_names[buf]: Tensor(bytes(buf.as_buffer()), dtype=buf.dtype, device=buf.device).realize() for buf in weight_names}

  if js_outfile:
    with open(js_outfile, "w") as f: f.write(js_code)
    safe_save(full_state_dict, f"{js_outfile}.safetensors")

  return js_code, full_state_dict