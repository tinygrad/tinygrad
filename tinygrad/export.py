from typing import Callable, Sequence, Optional, cast
import types, itertools
from tinygrad import Tensor, TinyJit, UOp, Device
from tinygrad.helpers import partition
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
def export_init(model: Callable, inputs: Sequence, state_dict:dict[str,Tensor]|None=None, fix_contiguous=True) -> \
  tuple[CapturedJit, dict[Buffer, int], dict[UOp, int], dict[Buffer, int], dict[Buffer, str], dict[Buffer, dict[str, str]]]:

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
  assert run.captured is not None and not run.captured._first_run
  # put input bufs back in ExecItems
  bufs = _prepare_jit_inputs(tuple(inputs), {})[0]
  for (j,i),idx in (cj:=run.captured)._input_replace.items(): cj._jit_cache[j].bufs[i] = bufs[idx]
  for ji in cj._jit_cache:
    if isinstance(ji.prg, GraphRunner):
      for (j,i),idx in ji.prg.input_replace.items(): ji.prg.jit_cache[j].bufs[i] = bufs[idx]
  # TODO: track input (+output?) metadata elsewhere, like CapturedJit or a runner class?
  reusable_inputs = [arg for arg in inputs if isinstance(arg, (Tensor, UOp))]
  # mapping bufs/vars to absolute indices of input args is needed if we want to have the same ordering when calling model from JS
  in_bufs: dict[Buffer, int] = {arg.lazydata.base.realized: i for i, arg in enumerate(reusable_inputs) if isinstance(arg, Tensor)}
  in_vars: dict[UOp, int] = {arg.unbind()[0]: i for i, arg in enumerate(reusable_inputs) if isinstance(arg, UOp)}
  out_bufs = cast(dict[Buffer, int], {t.lazydata.base.realized: i for i, t in enumerate(cj.ret)})

  buf_to_name = {v.lazydata.base.realized: k for k,v in state_dict.items()}
  state_bufs, empty_bufs = partition({b:None for ji in cj.jit_cache for b in ji.bufs if b not in in_bufs and b not in out_bufs},
                                      lambda x: x in cj.real_at_first_capture_bufs)
  ctr = itertools.count()
  empty_bufs = {buf: f"buf_{next(ctr)}" for buf in empty_bufs}
  state_bufs = {buf: {"default_name": (default:=f"buf_{next(ctr)}"), "state_name": buf_to_name.get(buf, default)} for buf in state_bufs}

  return cj, in_bufs, in_vars, out_bufs, empty_bufs, state_bufs

def export_webgpu(model:Callable, inputs:Sequence, js_outfile:Optional[str]=None, state_dict:dict[str,Tensor]|None=None,
                  model_name="model", save_weights=True, fix_contiguous=True) -> tuple[str, dict[str, Tensor]]:
  """
  Exports a javascript WebGPU implementation of a model together with its `state_dict`.
  """
  from tinygrad.runtime.graph.webgpu import render_js

  captured_jit, in_bufs, in_vars, out_bufs, empty_bufs, state_bufs = export_init(model, inputs, state_dict, fix_contiguous)

  js_code = render_js(captured_jit, in_bufs, in_vars, out_bufs, empty_bufs, state_bufs, model_name, save_weights)

  # ensure a complete state_dict is exported; for example, if no/incomplete state_dict is provided. Random seeds are missing from state_dict
  full_state_dict = {v["state_name"]: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in state_bufs.items()}

  if js_outfile:
    with open(js_outfile, "w") as f: f.write(js_code)
    safe_save(full_state_dict, f"{js_outfile}.safetensors")

  return js_code, full_state_dict