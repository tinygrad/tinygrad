from typing import cast
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.runtime.ops_webgpu import WebGPUProgram, execute_commands
from tinygrad.device import Buffer
from tinygrad.ops import Variable

class WebGPUGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    # TODO: capture this more cleanly?
    self._dev, self.timestamp_supported = (_prg:=jit_cache[0].prg._prg).dev, _prg.timestamp_supported

    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg._prg, WebGPUProgram) for ji in jit_cache): raise GraphException

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False) -> float|None:
    for (j,i),idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = rawbufs[idx]
    wait = wait and self.timestamp_supported

    def callback(command_encoder, comp_pass_desc):
      for ji in self.jit_cache:
        _prg = cast(WebGPUProgram, (prg:=cast(CompiledRunner, ji.prg))._prg)
        vals = tuple(var_vals[k] for k in prg.p.vars)
        _prg.add_compute_pass(command_encoder, comp_pass_desc, *[b._buf for b in ji.bufs], global_size=prg.p.launch_dims(var_vals)[0], vals=vals)

    time = execute_commands(self._dev, callback, wait)
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].bufs[i] = None # for CapturedJit.free_intermediates to work
    return time