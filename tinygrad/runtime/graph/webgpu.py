from typing import cast
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.runtime.ops_webgpu import WebGPUProgram, WGPUBufPtr
from tinygrad.device import Buffer
from tinygrad.ops import Variable

class WebGPUGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) and isinstance(ji.prg._prg, WebGPUProgram) for ji in jit_cache): raise GraphException

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False) -> float|None:
    for (j,i),idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = rawbufs[idx]

    def callback(command_encoder, comp_pass_desc):
      for ji in self.jit_cache:
        _prg = cast(WebGPUProgram, (prg:=cast(CompiledRunner, ji.prg))._prg)
        vals = tuple(var_vals[k] for k in prg.p.vars)
        bufs: list[WGPUBufPtr] = [b._buf for b in ji.bufs if b is not None]
        assert len(bufs) == len(ji.bufs)
        _prg.add_compute_pass(command_encoder, comp_pass_desc, *bufs, global_size=prg.p.launch_dims(var_vals)[0], vals=vals)

    time = cast(WebGPUProgram, cast(CompiledRunner, self.jit_cache[0].prg)._prg).execute_commands(callback, wait)
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].bufs[i] = None # for CapturedJit.free_intermediates to work
    return time