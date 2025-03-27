from typing import List, Dict, cast
import ctypes
from tinygrad.helpers import dedup, cpu_time_execution, DEBUG, to_function_name
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.ops import Variable
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.renderer.cstyle import ClangRenderer

class ClangGraph(GraphRunner):
  def __init__(self, device, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    if not issubclass(type(device.renderer), ClangRenderer) and not isinstance(device.renderer, ClangRenderer): raise GraphException
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    self.base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b not in input_rawbuffers)
    self.base_rawbufs = [b._buf for b in self.base_bufs]

    # TODO: dedup
    targs = [(f"arg{i}", (x.dtype.ptr(), False)) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", (dtypes.char.ptr(), False)) for i,x in enumerate(self.base_bufs)] + \
            sorted([(f"{v.expr}", (dtypes.int, False)) for v in var_vals])

    def render_arg(buf):
      if buf in input_rawbuffers: return f"arg{input_rawbuffers.index(buf)}"
      else: return f"({device.renderer.render_dtype(buf.dtype)}*)(cbuf{self.base_bufs.index(buf.base)} + {buf.offset})"

    batched = ["void batched("+','.join([f"{device.renderer.render_dtype(x[1][0])} {x[0]}" for x in targs])+") {"]
    for i, ji in enumerate(jit_cache):
      args = [render_arg(buf) for buf in ji.bufs] + [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      batched.append(f"  j{i}{to_function_name(ji.prg.p.name)}({','.join(args)});")
    batched.append("}")

    prerendered = [device.renderer._render(ji.prg.p.uops) for i,ji in enumerate(jit_cache)]
    rendered_funcs = [device.renderer._render_body(f'j{i}'+prerendered[i][0], *prerendered[i][1:], ji.prg.p.uops) for i,ji in enumerate(jit_cache)]

    defines = device.renderer._render_defines() if hasattr(device.renderer, '_render_defines') else ""
    entry = device.renderer._render_entry("batched", targs) if hasattr(device.renderer, '_render_entry') else ""
    code = defines + '\n' + '\n'.join([''.join(f) for f in rendered_funcs]) + '\n'.join(batched) + '\n' + entry

    if DEBUG >= 4: print(code)
    self.clprg = device.runtime("batched", device.compiler.compile_cached(code))

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.base_rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)
