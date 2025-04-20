from typing import cast
import itertools
from tinygrad.helpers import dedup, DEBUG, to_function_name
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer, CPUProgram, Device
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.ops import Variable
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import ClangRenderer

class CPUGraph(GraphRunner):
  def __init__(self, device, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    cpu_progs = [ item for item in jit_cache if isinstance(item.prg._prg, CPUProgram) ]
    if cpu_progs:
      if len(cpu_progs) != len(jit_cache): raise GraphException
      device = Device["CPU"]
    elif not isinstance(device.renderer, ClangRenderer): raise GraphException
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    self.base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b is not None and b not in input_rawbuffers)
    self.base_rawbufs = [b._buf for b in self.base_bufs] + [ji.prg._prg.fxn for ji in cpu_progs]
    init_lines: list[str] = []
    for i, ji in enumerate(cpu_progs):
      arg_dtypes = tuple(buf.dtype.ptr() for buf in ji.bufs if buf is not None) + tuple(v.dtype for v in cast(CompiledRunner, ji.prg).p.vars)
      s_args = ','.join([device.renderer.render_dtype(dt) for dt in arg_dtypes])
      fn_name = to_function_name(cast(CompiledRunner, ji.prg).p.name)
      init_lines.append(f"  void (*{fn_name})({s_args}) = (void (*)({s_args}))fxnp{i};")

    targs = [(f"arg{i}", (x.dtype.ptr(), False)) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", (dtypes.char.ptr(), False)) for i in range(len(self.base_bufs))] + \
            [(f"fxnp{i}", (dtypes.void.ptr(), False)) for i in range(len(cpu_progs))] + \
            sorted([(f"{v.expr}", (dtypes.int, False)) for v in var_vals])

    def render_arg(buf):
      if buf in input_rawbuffers: return f"arg{input_rawbuffers.index(buf)}"
      return f"({device.renderer.render_dtype(buf.dtype)}*)(cbuf{self.base_bufs.index(buf.base)} + {buf.offset})"

    batched = ["void batched("+','.join([f"{device.renderer.render_dtype(x[1][0])} {x[0]}" for x in targs])+") {"] + init_lines
    for i, ji in enumerate(jit_cache):
      args = [render_arg(buf) for buf in ji.bufs] + [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      batched.append(f"  {to_function_name(cast(CompiledRunner, ji.prg).p.name)}({','.join(args)});")
    batched.append("}")

    if not cpu_progs:
      prep = [device.renderer._render(cast(CompiledRunner, ji.prg).p.uops) for i,ji in enumerate(jit_cache)]
      funcs = dedup(device.renderer._render_body(prep[i][0], *prep[i][1:], cast(CompiledRunner, ji.prg).p.uops) for i,ji in enumerate(jit_cache))
      defines = dedup(itertools.chain.from_iterable(device.renderer._render_defines(cast(CompiledRunner, ji.prg).p.uops) for ji in jit_cache))
    else: funcs, defines = [], []
    entry = device.renderer._render_entry("batched", targs)
    code = '\n'.join(itertools.chain(defines, funcs, batched, (entry,)))

    if DEBUG >= 4: print(code)
    self.clprg = device.runtime("batched", device.compiler.compile_cached(code))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.base_rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)
