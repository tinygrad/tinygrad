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
    if issubclass(device.runtime, CPUProgram): device = Device["CPU"]
    elif not isinstance(device.renderer, ClangRenderer): raise GraphException
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    self.base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b is not None and b not in input_rawbuffers)
    self.rawbufs = [b._buf for b in self.base_bufs]

    def render_arg(buf):
      if buf in input_rawbuffers: return f"arg{input_rawbuffers.index(buf)}"
      return f"({device.renderer.render_dtype(buf.dtype)}*)(cbuf{self.base_bufs.index(buf.base)} + {buf.offset})"

    defines, funcs, batched_inner  = [], [], []
    rendered_casts: set[str] = set()
    for i, ji in enumerate(jit_cache):
      runner = cast(CompiledRunner, ji.prg)
      fn_name = to_function_name(runner.p.name)

      if isinstance(runner._prg, CPUProgram) and fn_name not in rendered_casts:
        arg_dtypes = tuple(buf.dtype.ptr() for buf in ji.bufs if buf is not None) + tuple(v.dtype for v in runner.p.vars)
        s_args = ','.join([device.renderer.render_dtype(dt) for dt in arg_dtypes])
        batched_inner.append(f"  void (*{fn_name})({s_args}) = (void (*)({s_args}))fxnp{len(rendered_casts)};")
        self.rawbufs.append(runner._prg.fxn)
        rendered_casts.add(fn_name)

      if not isinstance(runner._prg, CPUProgram):
        defines.extend(device.renderer._render_defines(runner.p.uops))
        funcs.append(device.renderer._render_body(*device.renderer._render(runner.p.uops), runner.p.uops))

      batched_inner.append(f"  {fn_name}({','.join([render_arg(buf) for buf in ji.bufs] + [x.expr for x in runner.p.vars])});")

    targs = [(f"arg{i}", (x.dtype.ptr(), False)) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", (dtypes.char.ptr(), False)) for i in range(len(self.base_bufs))] + \
            [(f"fxnp{i}", (dtypes.void.ptr(), False)) for i in range(len(rendered_casts))] + \
            sorted([(f"{v.expr}", (dtypes.int, False)) for v in var_vals])

    batched = ["void batched("+','.join([f"{device.renderer.render_dtype(x[1][0])} {x[0]}" for x in targs])+") {", *batched_inner, "}"]
    entry = device.renderer._render_entry("batched", targs)
    code = '\n'.join(itertools.chain(dedup(defines), dedup(funcs), batched, (entry,)))

    if DEBUG >= 4: print(code)
    self.clprg = device.runtime("batched", device.compiler.compile_cached(code))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)
