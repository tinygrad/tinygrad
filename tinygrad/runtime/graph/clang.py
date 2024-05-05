from typing import List, Dict, cast
import ctypes
from tinygrad.helpers import dedup, cpu_time_execution, GraphException, DEBUG
from tinygrad.engine.jit import GraphRunner
from tinygrad.device import Buffer, Device, CompiledRunner
from tinygrad.engine.realize import ExecItem
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_clang import ClangProgram
from tinygrad.renderer.cstyle import ClangLanguage
render_dtype = ClangLanguage().render_dtype

class ClangGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    prgs = '\n'.join(dedup([cast(CompiledRunner, ji.prg).prg for ji in jit_cache]))
    args = [f"{render_dtype(x.dtype)}* arg{i}" for i,x in enumerate(input_rawbuffers)]
    args += [f"int {v.expr}" for v in var_vals]
    code = ["void batched("+','.join(args)+") {"]
    for ji in jit_cache:
      args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          args.append(f"arg{input_rawbuffers.index(buf)}")
        else:
          args.append(f"({render_dtype(buf.dtype)}*)0x{ctypes.addressof(buf._buf):X}")
      args += [x.expr for x in cast(CompiledRunner, ji.prg).vars]
      code.append(f"  {cast(CompiledRunner, ji.prg).name}({','.join(args)});")
    code.append("}")
    if DEBUG >= 4: print("\n".join(code))
    compiler = Device["CLANG"].compiler
    assert compiler is not None
    self.clprg = ClangProgram("batched", compiler.compile(prgs+"\n"+"\n".join(code))) # no point in caching the pointers

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return cpu_time_execution(lambda: self.clprg(*[x._buf for x in rawbufs], *[x for x in var_vals.values()]), enable=wait)