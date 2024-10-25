from typing import List, Dict, cast
import ctypes
from tinygrad.helpers import OSX, cpu_time_execution, DEBUG, dedup
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.ops import Variable
from tinygrad.runtime.ops_clang import ClangProgram, ClangProgramOSX
from tinygrad.renderer.cstyle import ClangRenderer
render_dtype = ClangRenderer().render_dtype

class ClangGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    args = [f"{render_dtype(x.dtype)}* arg{i}" for i,x in enumerate(input_rawbuffers)]
    args += sorted([f"int {v.expr}" for v in var_vals])
    code = ["\nvoid batched("+','.join(args)+") " + ("__attribute__((section(\".tinygrad\")))" if not OSX else "") + " {"]
    i = 0
    for ji in jit_cache:
      kernel_args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          kernel_args.append((render_dtype(buf.dtype), f"arg{input_rawbuffers.index(buf)}", True))
        else:
          kernel_args.append((render_dtype(buf.dtype), f"({render_dtype(buf.dtype)}*)0x{ctypes.addressof(buf._buf):X}", True))
      variables = cast(CompiledRunner, ji.prg).p.vars
      kernel_args = kernel_args + [(render_dtype(v.dtype), v.expr, False) for v in variables]
      code.append(f"  typedef void kernel{i}_t(" + ",".join([(arg[0] + ("*" if arg[2] else "")) for arg in kernel_args]) + ");\n")
      code.append(f"  ((kernel{i}_t*)0x{cast(CompiledRunner, ji.prg).clprg.buf:X})({','.join([arg[1] for arg in kernel_args])});")
      i+=1
    code.append("}")
    if DEBUG >= 4: print("\n".join(code))
    compiler = Device["CLANG"].compiler
    assert compiler is not None
    self.clprg = ClangProgram("batched", compiler.compile("\n".join(code))) # no point in caching the pointers

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return cpu_time_execution(
    lambda: self.clprg(*[x._buf for x in rawbufs], *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)]), enable=wait)

class ClangGraphOSX(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    prgs = '\n'.join(dedup([cast(CompiledRunner, ji.prg).p.src for ji in jit_cache]))
    args = [f"{render_dtype(x.dtype)}* arg{i}" for i,x in enumerate(input_rawbuffers)]
    args += sorted([f"int {v.expr}" for v in var_vals])
    code = ["void batched("+','.join(args)+") {"]
    for ji in jit_cache:
      args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          args.append(f"arg{input_rawbuffers.index(buf)}")
        else:
          args.append(f"({render_dtype(buf.dtype)}*)0x{ctypes.addressof(buf._buf):X}")
      args += [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      code.append(f"  {cast(CompiledRunner, ji.prg).p.function_name}({','.join(args)});")
    code.append("}")
    if DEBUG >= 4: print("\n".join(code))
    compiler = Device["CLANG"].compiler
    assert compiler is not None
    self.clprg = ClangProgramOSX("batched", compiler.compile(prgs+"\n"+"\n".join(code))) # no point in caching the pointers

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return cpu_time_execution(
    lambda: self.clprg(*[x._buf for x in rawbufs], *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)]), enable=wait)
