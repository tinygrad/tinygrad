from typing import List, Dict, cast
import ctypes
from tinygrad.helpers import dedup, cpu_time_execution, DEBUG
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import dtypes, PtrDType

class ClangGraph(GraphRunner):
  def __init__(self, device, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    render_dtype = lambda x: device.renderer.render_dtype(x)+"*" if isinstance(x, PtrDType) else device.renderer.render_dtype(x)

    self.const_base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b not in input_rawbuffers)
    self.const_base_rawbufs = [b._buf for b in self.const_base_bufs]

    prgs = '\n'.join(dedup([cast(CompiledRunner, ji.prg).p.src for ji in jit_cache]))

    targs = [(f"arg{i}", PtrDType(x.dtype)) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", PtrDType(dtypes.char)) for i,x in enumerate(self.const_base_bufs)] + \
            sorted([(f"{v.expr}", dtypes.int) for v in var_vals])

    code = ["void batched("+','.join([f"{render_dtype(x[1])} {x[0]}" for x in targs])+") {"]
    for ji in jit_cache:
      args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          args.append(f"arg{input_rawbuffers.index(buf)}")
        else:
          args.append(f"({render_dtype(buf.dtype)}*)(cbuf{self.const_base_bufs.index(buf.base)} + {buf.offset})")
      args += [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      code.append(f"  {cast(CompiledRunner, ji.prg).p.function_name}({','.join(args)});")
    code.append("}")

    entry_jump = device.renderer._render_entry("batched", []) if hasattr(device.renderer, '_render_entry') else ""

    if DEBUG >= 1: print("\n".join(code))
    self.clprg = device.runtime("batched", device.compiler.compile_cached(prgs+"\n"+"\n".join(code)+"\n"+entry_jump))

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.const_base_rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)
