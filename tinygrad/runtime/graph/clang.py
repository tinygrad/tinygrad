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

    render_dtype = lambda x: device.renderer.render_dtype(x) if isinstance(x, PtrDType) else device.renderer.render_dtype(x)

    self.const_base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b not in input_rawbuffers)
    self.const_base_rawbufs = [b._buf for b in self.const_base_bufs]

    # TODO: dedup
    targs = [(f"arg{i}", (x.dtype.ptr(), False)) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", (dtypes.char.ptr(), False)) for i,x in enumerate(self.const_base_bufs)] + \
            sorted([(f"{v.expr}", (dtypes.int, False)) for v in var_vals])

    mp = {}
    code = ["void batched("+','.join([f"{render_dtype(x[1][0])} {x[0]}" for x in targs])+") {"]
    for i, ji in enumerate(jit_cache):
      args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          args.append(f"arg{input_rawbuffers.index(buf)}")
        else:
          args.append(f"({render_dtype(buf.dtype)}*)(cbuf{self.const_base_bufs.index(buf.base)} + {buf.offset})")
      args += [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      mp[i] = args
      code.append(f"  j{i}{to_function_name(ji.prg.p.name)}({','.join(args)});")
    code.append("}")

    srcs = ['{'.join(device.renderer.render(ji.prg.p.uops).replace(to_function_name(ji.prg.p.name), f"j{i}{to_function_name(ji.prg.p.name)}").split('{')[4:-2]) for i,ji in enumerate(jit_cache)]
    srcs = ['}'.join(src.split("}")[:-1]) + "}" for src in srcs]
    prgs = '\n'.join(['\n\n'.join(src.split("\n\n")[1:]) for src in srcs])

    entry = device.renderer._render_entry("batched", targs) if hasattr(device.renderer, '_render_entry') else ""
    x = device.renderer._render_defines("")+prgs+"\n"+"\n".join(code)+"\n"+entry
    
    if DEBUG >= 4: print(x)
    self.clprg = device.runtime("batched", device.compiler.compile_cached(x))

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.const_base_rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)
