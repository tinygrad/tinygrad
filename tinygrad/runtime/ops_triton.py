import hashlib
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
import triton
import triton.language as tl
from triton.compiler import compile as triton_compile
from triton.runtime import JITFunction
import torch
import math

from tinygrad.ops import BinaryOps, Compiled, ASTRunner, FusedOps, LazyOp, Op, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, LocalTypes, UOp, UOps
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, DType, ImageDType, prod, dtypes
from tinygrad.runtime.lib import RawBuffer, RawConst
from tinygrad.shape.shapetracker import MovementOps
from tinygrad.shape.symbolic import NumNode

class TritonProgram:
  def __init__(self, name:str, prg:str):
    self.name = name
    # hack to get the signature
    signature = ','.join(["*fp32" for _ in range(prg.splitlines()[1].count("data"))])
    hash = hashlib.md5(prg.encode("utf-8")).hexdigest()
    fn = f"/tmp/{hash}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals())
    self.prg = triton_compile(globals()[name], signature=signature)

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait:
      start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      start.record()
    print([1]*(3-len(global_size)) + global_size)
    self.prg[tuple([1]*(3-len(global_size)) + global_size)](*[x._buf for x in args]) # TODO: detect launch params
    if wait:
      end.record()
      torch.cuda.synchronize()
      return start.elapsed_time(end)*1e-3
  
class TritonCodegen(Linearizer):
  def codegen(self):
    self.process()

    self.linearize()

    loaded = set()

    kernel = []
    global_size = []
    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    kk("@triton.jit")
    kk("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")
    depth += 1

    output_shape = self.info.shape

    # copied from old ops_triton
    kernel += [f"  idx{len(output_shape)-1-i} = tl.program_id({i})" for i in range(3)]

    gid = [f"idx{i}" for i in range(3)]
    code_for_op: Final[Dict[Op, Callable]] = {
      UnaryOps.EXP: lambda x: f"tl.exp({x})",
      UnaryOps.LOG: lambda x: f"tl.log({x})",
      UnaryOps.SIN: lambda x: f"tl.sin({x})",
      BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})",
      BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
      BinaryOps.POW: lambda x,y: f"({x}**{y})", BinaryOps.MAX: lambda x,y: f"tl.max({x},{y})", # axis?
      BinaryOps.CMPEQ: lambda x,y: f"({x}=={y}).astype(np.float32)",
      FusedOps.MULACC: lambda s,a,b: f"({a}*{b}) + {s}",
    }
    bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]

    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode): continue # python doesnt have block scope
          else:
            if args[1] == "global":
              if len(args[0]) >= 4 and len(args[0])-i > 2: raise Exception("unimplemented: global loop with more than 3 dims")
              else:
                global_size.append(var.max+1)
                kk(f"{var.expr} = {gid[len(args[0])-1-i]} # {var.max+1}")
            elif args[1] == "local": raise Exception("unimplemented: local loop")
            else:
              kk(f"for {var.expr} in range({var.min}, {var.max+1}):")
              depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] == "local": raise Exception("unimplemented: local loop")
        else:
          depth -= 1
          kk(f"# end {args[1]}")
      elif uop == UOps.CONST:
        assert newvar is not None
        if args == -math.inf: kk(f"{newvar.render()} = -math.inf")
        else: kk(f"{newvar.render()} = {args}")
      elif uop == UOps.ALU:
        assert newvar is not None
        if newvar in vin: kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
      elif uop == UOps.LOAD:
        assert newvar is not None
        val = f"{bufnames[args.i]} + {args.idx.render()}" # defaults to render_python
        print(args.valid)
        if args.valid.min == 1: kk(f"{newvar.render()} = tl.load({val})")
        else: kk(f"{newvar.render()} = tl.load({val}) if ({args.valid.render()}) else 0.")
      elif uop == UOps.STORE:
        assert vin[0].ltype == LocalTypes.float, "unimplemented: float4 store"
        assert not isinstance(self.bufs[args.i].dtype, ImageDType), "unimplemented: image store"
        assert args.valid.min == 1, "store must be valid"
        kk(f"tl.store({bufnames[args.i]} + {args.idx.render()}, {vin[0].render()})")
      elif uop == UOps.CAST: raise Exception("unimplemented: cast")
      elif uop == UOps.DEFINE_LOCAL: raise Exception("unimplemented: define local")
      else:
        raise Exception(f"unimplemented: {uop}")

    prg = '\n'.join(kernel)
    return ASTRunner("fxn", prg, global_size[::-1] if len(global_size) else [1], op_estimate=self.info.flops)

class RawTritonBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[torch.Tensor]=None): super().__init__(size, dtype, buf) if buf is not None else super().__init__(size, dtype, torch.empty(size, dtype=torch.float, device='cuda'))
  @classmethod
  def fromCPU(cls, x): return cls(x.size, dtypes.from_np(x.dtype), buf=torch.from_numpy(x).requires_grad_(False).to('cuda'))
  def toCPU(self): return self._buf.cpu().numpy()

TritonBuffer = Compiled(RawTritonBuffer, TritonCodegen, TritonProgram)
