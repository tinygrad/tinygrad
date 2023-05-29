import hashlib
from typing import Any, Callable, Dict, Final, List, Tuple, Union
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
  def __init__(self, name:str, kernel:str, signature:str, key):
    self.name, self.kernel, self.signature, self.key = name, kernel, signature, key
    hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    self.fn = f"/tmp/{hash}.py"
    with open(self.fn, "w") as f: f.write(kernel)
    if DEBUG >= 4: print(kernel)

  def build(self):
    self.name, codeobj = self.name, compile(self.kernel, self.fn, "exec")
    exec(codeobj, globals())
    print(globals()[self.name])
    self.prg = triton_compile(globals()[self.name], signature=self.signature)

class TritonCodegen(Linearizer):
  def codegen(self):
    self.process()

    self.linearize()

    loaded = set()

    kernel = []
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
    return ("fxn", prg, ','.join(["*fp32" for _ in range(len(self.bufs))]), self.key)

class RawTritonBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:torch.Tensor): super().__init__(size, dtype, buf)
  @classmethod
  def fromCPU(cls, x): return cls(x.size, dtypes.from_np(x.dtype), buf=torch.from_numpy(x).requires_grad_(False).to('cuda'))
  def toCPU(self): return self._buf.cpu().numpy()

class _TritonBuffer:
  def __init__(self):
    self.buffer = RawTritonBuffer
  
  def exec_ast(self, ast:LazyOp, output, **kwargs):
    if ast.op in MovementOps and not isinstance(ast.src[0], LazyOp) and ast.src[0].realized is not None: return ast.src[0].realized

    output.realized = RawTritonBuffer(prod(output.shape), output.dtype, torch.empty(*output.shape, dtype=torch.float))

    k = TritonCodegen(ast, output)

    prg = TritonProgram(*k.codegen())

    prg.build()

    print([x.realized._buf.shape for x in k.bufs if x.realized is not None])

    print(tuple(reversed(output.shape)))

    prg.prg[(1,10,30)](*[x.realized._buf.cuda() for x in k.bufs if x.realized is not None and not isinstance(x.realized, RawConst)])

    return output.realized

TritonBuffer = _TritonBuffer()
