import hashlib
from typing import Callable, Dict, Final, Optional
from triton.compiler import compile as triton_compile # type: ignore
import torch
import math

from tinygrad.ops import BinaryOps, Compiled, ASTRunner, FusedOps, Op, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.helpers import DType, ImageDType, dtypes
from tinygrad.shape.symbolic import NumNode
from tinygrad.runtime.lib import RawBuffer

class TritonProgram:
  def __init__(self, name:str, prg:str, signature:str):
    self.name = name
    prg = "import triton\nimport triton.language as tl\n" + prg
    fn = f"/tmp/{hashlib.md5(prg.encode('utf-8')).hexdigest()}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals()) # pylint: disable=W0122
    self.prg = triton_compile(globals()[name], signature=signature)

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait:
      start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      start.record()
    self.prg[tuple(global_size + [1]*(3-len(global_size)))](*[x._buf for x in args])
    if wait:
      end.record()
      torch.cuda.synchronize()
      return start.elapsed_time(end)*1e-3
  
class TritonCodegen(Linearizer):
  supports_float4 = False
  supports_float4_alu = False

  def codegen(self):
    self.process()
    self.limit_global_dims(3)
    self.linearize()

    kernel = []
    global_size = []
    local_size = []
    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    kk("@triton.jit")
    kk("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")
    depth += 1

    gid = [f"tl.program_id({i})" for i in range(3)]
    code_for_op: Final[Dict[Op, Callable]] = {
      UnaryOps.EXP2: lambda x: f"tl.math.exp2({x})",
      UnaryOps.LOG2: lambda x: f"tl.math.log2({x})", # TODO: is fast_log2f ok?
      UnaryOps.SIN: lambda x: f"tl.sin({x})",
      BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})",
      BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})", # fdiv?
      BinaryOps.POW: lambda x,y: f"tl.math.pow({x}, {y})", BinaryOps.MAX: lambda x,y: f"tl.maximum({x},{y})", # axis?
      BinaryOps.CMPEQ: lambda x,y: f"({x}=={y})",
      FusedOps.MULACC: lambda a,b,c: f"tl.math.fma({a}, {b}, {c})"
    }
    bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]

    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode): continue # python doesnt have block scope
          else:
            if args[1] == "global":
              global_size.append(var.max+1)
              kk(f"{var.expr} = {gid[len(args[0])-1-i]} # {var.max+1}")
            elif args[1] == "local":
              local_size.append(var.max+1)
              kk(f"{var.expr} = tl.arange({var.min}, {var.max+1})")
            else:
              kk(f"for {var.expr} in range({var.min}, {var.max+1}):")
              depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] not in ["global", "local"]:
          depth -= 1
          kk(f"# end {args[1]}")
      elif uop == UOps.CONST:
        assert newvar is not None
        if args == -math.inf: kk(f"{newvar.render()} = -math.inf")
        else: kk(f"{newvar.render()} = {args}")
      elif uop == UOps.ALU:
        assert newvar is not None
        kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
      elif uop == UOps.LOAD:
        assert newvar is not None
        val = f"{bufnames[args.i]} + {args.idx.render()}" # defaults to render_python
        if args.valid.min == 1: kk(f"{newvar.render()} = tl.load({val})")#; kk(f"if ridx2 == 0: tl.device_print('test', {newvar.render()})")
        else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({val}, mask={args.valid.render()}), 0.0)")
      elif uop == UOps.STORE:
        assert vin[0].dtype == dtypes.float, "unimplemented: float4 store"
        assert not isinstance(self.bufs[args.i].dtype, ImageDType), "unimplemented: image store"
        assert args.valid.min == 1, "store must be valid"
        kk(f"tl.store({bufnames[args.i]} + {args.idx.render()}, {vin[0].render()})")
      elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
      elif uop == UOps.DEFINE_LOCAL: raise NotImplementedError("unimplemented: define local")
      else:
        raise NotImplementedError(f"unimplemented: {uop}")

    prg = '\n'.join(kernel)
    return ASTRunner("fxn", prg, global_size[::-1] if len(global_size) else [1], op_estimate=self.info.flops, runtime_args={"signature":','.join([{dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.float64: "*fp64", dtypes.int8: "*i8", dtypes.int32: "*i32", dtypes.int64: "*i64"}[buf.dtype] for buf in self.bufs])})

class RawTritonBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[torch.Tensor]=None): super().__init__(size, dtype, buf) if buf is not None else super().__init__(size, dtype, torch.empty(size, dtype={dtypes.float32: torch.float32, dtypes.float16: torch.float16, dtypes.float64: torch.float64, dtypes.int8: torch.int8, dtypes.uint8: torch.uint8, dtypes.int32: torch.int32, dtypes.int64: torch.int64, dtypes.bool: torch.bool}[dtype], device='cuda'))
  @classmethod
  def fromCPU(cls, x): return cls(x.size, dtypes.from_np(x.dtype), buf=torch.from_numpy(x).requires_grad_(False).to('cuda'))
  def toCPU(self): return self._buf.cpu().numpy()

TritonBuffer = Compiled(RawTritonBuffer, TritonCodegen, TritonProgram)
