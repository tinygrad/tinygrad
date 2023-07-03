from typing import Callable, Dict, Final
from triton.compiler import compile as triton_compile # type: ignore
import hashlib
import math

from tinygrad.ops import BinaryOps, ASTRunner, FusedOps, Op, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.helpers import DEBUG, ImageDType, dtypes
from tinygrad.shape.symbolic import NumNode

class TritonCodegen(Linearizer):
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
              kk(f"{var.expr} = {gid[i]} # {var.max+1}")
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
        triton_dtype = {dtypes.float32: "tl.float32", dtypes.float16: "tl.float16", dtypes.float64: "tl.float64", dtypes.int8: "tl.int8", dtypes.uint8: "tl.uint8", dtypes.int32: "tl.int32", dtypes.int64: "tl.int64"}[newvar.dtype]
        if args.valid.min == 1: kk(f"{newvar.render()} = tl.load({val}).to({triton_dtype})")#; kk(f"if ridx2 == 0: tl.device_print('test', {newvar.render()})")
        else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({val}, mask={args.valid.render()}), 0.0).to({triton_dtype})")
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
    if DEBUG >=5: print(prg)

    # write out python to compile
    prg = "import triton\nimport triton.language as tl\n" + prg
    fn = f"/tmp/{hashlib.md5(prg.encode('utf-8')).hexdigest()}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals()) # pylint: disable=W0122
    triton_prg = triton_compile(globals()["fxn"], signature=','.join([{dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.float64: "*fp64", dtypes.int8: "*i8", dtypes.uint8: "*u8", dtypes.int32: "*i32", dtypes.int64: "*i64"}[buf.dtype] for buf in self.bufs]), device_type="cuda", debug=True)
    asm = triton_prg.asm['ptx']
    name = asm.split(".visible .entry ")[1].split("(")[0]

    return ASTRunner(name, asm,
      global_size, local_size,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True, "shared": triton_prg.metadata['shared']})
