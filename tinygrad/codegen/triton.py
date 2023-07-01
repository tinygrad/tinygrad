from typing import Callable, Dict, Final, Optional
import math, hashlib
from triton.compiler import compile as triton_compile # type: ignore
from tinygrad.ops import BinaryOps, ASTRunner, FusedOps, Op, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.shape.symbolic import NumNode

class TritonCodegen(Linearizer):
  def codegen(self):
    self.process()

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
      UnaryOps.EXP2: lambda x: f"tl.exp2({x})",
      UnaryOps.LOG2: lambda x: f"tl.log2({x})",
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
              if len(args[0]) >= 4 and len(args[0])-i > 2: raise NotImplementedError("unimplemented: global loop with more than 3 dims")
              else:
                global_size.append(var.max+1)
                kk(f"{var.expr} = {gid[len(args[0])-1-i]} # {var.max+1}")
            elif args[1] == "local": raise NotImplementedError("unimplemented: local loop")
            else:
              kk(f"for {var.expr} in range({var.min}, {var.max+1}):")
              depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] == "local": raise NotImplementedError("unimplemented: local loop")
        else:
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
        if args.valid.min == 1: kk(f"{newvar.render()} = tl.load({val})")
        else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({val}, mask={args.valid.render()}), 0.0)")
      elif uop == UOps.STORE:
        assert args.valid.min == 1, "store must be valid"
        kk(f"tl.store({bufnames[args.i]} + {args.idx.render()}, {vin[0].render()})")
      elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
      elif uop == UOps.DEFINE_LOCAL: raise NotImplementedError("unimplemented: define local")
      else:
        raise NotImplementedError(f"unimplemented: {uop}")

    prg = '\n'.join(kernel)
    print(prg)

    # write out python to compile
    signature = ','.join(["*fp32" for _ in range(prg.splitlines()[1].count("data"))])
    prg = "import triton\nimport triton.language as tl\n" + prg
    fn = f"/tmp/{hashlib.md5(prg.encode('utf-8')).hexdigest()}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals()) # pylint: disable=W0122
    triton_prg = triton_compile(globals()["fxn"], signature=signature, device_type="cuda", debug=True)
    asm = triton_prg.asm['ptx']
    name = asm.split(".visible .entry ")[1].split("(")[0]

    return ASTRunner(name, asm,
      global_size[::-1], local_size[::-1],
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})
