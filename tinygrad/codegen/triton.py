from typing import Callable, Dict, Final
import math, hashlib
from triton.compiler import compile as triton_compile # type: ignore
from tinygrad.helpers import DEBUG, prod
from tinygrad.ops import BinaryOps, ASTRunner, Op, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.shape.symbolic import NumNode

class TritonCodegen(Linearizer):
  has_mulacc: bool = False

  def hand_coded_triton_optimizations(self):
    # only locals
    for axis in range(self.first_reduce - self.local_dims - 1, -1, -1):
      local_size = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce])
      if self.full_shape[axis] == 1: continue
      last_try = self.local_dims == 0 and axis == 0
      if any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))) or last_try:
        for sz in [x for x in [32,16,8,4,3] if self.full_shape[axis] % x == 0 and local_size*x <= 1024]:
          self.shift_to(axis, sz, insert_before=self.first_reduce-self.local_dims)
          self.local_dims += 1
          break
      if self.local_dims >= 3: break


  def codegen(self):
    self.process()
    #self.hand_coded_optimizations()
    self.hand_coded_triton_optimizations()
    self.simplify_ones()
    self.limit_global_dims(3)
    self.linearize()

    kernel = []
    global_size = []
    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    kk("@triton.jit")
    kk("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")
    depth += 1

    gid = [f"tl.program_id({i})" for i in range(3)]
    code_for_op: Final[Dict[Op, Callable]] = {
      UnaryOps.EXP2: lambda x: f"tl.exp({x}*{math.log(2)})",
      UnaryOps.LOG2: lambda x: f"(tl.log({x})*{1/math.log(2)})",
      UnaryOps.SIN: lambda x: f"tl.sin({x})",
      BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})",
      BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
      BinaryOps.POW: lambda x,y: f"({x}**{y})", BinaryOps.MAX: lambda x,y: f"tl.maximum({x},{y})", # axis?
      BinaryOps.CMPEQ: lambda x,y: f"(({x}=={y})*1.0)",
    }
    bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]

    full_local_shape = None
    acc_local_shape = 1
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
            elif args[1] == "local":
              full_local_shape = tuple([var.max+1 for var in args[0]])
              assert var.min == 0
              kk(f"{var.expr} = tl.view((tl.arange({0}, {prod(full_local_shape)})//{acc_local_shape})%{var.max+1}, {full_local_shape})")
              acc_local_shape *= var.max+1
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
        if args == -math.inf: ld = "-math.inf"
        else: ld = args
        if full_local_shape: ld = f"tl.full({full_local_shape}, {ld}, tl.float32)"
        kk(f"{newvar.render()} = {ld}")
      elif uop == UOps.ALU:
        assert newvar is not None
        kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
      elif uop == UOps.LOAD:
        assert newvar is not None
        val = f"{bufnames[args.i]} + {args.idx.render()}" # defaults to render_python
        if args.valid.min == 1: ld = f"tl.load({val})"
        else: ld = f"tl.where({args.valid.render()}, tl.load({val}, mask={args.valid.render()}), 0.0)"
        kk(f"{newvar.render()} = {ld}")
      elif uop == UOps.STORE:
        assert args.valid.min == 1, "store must be valid"
        kk(f"tl.store({bufnames[args.i]} + {args.idx.render()}, {vin[0].render()})")
      elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
      elif uop == UOps.DEFINE_LOCAL: raise NotImplementedError("unimplemented: define local")
      else:
        raise NotImplementedError(f"unimplemented: {uop}")

    prg = '\n'.join(kernel)
    if DEBUG >= 4: print(prg)

    # write out python to compile
    signature = ','.join(["*fp32" for _ in range(prg.splitlines()[1].count("data"))])
    prg = "import triton\nimport triton.language as tl\n" + prg
    fn = f"/tmp/{hashlib.md5(prg.encode('utf-8')).hexdigest()}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals()) # pylint: disable=W0122
    triton_prg = triton_compile(globals()["fxn"], signature=signature, device_type="cuda", debug=True)
    asm = triton_prg.asm['ptx']  # ['ast', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
    #print(triton_prg.asm['ttir'])

    # send along the ptx kernel
    name = asm.split(".visible .entry ")[1].split("(")[0]
    local_size = [int(x) for x in asm.split(".maxntid ")[1].split("\n")[0].split(", ")]  # [128, 1, 1] is num_warps=4
    return ASTRunner(name, asm,
      global_size[::-1], local_size,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})
