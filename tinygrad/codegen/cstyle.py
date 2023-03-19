from typing import Final, Dict, Callable
import math
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import ASTRunner, Op, UnaryOps, BinaryOps
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops, ctx)}/{self.b})"
render_cl[AndNode] = lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"

NATIVE_EXPLOG = getenv("NATIVE_EXPLOG", 0)  # this is needed as a switch for the tests to pass

class CStyleCodegen(Linearizer):
  code_for_op: Final[Dict[Op, Callable]] = {
    UnaryOps.EXP: lambda x: f"native_exp({x})" if NATIVE_EXPLOG else f"exp({x})",
    UnaryOps.LOG: lambda x: f"native_log({x})" if NATIVE_EXPLOG else f"log({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.POW: lambda a,b: f"pow({a},{b})", BinaryOps.MAX: lambda a,b: f"max({a},{b})",
    BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})",
  }

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.linearize()

    kernel = []
    global_size = []
    local_size = []
    pend_close = None

    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    for uop,newvar,args in self.uops:
      if uop == UOps.LOOP:
        root = None
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode):
            if args[1] == "global" and self.lang.gid: global_size.append(1)
            if args[1] == "local" and self.lang.lid: local_size.append(1)
            # one number, not an index
            kk("{")
          else:
            if args[1] == "global" and self.lang.gid:
              if len(args[0]) >= 4 and len(args[0])-i > 2:
                # sometimes, there's more dimensions. compact all the dimensions into the last CL dimension
                # TODO: these compactions should be searchable (they sort of are with reshapes and permutes)
                if i == 0:
                  kk(f"{{ int {var.expr} = {self.lang.gid[-1]}; /* {var.max+1} */")
                  root = var.expr
                  global_size.append(var.max+1)
                else:
                  kk(f"{{ int {var.expr} = {root} % {var.max+1}; {root} /= {var.max+1};")
                  global_size[-1] *= var.max+1
              else:
                kk(f"{{ int {var.expr} = {self.lang.gid[len(args[0])-1-i]};  /* {var.max+1} */")
                global_size.append(var.max+1)
            elif args[1] == "local" and self.lang.lid:
              kk(f"{{ int {var.expr} = {self.lang.lid[len(args[0])-1-i]};  /* {var.max+1} */")
              local_size.append(var.max+1)
            else:
              kk(f"for (int {var.expr} = {var.min}; {var.expr} <= {var.max}; ++{var.expr}) {{")
        depth += 1
      if uop == UOps.ENDLOOP:
        depth -= 1
        if args[1] == "local" and len(self.lang.lid):
          kk(self.lang.barrier)
          kk(f"if ({Variable.sum(args[0]).render(render_cl)} == 0) {{")
          pend_close = "}"*(len(args[0])+1) + f" /* {args[1]} */"
        else:
          if args[1] == "global" and pend_close:
            kernel.append(pend_close)
            pend_close = None
          kk("}"*len(args[0]) + f" /* {args[1]} */")
      if uop == UOps.CONST:
        if args[0] == -math.inf:
          kk(f"float {newvar} = -INFINITY;")
        else:
          kk(f"float {newvar} = {args[0]}f;")
      if uop == UOps.ALU:
        if newvar is None:
          kk(f"{args[2]} = {self.code_for_op[args[0]](*args[1])};")
        else:
          kk(f"float {newvar} = {self.code_for_op[args[0]](*args[1])};")
      if uop == UOps.LOAD:
        # NOTE: if min and max are both 0, it should be a CONST in the Linearizer
        if args[2].min == 1:
          kk(f"float {newvar} = {args[0]}[{args[1].render(render_cl)}];")
        else:
          kk(f"float {newvar} = ({args[2].render(render_cl)}) ? ({args[0]}[{args[1].render(render_cl)}]) : 0.0f;")
      if uop == UOps.STORE:
        kk(f"{args[0]}[{args[1].render(render_cl)}] = {args[3]};")
      if uop == UOps.DEFINE_LOCAL:
        kk(self.lang.smem_prefix + f"float {args[0]}[{args[1]}];")

    buftypes = [(i,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if x.dtype.name.startswith('image') else self.lang.buffer_prefix+"float*"+self.lang.buffer_suffix) for i,x in enumerate(self.bufs) if x is not None and not isinstance(x.realized, RawConst)]
    prg = ''.join([f"{self.lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
      [', '.join([f'{t} data{i}' for i,t in buftypes] + self.lang.extra_args)] +
      [") {\n"] + ['\n'.join(kernel), "\n}"])

    if DEBUG >= 3:
      print(prg)
    print(global_size, local_size)

    function_name = "exec"
    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name),
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops,
      mem_estimate=sum(x.dtype.itemsize*(x.realized.size if x.realized is not None else prod(x.shape)) for x in self.bufs if x is not None))