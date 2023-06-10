from typing import Tuple, List, NamedTuple, Any, Dict, Optional
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import ASTRunner, FusedOps, BinaryOps
from tinygrad.helpers import DType, dtypes
from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode
import functools

class Register(NamedTuple):
  num:int
  dtype:DType
  scalar:bool   # shared across the wave
  def __repr__(self): return f"%{('v' if not self.scalar else 's') if self.dtype == dtypes.float32 else ('p' if self.dtype == dtypes.bool else 'i')}{self.num}"

class AssemblyInstruction(NamedTuple):
  op: UOps
  out: Optional[Register]
  vin: List[Register]
  arg: Any = None

# warp size of 32, s registers are shared across the warp, v are 32-wide vectors
class AssemblyCodegen(Linearizer):
  def generate(self) -> Tuple[str, str, List[int], List[int]]:
    raise NotImplementedError("must be implemented")

  # s registers are the addresses and non local indexes
  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.limit_global_dims(3)  # all GPU asms have 3 (for now)
    self.linearize()

    num = 0
    tor: Dict[Any, Register] = {}
    def newreg(tok, dtype=dtypes.float32, scalar=False):
      nonlocal num, tor
      tor[tok] = ret = Register(num, dtype, scalar)
      num += 1
      return ret

    def render_numnode(self, ops, ctx):
      if self not in tor: ins.append(AssemblyInstruction(UOps.CONST, newreg(self, dtype=dtypes.int32), [], self.b))
      return tor[self]

    def render_alu(op, a, b):
      key = (op, a, b)
      if key not in tor:
        ins.append(AssemblyInstruction(op, newreg(key, dtype=dtypes.int32), [a, b]))
      return tor[key]

    render_ops = { Variable: lambda self, ops, ctx: tor[self], NumNode: render_numnode,
                   MulNode: lambda self, ops, ctx: render_alu(BinaryOps.MUL, self.a.render(ops, ctx), self.b),
                   DivNode: lambda self, ops, ctx: render_alu(BinaryOps.DIV, self.a.render(ops, ctx), self.b),
                   ModNode: lambda self, ops, ctx: render_alu(BinaryOps.MOD, self.a.render(ops, ctx), self.b),
                   LtNode: lambda self, ops, ctx: render_alu(BinaryOps.CMPLT, self.a.render(ops, ctx), self.b),
      SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_alu(BinaryOps.ADD, a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
      AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_alu(BinaryOps.MUL, a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)) }


    ins = []
    global_size, local_size = [], []
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.CONST and newvar is not None:
        ins.append(AssemblyInstruction(UOps.CONST, newreg(newvar), [], args))
      elif uop == UOps.LOOP:
        if args[1] == "global":
          for i,var in enumerate(args[0]):
            global_size.append(var.max+1)
            ins.append(AssemblyInstruction(UOps.SPECIAL, newreg(var, dtype=dtypes.int32), [], f"gid{len(args[0])-1-i}"))
        elif args[1] == "local":
          for i,var in enumerate(args[0]):
            local_size.append(var.max+1)
            ins.append(AssemblyInstruction(UOps.SPECIAL, newreg(var, dtype=dtypes.int32), [], f"lid{len(args[0])-1-i}"))
        else:
          for var in args[0]:
            if not isinstance(var, NumNode):  # TODO: why is this coming through?
              ins.append(AssemblyInstruction(UOps.CONST, newreg(var, dtype=dtypes.int32), [], 0))
      elif uop == UOps.ALU and newvar is not None:
        if args == FusedOps.MULACC: vin = [vin[1], vin[2], vin[0]]  # TODO: reorder MULACC everywhere
        ins.append(AssemblyInstruction(UOps.ALU, newreg(newvar), [tor[x] for x in vin], args))
      elif uop == UOps.LOAD and newvar is not None:
        idx = args.idx*self.bufs[args.i].dtype.itemsize
        args = None
        if isinstance(idx, SumNode):
          nums = [n.b for n in idx.nodes if isinstance(n, NumNode)]
          if len(nums) > 0:
            idx -= nums[0]
            args = nums[0]
        ins.append(AssemblyInstruction(UOps.LOAD, newreg(newvar), [idx.render(render_ops)], args))
      elif uop == UOps.STORE:
        ins.append(AssemblyInstruction(UOps.STORE, None, [tor[vin[0]]]))

    for i in ins: print(i)
    name, asm = "test", []

    name, asm, global_size, local_size = self.generate()

    return ASTRunner(name, asm,
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})
