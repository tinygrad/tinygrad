from typing import Tuple, List, NamedTuple, Any, Dict, Optional, Union
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import ASTRunner, FusedOps, BinaryOps, UnaryOps
from tinygrad.helpers import DType, dtypes, DEBUG
from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode
import functools
from collections import defaultdict

type_to_letter = {dtypes.float32: 'f', dtypes.bool: 'p', dtypes.int32: 'i', dtypes.int64: 'a'}

class Register(NamedTuple):
  nm:str
  dtype:DType
  def __repr__(self): return self.nm

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

    cnts = defaultdict(int)
    tor: Dict[Any, Register] = {}
    def newreg(tok, dtype=dtypes.float32):
      nonlocal cnts, tor
      tor[tok] = ret = Register(f"%{type_to_letter[dtype]}{cnts[dtype]}", dtype)
      cnts[dtype] += 1
      return ret

    def render_numnode(b):
      key = ("num", b)
      if key not in tor: ins.append(AssemblyInstruction(UOps.CONST, newreg(key, dtype=dtypes.int32), [], b))
      return tor[key]

    def render_alu(op, a:Register, b:Union[Register, int, float], dtype=dtypes.int32) -> Register:
      key = (op, a, b)
      if key not in tor:
        #if not isinstance(b, Register): b = render_numnode(b)
        ins.append(AssemblyInstruction(UOps.ALU, newreg(key, dtype=dtype), [a, b], op))
      return tor[key]

    def render_cast(a:Register, new_dtype:DType) -> Register:
      if a.dtype == new_dtype: return a
      key = (a, new_dtype)
      if key not in tor:
        ins.append(AssemblyInstruction(UOps.CAST, newreg(key, dtype=new_dtype), [a]))
      return tor[key]

    render_ops = { Variable: lambda self, ops, ctx: tor[self], NumNode: lambda self, ops, ctx: render_numnode(self.b),
                   MulNode: lambda self, ops, ctx: render_alu(BinaryOps.MUL, self.a.render(ops, ctx), self.b),
                   DivNode: lambda self, ops, ctx: render_alu(BinaryOps.DIV, self.a.render(ops, ctx), self.b),
                   ModNode: lambda self, ops, ctx: render_alu(BinaryOps.MOD, self.a.render(ops, ctx), self.b),
                   LtNode: lambda self, ops, ctx: render_alu(BinaryOps.CMPLT, self.a.render(ops, ctx), self.b, dtype=dtypes.bool),
      SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_alu(BinaryOps.ADD, a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
      AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_alu(BinaryOps.MUL, a, b.render(ops,ctx), dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

    def addr_w_offset(args):
      idx = args.idx*self.bufs[args.i].dtype.itemsize
      off = None
      if isinstance(idx, SumNode):
        nums = [n.b for n in idx.nodes if isinstance(n, NumNode)]
        if len(nums) > 0:
          idx -= nums[0]
          off = nums[0]
      reg = idx.render(render_ops)
      reg = render_alu(BinaryOps.ADD, render_cast(reg, dtypes.int64), tor[f"buf{args.i}"], dtype=dtypes.int64)
      return reg, off

    ins = []
    ins += [AssemblyInstruction(UOps.SPECIAL, newreg(f"buf{i}", dtype=dtypes.int64), [], f"buf{i}") for i in range(len(self.bufs))]
    global_size, local_size = [], []
    skipload_branch = 0
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.CONST and newvar is not None:
        ins.append(AssemblyInstruction(UOps.CONST, newreg(newvar), [], args))
      elif uop == UOps.DEFINE_LOCAL:
        raise Exception("not implemented")
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
              ins.append(AssemblyInstruction(UOps.LABEL, None, [], "$loop_"+var.expr))
      elif uop == UOps.ENDLOOP:
        if args[1] not in ["global", "local"]:
          for var in reversed(args[0]):
            pred = render_alu(BinaryOps.CMPLT, tor[var], var.max, dtypes.bool)
            ins.append(AssemblyInstruction(UOps.ALU, tor[var], [tor[var], 1], BinaryOps.ADD))
            ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], ("$loop_"+var.expr, True)))
      elif uop == UOps.ALU and newvar is not None:
        if args == FusedOps.MULACC: vin = [vin[1], vin[2], vin[0]]  # TODO: reorder MULACC everywhere
        # this is the only thing that can violate SSA
        if args in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          pred_reg = newreg((newvar, 'pred'), dtype=dtypes.bool)
          ins.append(AssemblyInstruction(UOps.ALU, pred_reg, [tor[x] for x in vin], args))
          ins.append(AssemblyInstruction(UOps.CAST, newreg(newvar), [pred_reg], args))
        else:
          ins.append(AssemblyInstruction(UOps.ALU, newreg(newvar) if newvar not in tor else tor[newvar], [tor[x] for x in vin], args))
      elif uop == UOps.LOAD and newvar is not None:
        idx, off = addr_w_offset(args)
        reg = newreg(newvar)
        if args.valid.min == 0:
          ins.append(AssemblyInstruction(UOps.CONST, reg, [], 0))
          if args.valid.max == 1:
            pred = args.valid.render(render_ops)
            ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], (f"$skipload_{skipload_branch}", False)))
        if args.valid.max == 1:
          # NOTE: you can't compute the index in here, because it assumes it's all available later
          ins.append(AssemblyInstruction(UOps.LOAD, reg, [idx], off))
        if args.valid.min == 0 and args.valid.max == 1:
          ins.append(AssemblyInstruction(UOps.LABEL, None, [], f"$skipload_{skipload_branch}"))
          skipload_branch += 1
      elif uop == UOps.STORE:
        idx, off = addr_w_offset(args)
        ins.append(AssemblyInstruction(UOps.STORE, None, [idx, tor[vin[0]]], off))

    # define registers
    ins = [AssemblyInstruction(UOps.DEFINE_REGISTER, None, [], (dtype, type_to_letter[dtype], c)) for dtype,c in cnts.items()] + ins

    if DEBUG >= 4:
      for i in ins: print(i)
    name, asm = self.specialize(ins)

    #name, asm, global_size, local_size = self.generate()

    return ASTRunner(name, asm,
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})
