from typing import Tuple, List, NamedTuple, Any, Dict, Optional, Union, DefaultDict
from tinygrad.codegen.linearizer import Linearizer, UOps, Token
from tinygrad.ops import ASTRunner, BinaryOps, UnaryOps
from tinygrad.helpers import DType, dtypes, DEBUG
from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode
import functools
import math
from collections import defaultdict

_type_to_letter = {dtypes.float32: 'f', dtypes.bool: 'p', dtypes.int32: 'i', dtypes.int64: 'a', dtypes.uint32: 'u', dtypes.uint64: 'b', dtypes._float4: 'x'}
def type_to_letter(x): return _type_to_letter[x[0]].upper() if x[1] else _type_to_letter[x[0]]

class Register(NamedTuple):
  nm:str
  dtype:DType
  scalar:bool
  off:Optional[int] = None
  def __repr__(self): return self.nm if self.off is None else f"{self.nm}:{self.off}"
  def subregs(self):
    if self.dtype == dtypes._float4:
      return [Register(self.nm, dtypes.float, False, off=off) for off in range(4)]
    return []
class AssemblyInstruction(NamedTuple):
  op: UOps
  out: Optional[Register]
  vin: List[Union[Register, int, float]]
  arg: Any = None

# warp size of 32, s registers are shared across the warp, v are 32-wide vectors
class AssemblyCodegen(Linearizer):
  supports_load3: bool = False
  sin_is_sin2pi: bool = False
  no_div: bool = False

  def specialize(self, asm:List[AssemblyInstruction]) -> Tuple[str, str]:
    raise NotImplementedError("must be implemented")

  # s registers are the addresses and non local indexes
  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.limit_global_dims(3)  # all GPU asms have 3 (for now)
    self.linearize()

    cnts:DefaultDict[Tuple[DType, bool], int] = defaultdict(int)
    tor: Dict[Any, Register] = {}
    def newreg(tok, dtype=dtypes.float32, scalar=False):
      nonlocal cnts, tor
      if isinstance(tok, Token): dtype = tok.dtype  # this
      tor[tok] = ret = Register(f"%{type_to_letter((dtype, scalar))}{cnts[(dtype, scalar)]}", dtype, scalar)
      if dtype == dtypes._float4:
        for off in range(4):
          tor[Token(tok.name, tok.dtype, off)] = Register(ret.nm, dtypes.float, ret.scalar, off)
      cnts[(dtype, scalar)] += 1
      return ret

    def render_numnode(b):
      key = ("num", b)
      if key not in tor: ins.append(AssemblyInstruction(UOps.CONST, newreg(key, scalar=True, dtype=dtypes.int32), [], b))
      return tor[key]

    def render_alu(op, a:Register, b:Union[Register, int, float], dtype=dtypes.int32) -> Register:
      key = (op, a, b)
      if key not in tor:
        #if not isinstance(b, Register): b = render_numnode(b)
        ins.append(AssemblyInstruction(UOps.ALU, newreg(key, dtype=dtype, scalar=a.scalar and (not isinstance(b, Register) or b.scalar)), [a, b], op))
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
      off = 0  # TODO: should this be None?
      if isinstance(idx, SumNode):
        nums = [n.b for n in idx.nodes if isinstance(n, NumNode)]
        if len(nums) > 0 and nums[0] < 4096 and (idx-nums[0]).min >= 0:  # TODO: different for each GPU?
          idx -= nums[0]
          off = nums[0]
      reg = idx.render(render_ops)
      if self.supports_load3:
        if reg.scalar:
          new_reg = newreg((reg.nm, 'vec'), dtype=reg.dtype)
          ins.append(AssemblyInstruction(UOps.ALU, new_reg, [reg], UnaryOps.NOOP))
          reg = new_reg
        return tor[f"buf{args.i}"], reg, off
      else:
        reg = render_alu(BinaryOps.ADD, render_cast(reg, dtypes.uint64), tor[f"buf{args.i}"], dtype=dtypes.uint64)
        return reg, None, off

    ins = []
    ins += [AssemblyInstruction(UOps.SPECIAL, newreg(f"buf{i}", dtype=dtypes.uint64, scalar=True), [], f"buf{i}") for i in range(len(self.bufs))]
    global_size, local_size = [], []
    skipload_branch = 0
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.CONST and newvar is not None:
        ins.append(AssemblyInstruction(UOps.CONST, newreg(newvar, dtype=newvar.dtype), [], args))
      elif uop == UOps.DEFINE_LOCAL:
        ins.append(AssemblyInstruction(UOps.DEFINE_LOCAL, None, [], args))
        ins.append(AssemblyInstruction(UOps.ALU, newreg("buf-1", dtype=dtypes.uint64), [args[0]], UnaryOps.NOOP))
      elif uop == UOps.LOOP:
        if args[1] == "global":
          for i,var in enumerate(args[0]):
            global_size.append(var.max+1)
            ins.append(AssemblyInstruction(UOps.SPECIAL, newreg(var, dtype=dtypes.int32), [], f"gid{len(args[0])-1-i}"))
        elif args[1] == "local":
          for i,var in enumerate(args[0]):
            local_size.append(var.max+1)
            global_size[i] *= local_size[i]
            ins.append(AssemblyInstruction(UOps.SPECIAL, newreg(var, dtype=dtypes.int32), [], f"lid{len(args[0])-1-i}"))
        else:
          for var in args[0]:
            if not isinstance(var, NumNode):  # TODO: why is this coming through?
              ins.append(AssemblyInstruction(UOps.CONST, newreg(var, dtype=dtypes.int32, scalar=True), [], 0))
              ins.append(AssemblyInstruction(UOps.LABEL, None, [], "$loop_"+var.expr))
      elif uop == UOps.ENDLOOP:
        if args[1] not in ["global", "local"]:
          for var in reversed(args[0]):
            if not isinstance(var, NumNode):  # TODO: why is this coming through?
              ins.append(AssemblyInstruction(UOps.ALU, tor[var], [tor[var], 1], BinaryOps.ADD))
              pred = render_alu(BinaryOps.CMPLT, tor[var], var.max+1, dtypes.bool)
              ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], ("$loop_"+var.expr, True)))
      elif uop == UOps.CAST and newvar is not None:
        # TODO: we should reconsider outputting CAST in the linearizer. these are needless copies
        out = newreg(newvar)
        for i,sr in enumerate(out.subregs()):
          ins.append(AssemblyInstruction(UOps.ALU, sr, [tor[vin[i]]], UnaryOps.NOOP))
      elif uop == UOps.ALU and newvar is not None:
        out = newreg(newvar) if newvar not in tor else tor[newvar]
        # this is the only thing that can violate SSA
        if args in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          pred_reg = newreg((newvar, 'pred'), dtype=dtypes.bool)
          ins.append(AssemblyInstruction(UOps.ALU, pred_reg, [tor[x] for x in vin], args))
          ins.append(AssemblyInstruction(UOps.CAST, out, [pred_reg], args))
        elif args == BinaryOps.POW:
          # TODO: add UnaryOps.SQRT
          tmp = newreg((newvar, "exp_a"))
          tmp2 = newreg((newvar, "exp_a_times_b"))
          ins.append(AssemblyInstruction(UOps.ALU, tmp, [tor[vin[0]]], UnaryOps.LOG2))
          ins.append(AssemblyInstruction(UOps.ALU, tmp2, [tmp, tor[vin[1]]], BinaryOps.MUL))
          ins.append(AssemblyInstruction(UOps.ALU, out, [tmp2], UnaryOps.EXP2))
        elif args == BinaryOps.DIV and self.no_div:
          tmp = newreg((newvar, "rcp"))
          ins.append(AssemblyInstruction(UOps.ALU, tmp, [tor[vin[1]]], UnaryOps.RECIP))
          ins.append(AssemblyInstruction(UOps.ALU, out, [tor[vin[0]], tmp], BinaryOps.MUL))
        elif args == UnaryOps.SIN and self.sin_is_sin2pi:
          tmp = newreg((newvar, "2pi"))
          ins.append(AssemblyInstruction(UOps.ALU, tmp, [tor[vin[0]], 1/(math.pi*2)], BinaryOps.MUL))
          ins.append(AssemblyInstruction(UOps.ALU, out, [tmp], args))
        else:
          ins.append(AssemblyInstruction(UOps.ALU, out, [tor[x] for x in vin], args))
      elif uop == UOps.LOAD and newvar is not None:
        idx, treg, off = addr_w_offset(args)
        reg = newreg(newvar, dtype=newvar.dtype, scalar=(idx.scalar and (not isinstance(treg, Register) or treg.scalar))) # and not dtypes.is_float(newvar.dtype)))
        if args.valid.min == 0:
          ins.append(AssemblyInstruction(UOps.CONST, reg, [], 0))
          if args.valid.max == 1:
            pred = args.valid.render(render_ops)
            ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], (f"$skipload_{skipload_branch}", False)))
        if args.valid.max == 1:
          # NOTE: you can't compute the index in here, because it assumes it's all available later
          ins.append(AssemblyInstruction(UOps.LOAD, reg, [idx] + ([treg] if treg is not None else []), (off, 'global' if args.i != -1 else 'shared')))
        if args.valid.min == 0 and args.valid.max == 1:
          ins.append(AssemblyInstruction(UOps.LABEL, None, [], f"$skipload_{skipload_branch}"))
          skipload_branch += 1
      elif uop == UOps.STORE:
        idx, treg, off = addr_w_offset(args)
        ins.append(AssemblyInstruction(UOps.STORE, None, [idx, tor[vin[0]]] + ([treg] if treg is not None else []), (off, 'global' if args.i != -1 else 'shared')))

    # define registers
    ins = [AssemblyInstruction(UOps.DEFINE_REGISTER, None, [], (dtype, type_to_letter(dtype), c)) for dtype,c in cnts.items()] + ins

    if DEBUG >= 4:
      for tins in ins: print(tins)
    name, asm = self.specialize(ins)

    return ASTRunner(name, asm,
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})
