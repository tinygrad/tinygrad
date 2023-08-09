from typing import Tuple, List, NamedTuple, Any, Dict, Optional, Union, DefaultDict
from tinygrad.codegen.linearizer import Linearizer, UOps, Token, ConstOp, MemOp, UOp
from tinygrad.ops import BinaryOps, UnaryOps
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
class AssemblyLanguage(NamedTuple):
  supports_load3: bool = False
  sin_is_sin2pi: bool = False
  no_div: bool = False

  #  FIXME: cleanup?
  # stateful
  cnts:DefaultDict[Tuple[DType, bool], int] = defaultdict(int)
  tor: Dict[Any, Register] = {}
  bufs_cnt = 0
  ins = []

  def newreg(self, tok, dtype=dtypes.float32, scalar=False):
    if isinstance(tok, Token): dtype = tok.dtype  # this
    self.tor[tok] = ret = Register(f"%{type_to_letter((dtype, scalar))}{self.cnts[(dtype, scalar)]}", dtype, scalar)
    if dtype == dtypes._float4:
      for off in range(4):
        self.tor[Token(tok.name, tok.dtype, off)] = Register(ret.nm, dtypes.float, ret.scalar, off)
    self.cnts[(dtype, scalar)] += 1
    return ret

  def render_numnode(self, b):
    key = ("num", b)
    # if key not in self.tor: self.ins.append(AssemblyInstruction(UOps.CONST, self.newreg(key, scalar=True, dtype=dtypes.int32), [], b))
    if key not in self.tor: self.ins.append(AssemblyInstruction(UOps.LOAD, self.newreg(key, scalar=True, dtype=dtypes.int32), [], ConstOp(b, None))) # FIXME: what should valid be
    return self.tor[key]

  def render_alu(self, op, a:Register, b:Union[Register, int, float], dtype=dtypes.int32) -> Register:
    key = (op, a, b)
    if key not in self.tor:
      #if not isinstance(b, Register): b = render_numnode(b)
      self.ins.append(AssemblyInstruction(UOps.ALU, self.newreg(key, dtype=dtype, scalar=a.scalar and (not isinstance(b, Register) or b.scalar)), [a, b], op))
    return self.tor[key]

  def render_cast(self, a:Register, new_dtype:DType) -> Register:
    if a.dtype == new_dtype: return a
    key = (a, new_dtype)
    if key not in self.tor:
      self.ins.append(AssemblyInstruction(UOps.CAST, self.newreg(key, dtype=new_dtype), [a]))
    return self.tor[key]

  render_ops = { Variable: lambda self, ops, ctx: ctx.tor[self], NumNode: lambda self, ops, ctx: ctx.render_numnode(self.b),
                 MulNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.MUL, self.a.render(ops, ctx), self.b),
                 DivNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.DIV, self.a.render(ops, ctx), self.b),
                 ModNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.MOD, self.a.render(ops, ctx), self.b),
                 LtNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.CMPLT, self.a.render(ops, ctx), self.b, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.render_alu(BinaryOps.ADD, a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.render_alu(BinaryOps.MUL, a, b.render(ops,ctx), dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def addr_w_offset(self, args):
    assert isinstance(args, MemOp)
    # idx = args.idx*self.bufs[args.i].dtype.itemsize
    idx = args.idx*args.memory_dtype.itemsize
    off = 0  # TODO: should this be None?
    if isinstance(idx, SumNode):
      nums = [n.b for n in idx.nodes if isinstance(n, NumNode)]
      if len(nums) > 0 and nums[0] < 4096 and (idx-nums[0]).min >= 0:  # TODO: different for each GPU?
        idx -= nums[0]
        off = nums[0]
    reg = idx.render(self.render_ops, self)
    if self.supports_load3:
      if reg.scalar:
        new_reg = self.newreg((reg.nm, 'vec'), dtype=reg.dtype)
        self.ins.append(AssemblyInstruction(UOps.ALU, new_reg, [reg], UnaryOps.NOOP))
        reg = new_reg
      # return self.tor[f"buf{args.i}"], reg, off
      return self.tor[args.name], reg, off
    # reg = self.render_alu(BinaryOps.ADD, self.render_cast(reg, dtypes.uint64), self.tor[f"buf{args.i}"], dtype=dtypes.uint64)
    reg = self.render_alu(BinaryOps.ADD, self.render_cast(reg, dtypes.uint64), self.tor[args.name], dtype=dtypes.uint64)
    return reg, None, off

# s registers are the addresses and non local indexes
def uops_to_asmstyle(lang, function_name:str, uops:List[UOp]):
  lang.ins.clear()
  lang.tor.clear()
  lang.cnts.clear()
  # FIXME: Think this is taken care of by DEFINE_GLOBAL now?
  # ins += [AssemblyInstruction(UOps.SPECIAL, newreg(f"buf{i}", dtype=dtypes.uint64, scalar=True), [], f"buf{i}") for i in range(len(self.bufs))]
  global_size, local_size = [], []
  skipload_branch = 0
  for uop,newvar,vin,args in uops:
    if uop == UOps.DEFINE_GLOBAL:
      lang.bufs_cnt += 1
      # lang.ins.append(AssemblyInstruction(UOps.SPECIAL, lang.newreg(args[0], dtype=args[1], scalar=True), [], args[0])) #FIXME: Why don't we use the passed dtype?
      lang.ins.append(AssemblyInstruction(UOps.SPECIAL, lang.newreg(args[0], dtype=dtypes.uint64, scalar=True), [], args[0]))
    elif uop == UOps.DEFINE_LOCAL:
      lang.ins.append(AssemblyInstruction(UOps.DEFINE_LOCAL, None, [], args))
      # lang.ins.append(AssemblyInstruction(UOps.ALU, lang.newreg("buf-1", dtype=dtypes.uint64), [args[0]], UnaryOps.NOOP))
      lang.ins.append(AssemblyInstruction(UOps.ALU, lang.newreg(args[0], dtype=dtypes.uint64), [args[0]], UnaryOps.NOOP))
    elif uop == UOps.LOOP:
      if args[1] == "global":
        for i,var in enumerate(args[0]):
          global_size.append(var.max+1)
          lang.ins.append(AssemblyInstruction(UOps.SPECIAL, lang.newreg(var, dtype=dtypes.int32), [], f"gid{len(args[0])-1-i}"))
      elif args[1] == "local":
        for i,var in enumerate(args[0]):
          local_size.append(var.max+1)
          lang.ins.append(AssemblyInstruction(UOps.SPECIAL, lang.newreg(var, dtype=dtypes.int32), [], f"lid{len(args[0])-1-i}"))
      else:
        for var in args[0]:
          if not isinstance(var, NumNode):  # TODO: why is this coming through?
            # lang.ins.append(AssemblyInstruction(UOps.CONST, lang.newreg(var, dtype=dtypes.int32, scalar=True), [], 0))
            lang.ins.append(AssemblyInstruction(UOps.LOAD, lang.newreg(var, dtype=dtypes.int32, scalar=True), [], ConstOp(0, None))) #FIXME: what should valid be here?
            lang.ins.append(AssemblyInstruction(UOps.LABEL, None, [], "$loop_"+var.expr))
    elif uop == UOps.ENDLOOP:
      if args[1] not in ["global", "local", "global+local"]:
        for var in reversed(args[0]):
          if not isinstance(var, NumNode):  # TODO: why is this coming through?
            lang.ins.append(AssemblyInstruction(UOps.ALU, lang.tor[var], [lang.tor[var], 1], BinaryOps.ADD))
            pred = lang.render_alu(BinaryOps.CMPLT, lang.tor[var], var.max+1, dtypes.bool)
            lang.ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], ("$loop_"+var.expr, True)))
      elif args[1] == "local":
        lang.ins.append(AssemblyInstruction(UOps.ENDLOOP, None, [], None))
      elif args[1] == "global+local":
        lang.ins.append(AssemblyInstruction(UOps.ENDLOOP, None, [], None))
        #FIXME: doublecheck when we need sync
      # elif args[1] == "global":
      #   lang.ins.append(AssemblyInstruction(UOps.ENDLOOP, None, [], None))
    elif uop == UOps.CAST and newvar is not None:
      # TODO: we should reconsider outputting CAST in the linearizer. these are needless copies
      out = lang.newreg(newvar)
      for i,sr in enumerate(out.subregs()):
        lang.ins.append(AssemblyInstruction(UOps.ALU, sr, [lang.tor[vin[i]]], UnaryOps.NOOP))
    elif uop == UOps.ALU and newvar is not None:
      out = lang.newreg(newvar) if newvar not in lang.tor else lang.tor[newvar]
      # this is the only thing that can violate SSA
      if args in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
        pred_reg = lang.newreg((newvar, 'pred'), dtype=dtypes.bool)
        lang.ins.append(AssemblyInstruction(UOps.ALU, pred_reg, [lang.tor[x] for x in vin], args))
        lang.ins.append(AssemblyInstruction(UOps.CAST, out, [pred_reg], args))
      elif args == BinaryOps.DIV and lang.no_div:
        tmp = lang.newreg((newvar, "rcp"))
        lang.ins.append(AssemblyInstruction(UOps.ALU, tmp, [lang.tor[vin[1]]], UnaryOps.RECIP))
        lang.ins.append(AssemblyInstruction(UOps.ALU, out, [lang.tor[vin[0]], tmp], BinaryOps.MUL))
      elif args == UnaryOps.SIN and lang.sin_is_sin2pi:
        tmp = lang.newreg((newvar, "2pi"))
        lang.ins.append(AssemblyInstruction(UOps.ALU, tmp, [lang.tor[vin[0]], 1/(math.pi*2)], BinaryOps.MUL))
        lang.ins.append(AssemblyInstruction(UOps.ALU, out, [tmp], args))
      else:
        lang.ins.append(AssemblyInstruction(UOps.ALU, out, [lang.tor[x] for x in vin], args))
    elif uop == UOps.LOAD and newvar is not None:
      # TODO:
      if isinstance(args, ConstOp):
        lang.ins.append(AssemblyInstruction(UOps.LOAD, lang.newreg(newvar, dtype=newvar.dtype), [], args))
      else: # args is MemOp
        idx, treg, off = lang.addr_w_offset(args)
        reg = lang.newreg(newvar, dtype=newvar.dtype, scalar=(idx.scalar and (not isinstance(treg, Register) or treg.scalar))) # and not dtypes.is_float(newvar.dtype)))
        if args.valid.min == 0:
          #FIXME: We have MemOp w/ args.valid.min = 0 then do a ConstOp. Is this right? What is args.valid exactly
          # lang.ins.append(AssemblyInstruction(UOps.CONST, reg, [], 0))
          lang.ins.append(AssemblyInstruction(UOps.LOAD, reg, [], ConstOp(0, args.valid)))
          if args.valid.max == 1:
            pred = args.valid.render(lang.render_ops, lang)
            lang.ins.append(AssemblyInstruction(UOps.COND_BRANCH, None, [pred], (f"$skipload_{skipload_branch}", False)))
        if args.valid.max == 1:
          # NOTE: you can't compute the index in here, because it assumes it's all available later
          # lang.ins.append(AssemblyInstruction(UOps.LOAD, reg, [idx] + ([treg] if treg is not None else []), (off, 'global' if args.i != -1 else 'shared')))
          lang.ins.append(AssemblyInstruction(UOps.LOAD, reg, [idx] + ([treg] if treg is not None else []), (off, 'global' if not args.local else 'shared')))
        if args.valid.min == 0 and args.valid.max == 1:
          lang.ins.append(AssemblyInstruction(UOps.LABEL, None, [], f"$skipload_{skipload_branch}"))
          skipload_branch += 1
    elif uop == UOps.STORE:
      idx, treg, off = lang.addr_w_offset(args)
      # lang.ins.append(AssemblyInstruction(UOps.STORE, None, [idx, tor[vin[0]]] + ([treg] if treg is not None else []), (off, 'global' if args.i != -1 else 'shared')))
      lang.ins.append(AssemblyInstruction(UOps.STORE, None, [idx, lang.tor[vin[0]]] + ([treg] if treg is not None else []), (off, 'global' if not args.local else 'shared')))

  # define registers
  lang.ins = [AssemblyInstruction(UOps.DEFINE_REGISTER, None, [], (dtype, type_to_letter(dtype), c)) for dtype,c in lang.cnts.items()] + lang.ins

  if DEBUG >= 4:
    for tins in lang.ins: print(tins)
  return global_size, local_size
