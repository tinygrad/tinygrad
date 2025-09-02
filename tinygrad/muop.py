from __future__ import annotations
from dataclasses import dataclass, field
from tinygrad.dtype import DType, dtypes
from tinygrad.uop import Ops

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  size: int = field(hash=False, compare=False)

  def __str__(self): return self.name

@dataclass(frozen=True)
class Immediate:
  value: int
  size: int

  def __str__(self): return str(self.value)

@dataclass(frozen=True)
class Memory:
  size: int
  base: Register
  index: Register|None = None
  scale: int = 1
  disp: Immediate = Immediate(0, 4)

  def __str__(self):
    si = f" + {self.index}*{self.scale}" if self.index is not None else ""
    disp = f" + {self.disp}" if self.disp.value != 0 else ""
    return f"[{self.base}{si}{disp}]"

@dataclass(frozen=True)
class Label:
  name: str

  def __str__(self): return self.name

Operand = Register|Memory|Immediate|Label

@dataclass(frozen=True)
class MUOp:
  opstr: str
  opcode: int
  out: Operand|None = None
  ins: tuple[Operand, ...] = ()
  out_con: tuple[Register, ...] = ()
  ins_con: tuple[tuple[Register, ...], ...] = ()

  def __str__(self):
    return (self.opstr+" " if self.opstr else "") + ", ".join(([str(self.out)] if self.out is not None else []) + [str(i) for i in self.ins])
  def replace(self, out: Operand, ins: tuple[Operand, ...]): raise NotImplementedError("arch specific")
  def load(self, dest:Register, src:Memory) -> MUOp: raise NotImplementedError("arch specific")
  def store(self, dest:Memory, src:Register) -> MUOp: raise NotImplementedError("arch specific")
  def assign(self, dest:Register, src:Register) -> MUOp: raise NotImplementedError("arch specific")
  def encode(self) -> bytes: raise NotImplementedError("arch specific")

def assemble(src:list[MUOp]) -> bytes:
  # TODO: don't hardcore jump size (6)
  binary = bytearray()
  targets: dict[Label, int] = {}
  fixups: list[tuple[Label, int]] = []
  for mu in src:
    if isinstance(mu.out, Label):
      targets[mu.out] = len(binary)
      continue
    if mu.ins and isinstance(mu.ins[0], Label):
      if mu.ins[0] in targets:
        mu = mu.replace(mu.out, (Immediate(targets[mu.ins[0]] - (len(binary) + 6), 4),))
      else:
        fixups.append((mu.ins[0], len(binary) + 2))
        mu = mu.replace(mu.out, (Immediate(0, 4),))
    binary.extend(mu.encode())
  # patch offsets for forward jumps
  for label,loc in fixups:
    offset = targets[label] - (loc + 4)
    binary[loc:loc+4] = offset.to_bytes(4, "little", signed=True)
  return bytes(binary)

# *** X86 ***
GPR = tuple(Register(nm, i, 8) for i,nm in enumerate(["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + ["r"+str(i) for i in range(8,16)]))
VEC = tuple(Register("xmm"+str(i), i, 16) for i in range(16))
#https://wiki.osdev.org/X86-64_Instruction_Encoding
@dataclass(frozen=True)
class MUOpX86(MUOp):
  # fields known at MUOp init, rest depend on regalloc
  # MODR/M fields
  reg: Register|int = 0
  rm: Register|Memory|None = None
  # VEX fields
  pp: int = 0
  map_select: int = 0
  we: int = 0
  l: int = 0
  vvvv: Register|int = 0
  # REX fields
  prefix: int = 0
  w: int = 0
  # Immediate field
  imm: Immediate|Register|None = None
  # REX methods
  def RM(opstr, opcode, rm, w=0, prefix=0): return MUOpX86(opstr, opcode, rm, out_con=GPR, rm=rm, prefix=prefix, w=w)
  def _RM(opstr, opcode, reg, rm, w=0, prefix=0, in_cons=GPR): return MUOpX86(opstr, opcode, None, (rm,), (), (in_cons,), reg, rm, prefix=prefix, w=w) # noqa: E501
  def R_RM(opstr, opcode, reg, rm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (rm,), GPR, (GPR,), reg, rm, prefix=prefix, w=w)
  def _R_RM(opstr, opcode, reg, rm, w=0, prefix=0): return MUOpX86(opstr, opcode, None, (reg, rm), (), (GPR, GPR), reg, rm, prefix=prefix, w=w)
  def RM_R(opstr, opcode, rm, reg, w=0, prefix=0): return MUOpX86(opstr, opcode, rm, (reg,), GPR, (GPR,), reg, rm, prefix=prefix, w=w)
  def R_I(opstr, opcode, reg, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (imm,), GPR, ((),), reg, prefix=prefix, w=w, imm=imm)
  def RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, rm, (imm,), GPR, ((),), reg, rm, prefix=prefix, w=w, imm=imm)
  def _RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, None, (rm, imm), (), (GPR, ()), reg, rm, prefix=prefix, w=w, imm=imm) # noqa: E501
  def R_RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (rm, imm), GPR, (GPR, ()), reg, rm, prefix=prefix, w=w, imm=imm) # noqa: E501
  # VEX methods
  def V_M(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, ((),), reg, rm, pp, sel, w, l)
  def M_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), (), (VEC,), reg, rm, pp, sel, w, l)
  def V_VM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, (VEC,), reg, rm, pp, sel, w, l)
  def VM_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), VEC, (VEC,), reg, rm, pp, sel, w, l)
  def V_RM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, (GPR,), reg, rm, pp, sel, w, l)
  def RM_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), GPR, (VEC,), reg, rm, pp, sel, w, l)
  def R_VM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), GPR, (VEC,), reg, rm, pp, sel, w, l)
  def V_V_V(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, VEC), reg, rm, pp, sel, w, l, vvvv) # noqa: E501
  def V_V_VM(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, VEC), reg, rm, pp, sel, w, l, vvvv) # noqa: E501
  def V_V_RM(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, GPR), reg, rm, pp, sel, w, l, vvvv) # noqa: E501
  def V_VM_I(opstr, opcode, reg, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm, imm), VEC, (VEC, ()), reg, rm, pp, sel, w, l, imm=imm) # noqa: E501
  def VM_V_I(opstr, opcode, rm, reg, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg, imm), VEC, (VEC, ()), reg, rm, pp, sel, w, l, imm=imm) # noqa: E501
  def RM_V_I(opstr, opcode, rm, reg, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg, imm), GPR, (VEC, ()), reg, rm, pp, sel, w, l, imm=imm) # noqa: E501
  def V_V_VM_V(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, VEC, VEC), reg, rm, pp, sel, w, l, vvvv, imm=imm) # noqa: E501
  def V_V_RM_I(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, GPR, ()), reg, rm, pp, sel, w, l, vvvv, imm=imm) # noqa: E501
  def V_V_VM_I(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, VEC, ()), reg, rm, pp, sel, w, l, vvvv, imm=imm) # noqa: E501
  def replace(self, out: Operand, ins: tuple[Operand, ...]) -> MUOp:
    def _sub(x):
      for old,new in zip((self.out,)+self.ins, (out,)+ins):
        if x is old: return new
      return x
    return MUOpX86(self.opstr, self.opcode, out, ins, self.out_con, self.ins_con, _sub(self.reg), _sub(self.rm), self.pp, self.map_select, self.we,
                   self.l, _sub(self.vvvv), self.prefix, self.w, _sub(self.imm))
  def i_ops(op:Ops, dt:DType):
    if dt.itemsize == 1: return {Ops.ADD: ("add", 0x02), Ops.SUB: ("sub", 0x2A), Ops.AND: ("and", 0x22), Ops.OR: ("or", 0x0A),
                                 Ops.XOR: ("xor", 0x32), Ops.CMPNE: ("cmp", 0x3A), Ops.CMPLT: ("cmp", 0x3A), Ops.CMPEQ: ("cmp", 0x3A)}[op]
    return {Ops.ADD: ("add", 0x03), Ops.SUB: ("sub", 0x2B), Ops.MUL: ("imul", 0x0FAF), Ops.AND: ("and", 0x23), Ops.OR: ("or", 0x0B),
            Ops.XOR: ("xor", 0x33), Ops.CMPNE: ("cmp", 0x3B), Ops.CMPLT: ("cmp", 0x3B), Ops.CMPEQ: ("cmp", 0x3B)}[op]
  def imm_ops(op:Ops, dt:DType):
    if dt.itemsize == 1:
      if op is Ops.SHR: return ("shr", 0xC0, 5) if dtypes.is_unsigned(dt) else ("sar", 0xC0, 7)
      return {Ops.ADD: ("add", 0x80, 0), Ops.OR: ("or", 0x80, 1), Ops.AND: ("and", 0x80, 4), Ops.SUB: ("sub", 0x80, 5), Ops.XOR: ("xor", 0x80, 6),
              Ops.SHL: ("shl", 0xC0, 4), Ops.CMPNE: ("cmp", 0x80, 7), Ops.CMPLT: ("cmp", 0x80, 7), Ops.CMPEQ: ("cmp", 0x80, 7)}[op]
    if op is Ops.SHR: return ("shr", 0xC1, 5) if dtypes.is_unsigned(dt) else ("sar", 0xC1, 7)
    return {Ops.ADD: ("add", 0x81, 0), Ops.OR: ("or", 0x81, 1), Ops.AND: ("and", 0x81, 4), Ops.SUB: ("sub", 0x81, 5), Ops.XOR: ("xor", 0x81, 6),
            Ops.SHL: ("shl", 0xC1, 4), Ops.CMPNE: ("cmp", 0x81, 7), Ops.CMPLT: ("cmp", 0x81, 7), Ops.CMPEQ: ("cmp", 0x81, 7)}[op]
  def idiv(x:Register, a:Register, b:Register, is_signed:bool) -> list[MUOp]:
    rax = Register("rax", 0, 8)
    rdx = Register("rdx", 2, 8)
    in_cons = tuple(r for r in GPR if r not in (rax, rdx))
    move = MUOpX86("mov", 0x8B, x, (a,), (rax,), (GPR,), x, a, w=1)
    push = MUOpX86._RM("push", 0xFF, 6, rdx)
    if x.size == 1:
      extend = MUOpX86("cbw", 0x98, prefix=0x66) if is_signed else MUOpX86.R_RM("movzx", 0x0FB6, x, x)
      div = MUOpX86._RM("idiv", 0xF6, 7, b, in_cons=in_cons) if is_signed else MUOpX86._RM("div", 0xF6, 6, b, in_cons=in_cons)
    elif x.size == 2:
      extend = MUOpX86("cwd", 0x99, prefix=0x66) if is_signed else MUOpX86.R_RM("xor", 0x33, rdx, rdx, 1)
      div = MUOpX86._RM("idiv", 0xF7, 7, b, prefix=0x66, in_cons=in_cons) if is_signed else \
            MUOpX86._RM("div", 0xF7, 6, b, prefix=0x66, in_cons=in_cons)
    elif x.size == 4:
      extend = MUOpX86("cdq", 0x99) if is_signed else MUOpX86.R_RM("xor", 0x33, rdx, rdx, 1)
      div = MUOpX86._RM("idiv", 0xF7, 7, b, in_cons=in_cons) if is_signed else MUOpX86._RM("div", 0xF7, 6, b, in_cons=in_cons)
    elif x.size == 8:
      extend = MUOpX86("cqo", 0x99, w=1) if is_signed else MUOpX86.R_RM("xor", 0x33, rdx, rdx, 1)
      div = MUOpX86._RM("idiv", 0xF7, 7, b, 1, in_cons=in_cons) if is_signed else MUOpX86._RM("div", 0xF7, 6, b, 1, in_cons=in_cons)
    pop = MUOpX86._RM("pop", 0x8F, 0, rdx)
    return [move, push, extend, div, pop]
  def load(dest:Register, src:Memory) -> MUOp:
    if dest in GPR and dest.size == 1: return MUOpX86.R_RM("mov", 0x8A, dest, src)
    if dest in GPR and dest.size == 2: return MUOpX86.R_RM("mov", 0x8B, dest, src, 0, 0x66)
    if dest in GPR and dest.size == 4: return MUOpX86.R_RM("mov", 0x8B, dest, src)
    if dest in GPR and dest.size == 8: return MUOpX86.R_RM("mov", 0x8B, dest, src, 1)
    if dest in VEC and dest.size == 2: return MUOpX86.V_V_RM_I("vpinsrw", 0xC4, dest, dest, src, Immediate(0, 1), 1, 1)
    if dest in VEC and dest.size == 4: return MUOpX86.V_M("vmovss", 0x10, dest, src, 2, 1)
    if dest in VEC and dest.size == 8: return MUOpX86.V_M("vmovsd", 0x10, dest, src, 3, 1)
    if dest in VEC and dest.size == 16: return MUOpX86.V_VM("vmovups", 0x10, dest, src, 0, 1)
    raise RuntimeError("load missing")
  def store(dest:Memory, src:Register) -> MUOp:
    if src in GPR and src.size == 1: return MUOpX86.RM_R("mov", 0x88, dest, src)
    if src in GPR and src.size == 2: return MUOpX86.RM_R("mov", 0x89, dest, src, 0, 0x66)
    if src in GPR and src.size == 4: return MUOpX86.RM_R("mov", 0x89, dest, src)
    if src in GPR and src.size == 8: return MUOpX86.RM_R("mov", 0x89, dest, src, 1)
    if src in VEC and src.size == 2: return MUOpX86.RM_V_I("vpextrw", 0x15, dest, src, Immediate(0, 1), 1, 3)
    if src in VEC and src.size == 4: return MUOpX86.M_V("vmovss", 0x11, dest, src, 2, 1)
    if src in VEC and src.size == 8: return MUOpX86.M_V("vmovsd", 0x11, dest, src, 3, 1)
    if src in VEC and src.size == 16: return MUOpX86.VM_V("vmovups", 0x11, dest, src, 0, 1)
    raise RuntimeError("store missing")
  def assign(dest:Register, src:Register) -> MUOp:
    if dest in GPR and dest.size == 1: return MUOpX86.R_RM("mov", 0x8A, dest, src)
    if dest in GPR and dest.size == 2: return MUOpX86.R_RM("mov", 0x8B, dest, src, 0, 0x66)
    if dest in GPR and dest.size == 4: return MUOpX86.R_RM("mov", 0x8B, dest, src)
    if dest in GPR and dest.size == 8: return MUOpX86.R_RM("mov", 0x8B, dest, src, 1)
    if dest in VEC and dest.size <= 4: return MUOpX86.V_V_V("vmovss", 0x10, dest, src, src, 2, 1)
    if dest in VEC and dest.size == 8: return MUOpX86.V_V_V("vmovsd", 0x10, dest, src, src, 3, 1)
    if dest in VEC and dest.size == 16: return MUOpX86.VM_V("vmovups", 0x11, dest, src, 0, 1)
    raise RuntimeError("assign missing")
  # TODO: clean up all of this, more fields should be in class
  def encode(self) -> bytes:
    inst = bytearray()
    # *** EXCEPTIONS *** certain instructions have specific encodings
    if self.opstr == "": return b'' # fake MUOp
    if self.opcode == 0xB8: # 64bit imm load
      return ((0b0100 << 4) | (self.w << 3) | (0b00 << 2) | (int(self.out.index > 7) & 0b1)).to_bytes() + \
        int(self.opcode + (self.out.index % 8)).to_bytes() + self.imm.value.to_bytes(self.imm.size, 'little', signed=self.imm.value < 0)
    if self.opcode in (0x0F8C, 0x0F84): # jumps
      inst.extend(self.opcode.to_bytes(2))
      inst.extend(self.ins[0].value.to_bytes(self.ins[0].size, 'little', signed=True))
      return bytes(inst)
    # extends reg field
    r = int(isinstance(self.reg, Register) and self.reg.index > 7)
    # extends reg for index
    x = int(isinstance(self.rm, Memory) and self.rm.index is not None and self.rm.index.index > 7)
    # extends reg for base in sib or extends rm field
    b = int(isinstance(self.rm, Memory) and self.rm.base.index > 7 or isinstance(self.rm, Register) and self.rm.index > 7)
    if self.map_select:
      # *** VEX prefix ***
      vvvv = ~(self.vvvv.index if isinstance(self.vvvv, Register) else 0) & 0b1111
      # compact case
      if b == 0 and x == 0 and self.map_select == 1 and self.we == 0:
        inst.append(0xC5)
        inst.append(((~r & 0b1) << 7) | (vvvv << 3) | (self.l << 2) | self.pp)
      else: # general case
        inst.append(0xC4)
        inst.append(((~r & 0b1) << 7) | ((~x & 0b1) << 6) | ((~b & 0b1) << 5) | self.map_select)
        inst.append((self.we << 7) | (vvvv << 3) | (self.l << 2) | self.pp)
    else:
      # *** PREFIX byte ***
      if self.prefix: inst.append(self.prefix)
      # *** REX byte ***
      # if 64bit or extended register (index 8 - 15) is used or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
      if self.w or r or x or b or any(isinstance(v, Register) and v.size == 1 and v.name in ("rsp", "rbp", "rsi", "rdi") \
                                      for v in (self.reg, self.rm)):
        inst.append((0b0100 << 4) | (self.w << 3) | (r << 2) | (x << 1) | b)
    # *** OPCODE byte ***
    inst.extend(self.opcode.to_bytes((self.opcode.bit_length() + 7) // 8))
    # *** MODR/M byte ***
    if self.rm is not None:
      # reg field can be register or opcode extension
      # r/m field can be register, base register in memory or signal a sib byte is required
      reg, rm = self.reg, 0b000
      if isinstance(self.reg, Register): reg = self.reg.index & 0b111
      if isinstance(self.rm, Register): rm = self.rm.index & 0b111
      elif isinstance(self.rm, Memory): rm = self.rm.base.index & 0b111 if self.rm.index is None else 0b100
      # specifies operand types
      mod = 0b11
      # TODO: support 8 bit displacement
      #if isinstance(self.rm, Memory): mod = 0b00 if self.rm.disp.value == 0 else 0b01 if -128 <= self.rm.disp.value < 128 else 0b10
      if isinstance(self.rm, Memory):
        if self.rm.disp.value == 0: mod = 0b10 if self.rm.base.name == "r13" else 0b00
        else: mod = 0b10
      inst.append((mod << 6) | (reg << 3) | rm)
    # *** SIB byte ***
    if isinstance(self.rm, Memory) and (self.rm.index is not None or self.rm.base.name in ("rsp", "r12")):
      index = self.rm.index.index & 0b111 if self.rm.index is not None else 0b100
      scale = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[self.rm.scale]
      inst.append((scale << 6) | (index << 3) | (self.rm.base.index & 0b111))
    # *** DISPLACEMENT bytes ***
    if isinstance(self.rm, Memory) and (self.rm.disp.value or self.rm.base.name == "r13"):
      inst.extend(self.rm.disp.value.to_bytes(self.rm.disp.size, 'little', signed=True))
    # *** IMMEDIATE bytes *** the fourth register is in the upper 4 bits of an 8 bit immediate
    if isinstance(self.imm, Register): inst.append((self.imm.index & 0b1111) << 4 | 0b0000)
    elif self.imm is not None: inst.extend(self.imm.value.to_bytes(self.imm.size, 'little', signed=self.imm.value < 0))
    return bytes(inst)
