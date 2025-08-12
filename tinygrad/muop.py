from __future__ import annotations
from dataclasses import dataclass, field

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
  disp: Immediate = Immediate(0, 1)

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
  def replace(self, out: Operand, ins: tuple[Operand, ...]): raise RuntimeError("arch specific")
  def load(self, dest:Register, src:Memory) -> MUOp: raise RuntimeError("arch specific")
  def store(self, dest:Memory, src:Register) -> MUOp: raise RuntimeError("arch specific")
  def assign(self, dest:Register, src:Register) -> MUOp: raise RuntimeError("arch specific")
  def encode(self) -> bytes: raise RuntimeError("arch specific")

def assemble(src:list[MUOp]) -> bytes:
  # TODO: don't hardcore jump size (6)
  bin, size = bytearray(), 0
  targets: dict[Label, int] = {}
  fixups: list[tuple[Label, int]] = []
  for mu in src:
    if isinstance(mu.out, Label):
      targets[mu.out] = size
      continue
    elif mu.ins and isinstance(mu.ins[0], Label):
      if mu.ins[0] in targets:
        mu = mu.replace(mu.out, (Immediate(targets[mu.ins[0]] - (size+6), 4),))
      else:
        fixups.append((mu.ins[0], size + 2))
        mu = mu.replace(mu.out, (Immediate(0, 4),))
    enc = mu.encode()
    size += len(enc)
    bin.extend(enc)
  # patch offsets for forward jumps
  for label,loc in fixups:
    offset = targets[label] - (loc + 4)
    bin[loc:loc+4] = offset.to_bytes(4, "little", signed=True)
  return bytes(bin)

# *** X86 ***
GPR = tuple(Register(name, i, 8) for i,name in enumerate(["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + ["r"+str(i) for i in range(8,16)]))
VEC = tuple(Register(name, i, 16) for i,name in enumerate(["xmm"+str(i) for i in range(0,16)]))
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
  def R_RM(opstr, opcode, reg, rm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (rm,), GPR, (GPR,), reg, rm, prefix=prefix, w=w)
  def _R_RM(opstr, opcode, reg, rm, w=0, prefix=0): return MUOpX86(opstr, opcode, None, (reg, rm), (), (GPR, GPR), reg, rm, prefix=prefix, w=w)
  def RM_R(opstr, opcode, rm, reg, w=0, prefix=0): return MUOpX86(opstr, opcode, rm, (reg,), GPR, (GPR,), reg, rm, prefix=prefix, w=w)
  def R_I(opstr, opcode, reg, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (imm,), GPR, ((),), reg, prefix=prefix, w=w, imm=imm)
  def RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, rm, (imm,), GPR, ((),), reg, rm, prefix=prefix, w=w, imm=imm)
  def _RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, None, (rm, imm), (), (GPR, ()), reg, rm, prefix=prefix, w=w, imm=imm)
  def R_RM_I(opstr, opcode, reg, rm, imm, w=0, prefix=0): return MUOpX86(opstr, opcode, reg, (rm, imm), GPR, (GPR, ()), reg, rm, prefix=prefix, w=w, imm=imm)
  # VEX methods
  def V_M(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, ((),), reg, rm, pp, sel, w, l)
  def M_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), (), (VEC,), reg, rm, pp, sel, w, l)
  def V_VM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, (VEC,), reg, rm, pp, sel, w, l)
  def VM_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), VEC, (VEC,), reg, rm, pp, sel, w, l)
  def V_RM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), VEC, (GPR,), reg, rm, pp, sel, w, l)
  def RM_V(opstr, opcode, rm, reg, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg,), GPR, (VEC,), reg, rm, pp, sel, w, l)
  def R_VM(opstr, opcode, reg, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (rm,), GPR, (VEC,), reg, rm, pp, sel, w, l)
  def V_V_V(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, VEC), reg, rm, pp, sel, w, l, vvvv)
  def V_V_VM(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, VEC), reg, rm, pp, sel, w, l, vvvv)
  def V_V_RM(opstr, opcode, reg, vvvv, rm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm), VEC, (VEC, GPR), reg, rm, pp, sel, w, l, vvvv)
  def VM_V_I(opstr, opcode, rm, reg, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg, imm), VEC, (VEC, ()), reg, rm, pp, sel, w, l, imm=imm)
  def RM_V_I(opstr, opcode, rm, reg, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, rm, (reg, imm), GPR, (VEC, ()), reg, rm, pp, sel, w, l, imm=imm)
  def V_V_VM_V(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, VEC, VEC), reg, rm, pp, sel, w, l, vvvv, imm=imm)
  def V_V_RM_I(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, GPR, ()), reg, rm, pp, sel, w, l, vvvv, imm=imm)
  def V_V_VM_I(opstr, opcode, reg, vvvv, rm, imm, pp, sel, w=0, l=0): return MUOpX86(opstr, opcode, reg, (vvvv, rm, imm), VEC, (VEC, VEC, ()), reg, rm, pp, sel, w, l, vvvv, imm=imm)
  def replace(self, out: Operand, ins: tuple[Operand, ...]) -> MUOp:
    def _sub(x):
      for old,new in zip((self.out,)+self.ins, (out,)+ins):
        if x is old: return new
      return x
    return MUOpX86(self.opstr, self.opcode, out, ins, self.out_con, self.ins_con, _sub(self.reg), _sub(self.rm), self.pp, self.map_select, self.we,
                   self.l, _sub(self.vvvv), self.prefix, self.w, _sub(self.imm))
  def load(dest:Register, src:Memory) -> MUOp:
    if dest in GPR and dest.size == 1: return MUOpX86.R_RM("mov", 0x8A, dest, src)
    if dest in GPR and dest.size == 2: return MUOpX86.R_RM("mov", 0x8B, dest, src, 0, 0x66)
    if dest in GPR and dest.size == 4: return MUOpX86.R_RM("mov", 0x8B, dest, src)
    if dest in GPR and dest.size == 8: return MUOpX86.R_RM("mov", 0x8B, dest, src, 1)
    if dest in VEC and dest.size == 4: return MUOpX86.V_M("vmovss", 0x10, dest, src, 2, 1)
    if dest in VEC and dest.size == 8: return MUOpX86.V_M("vmovsd", 0x10, dest, src, 3, 1)
    if dest in VEC and dest.size == 16: return MUOpX86.V_VM("vmovups", 0x10, dest, src, 0, 1)
    raise RuntimeError("load missing")
  def store(dest:Memory, src:Register) -> MUOp:
    if src in GPR and src.size == 1: return MUOpX86.RM_R("mov", 0x88, dest, src)
    if src in GPR and src.size == 2: return MUOpX86.RM_R("mov", 0x89, dest, src, 0, 0x66)
    if src in GPR and src.size == 4: return MUOpX86.RM_R("mov", 0x89, dest, src)
    if src in GPR and src.size == 8: return MUOpX86.RM_R("mov", 0x89, dest, src, 1)
    if src in VEC and src.size == 4: return MUOpX86.M_V("vmovss", 0x11, dest, src, 2, 1)
    if src in VEC and src.size == 8: return MUOpX86.M_V("vmovsd", 0x11, dest, src, 3, 1)
    if src in VEC and src.size == 16: return MUOpX86.VM_V("vmovups", 0x11, dest, src, 0, 1)
    raise RuntimeError("store missing")
  def assign(dest:Register, src:Register) -> MUOp:
    if dest in GPR and dest.size == 1: return MUOpX86.R_RM("mov", 0x8A, dest, src)
    if dest in GPR and dest.size == 2: return MUOpX86.R_RM("mov", 0x8B, dest, src, 0, 0x66)
    if dest in GPR and dest.size == 4: return MUOpX86.R_RM("mov", 0x8B, dest, src)
    if dest in GPR and dest.size == 8: return MUOpX86.R_RM("mov", 0x8B, dest, src, 1)
    if dest in VEC and dest.size == 4: return MUOpX86.V_V_V("vmovss", 0x10, dest, src, src, 2, 1)
    if dest in VEC and dest.size == 8: return MUOpX86.V_V_V("vmovsd", 0x10, dest, src, src, 3, 1)
    if dest in VEC and dest.size == 16: return MUOpX86.VM_V("vmovups", 0x11, dest, src, 0, 1)
    raise RuntimeError("assign missing")
  # TODO: clean up all of this, more fields should be in class
  def encode(self) -> bytes:
    inst = bytearray()
    # *** EXCEPTIONS *** certain instructions have specific encodings
    if self.opstr == "": return b'' # fake MUOp
    if self.opcode == 0xC3: return self.opcode.to_bytes() # ret
    if self.opcode in (0x50, 0x58) and not self.map_select: return int(self.opcode + self.ins[0].index).to_bytes((self.opcode.bit_length() + 7) // 8) # push/pop
    if self.opcode == 0xB8: # 64bit imm load
      return ((0b0100 << 4) | (self.w << 3) | ((int(self.out.index > 7) & 0b1) << 2) | 0b00).to_bytes() + \
        int(self.opcode + (self.out.index % 8)).to_bytes() + self.imm.value.to_bytes(self.imm.size, 'little')
    if self.opcode in (0x0F8C, 0x0F84): # jumps
      inst.extend(self.opcode.to_bytes(2))
      inst.extend(self.ins[0].value.to_bytes(self.ins[0].size, 'little', signed=True))
      return bytes(inst)
    # extends reg field
    r = int(isinstance(self.reg, Register) and self.reg.index > 7)
    # extends reg for index
    x = int(isinstance(self.rm, Memory) and self.rm.index is not None and self.rm.index.index > 7)
    # extends reg for base in sib or extends rmop field
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
      # *** PREFIX byte (optional) ***
      if self.prefix: inst.append(self.prefix)
      # *** REX byte (optional) ***
      # if 64bit or extended register (index 8 - 15) is used or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
      if self.w or r or x or b or any(isinstance(v, Register) and v.size == 1 and v.name in ("rsp", "rbp", "rsi", "rdi") for v in (self.reg, self.rm)):
        inst.append((0b0100 << 4) | (self.w << 3) | (r << 2) | (x << 1) | b)
    # *** OPCODE byte ***
    inst.extend(self.opcode.to_bytes((self.opcode.bit_length() + 7) // 8))
    # *** MODR/M byte ***
    # reg field can be register or opcode extension
    reg = self.reg.index & 0b111 if isinstance(self.reg, Register) else self.reg
    # r/m field can be register, base register in memory or signal a sib byte is required
    rm = 0b000
    if isinstance(self.rm, Register): rm = self.rm.index & 0b111
    elif isinstance(self.rm, Memory): rm = self.rm.base.index & 0b111 if self.rm.index is None else 0b100
    # specifies operand types
    mod = 0b11
    # TODO: support 8 bit displacement
    #if isinstance(rmop, Memory): mod = 0b00 if rmop.disp.value == 0 else 0b01 if -128 <= rmop.disp.value < 128 else 0b10
    if isinstance(self.rm, Memory):
      if self.rm.disp.value == 0: mod = 0b01 if self.rm.base.name == "r13" else 0b00
      else: mod = 0b10
    inst.append((mod << 6) | (reg << 3) | rm)
    # *** SIB byte (optional) ***
    if isinstance(self.rm, Memory) and (self.rm.index is not None or self.rm.base.name in ("rsp", "r12")):
      index = self.rm.index.index & 0b111 if self.rm.index is not None else 0b100
      scale = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[self.rm.scale]
      inst.append((scale << 6) | (index << 3) | (self.rm.base.index & 0b111))
    # *** DISPLACEMENT bytes (optional) ***
    if isinstance(self.rm, Memory) and (self.rm.disp.value or self.rm.base.name == "r13"):
      inst.extend(self.rm.disp.value.to_bytes(self.rm.disp.size, 'little', signed=True))
    # *** IMMEDIATE bytes (optional) *** the fourth register is in the upper 4 bits of an 8 bit immediate
    if isinstance(self.imm, Register): inst.append((self.imm.index & 0b1111) << 4 | 0b0000)
    # TODO: fix signage
    elif self.imm is not None: inst.extend(self.imm.value.to_bytes(self.imm.size, 'little', signed=self.imm.value < 0))
    return bytes(inst)
