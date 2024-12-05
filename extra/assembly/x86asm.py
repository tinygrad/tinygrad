from typing import List, Dict, cast
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct

x86_mov_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.ASSIGN: "mov", Ops.DEFINE_ACC: "mov"}
x86_unsigned_ops = {**x86_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "div", Ops.MOD: "div", Ops.CMPNE: "cmp",
                    Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor"}
x86_signed_ops = {**x86_unsigned_ops, Ops.IDIV: "idiv", Ops.MOD: "idiv"}
x86_float32_ops = {Ops.ADD: "addss", Ops.SUB: "subss", Ops.MUL: "mulss", Ops.FDIV: "divss", Ops.CMPLT: "ucomiss", Ops.CMPNE: "ucomiss",
                 Ops.SQRT: "sqrtss", **{k:v+"ss" for k,v in x86_mov_ops.items()}}
x86_float64_ops = {**{k:v[:-1]+'d' for k,v in x86_float32_ops.items()}}
# NOTE: half dtype only supported in load/store, load can be zero extend followed by bitcast to float reg
#x86_float16_ops = {Ops.STORE: "", Ops.LOAD: "vmovdqu16"}
x86_float16_ops = {Ops.STORE: "movd", Ops.LOAD: "movd"}
# NOTE: are doubles vectorized? 2 doubles is "ups" not "lps", use a instead of u
x86_vec2_ops = {**{k:v+"lps" for k,v in x86_mov_ops.items()}}
x86_vec4_ops = {**{k:v+"ups" for k,v in x86_mov_ops.items()}}
#TODO: add float16 support?
x86op = {**{x:x86_unsigned_ops for x in (dtypes.bool,)+dtypes.uints}, **{x:x86_signed_ops for x in dtypes.sints},
          dtypes.float32:x86_float32_ops, dtypes.float64:x86_float64_ops, dtypes.float16:x86_float16_ops,
          dtypes.float32.vec(2):x86_vec2_ops, dtypes.float32.vec(4):x86_vec4_ops}

gep_imm = {0: "0x00", 1: "0x40", 2:"0x80", 3:"0xC0"}
vec_imm = {0: "0x00", 1: "0x10", 2:"0x20", 3:"0x30"}

x86_reg_map = {"rdi": {4: "edi", 2: "di", 1: "dil"}, "rsi": {4: "esi", 2: "si", 1: "sil"}, "rdx": {4: "edx", 2: "dx", 1: "dl"},
               "rcx": {4: "ecx", 2: "cx", 1: "cl"},  "rax": {4: "eax", 2: "ax", 1: "al"},  "rbx": {4: "ebx", 2: "bx", 1: "bl"},
               **{f"r{i}": {4: f"r{i}d", 2: f"r{i}w", 1: f"r{i}b"} for i in range(8,16)}}

size_prefix = {1: " byte ptr", 2: " word ptr", 4: " dword ptr", 8: " qword ptr"}

def to_hex(x, dt:DType) -> str:
  if not dtypes.is_float(dt): return hex(x)
  if dt is dtypes.float64: return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
  return hex(struct.unpack('<I', struct.pack('<f', x))[0])

def cflag(x:UOp) -> str:
  if x.op is Ops.CMPNE: return "setne"
  return "setl" if x.src[0].dtype in dtypes.sints else "setb"

def float_cast(x:DType, s:DType) -> str:
  if s is dtypes.float16: return "vcvtph2ps"
  if x is dtypes.float16: return "vcvtps2ph"
  cfrom = "si" if not dtypes.is_float(s) else "sd" if s.itemsize == 8 else "ss"
  cto = "si" if not dtypes.is_float(x) else "sd" if x.itemsize == 8 else "ss"
  if cto == "si": cfrom = "t" + cfrom
  return f"cvt{cfrom}2{cto}"

x86_rewrite = PatternMatcher([
  # loads/stores/movs
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"lea {ctx[x]}, [{ctx[x.src[0]]} + {ctx.r[x.src[1]]}*{x.src[0].dtype.itemsize}]"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[alt]}\ntest {ctx[mask]}, 1\n"
   f"jz .L{ctx.uops.index(x)}\n{x86op[x.dtype][x.op]} {ctx[x]}, [{ctx[idx]}]\n.L{ctx.uops.index(x)}:"),
  (UPat(Ops.LOAD, src=(UPat.var("idx"),), name="x"), lambda ctx,x,idx: f"{x86op[x.dtype][x.op]} {ctx[x]}, [{ctx[idx]}]"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x:
   f"{x86op[x.src[1].dtype][x.op]}{size_prefix[x.src[1].dtype.itemsize] if x.src[1].op is Ops.CONST else ''} [{ctx[x.src[0]]}], {ctx[x.src[1]]}"),
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}" if ctx[x] != ctx[x.src[1]] else None),
  # devectorize/vectorize
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"insertps {ctx[x]}, {ctx[x.src[0]]}, {gep_imm[x.arg[0]]}"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join(f"insertps {ctx[x]}, {ctx[s]}, {vec_imm[i]}" for i,s in enumerate(x.src))),
  # range
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"mov {ctx[x]}, {ctx[x.src[0]]}\n.LOOP_{x.arg}:"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x: f"inc {ctx[x.src[0]]}\ncmp {ctx[x.src[0]]}, {ctx[x.src[0].src[1]]}\njl .LOOP_{x.src[0].arg}"),
  # casting
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=(dtypes.bool,) + dtypes.uints),), name="x"), lambda ctx,x: f"movzx {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.sints),), name="x"),
   lambda ctx,x: f"movs{'x' if x.src[0].dtype.itemsize < 4 else 'xd'} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.float16, name="x"), lambda ctx,x: f"{float_cast(x.dtype, x.src[0].dtype)} {ctx[x]}, {ctx[x.src[0]]}, 0x4"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"{float_cast(x.dtype, x.src[0].dtype)} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"mov{'q' if x.dtype.itemsize == 8 else 'd'} {ctx[x]}, {ctx[x.src[0]]}"),
  # ternary ops (no cmov for floats)
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"),
   lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[1]]}\ntest {ctx[x.src[0]]}, 1\n"
   f"jnz .L{ctx.uops.index(x)}\n{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[2]]}\n.L{ctx.uops.index(x)}:"),
  (UPat(Ops.WHERE, name="x"),
   lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[1]]}\ntest {ctx[x.src[0]]}, 1\ncmovz {ctx[x]}, {ctx[x.src[2]]}"),
  # binary ops
  # float cmp requires nan check
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(UPat(dtype=dtypes.floats), UPat()), name="x"),
   lambda ctx,x: f"{x86op[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}\nsetp r15b\nxor {ctx[x]}, r15b"),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"), lambda ctx,x: f"{x86op[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}"),
  # requires rax/rdx
  #TODO: prealloc rax to idiv
  (UPat((Ops.IDIV, Ops.MOD), name="x"), lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}\n{ctx.idiv(x, x.src[1])}"),
  (UPat(GroupOp.Binary, name="x"),
   lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}\n{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}"),
  # unary ops
  (UPat(Ops.SQRT, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),
  # if
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"test {ctx[x.src[0]]}, 1\njz .L{ctx.uops.index(x)}"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f".L{ctx.uops.index(x.src[0])}:"),
])

x86_matcher = PatternMatcher([
  # value to store is float32 if it came from an alu
  (UPat(Ops.STORE, src=(UPat(dtype=dtypes.float16.ptr()), UPat(dtype=dtypes.float32)),
        name="x"), lambda x: x.src[0].store(x.src[1].cast(dtypes.float16))),
  # we use general register to store the 2 bytes of float16 (casting to int16 is a noop but means we use the correct register)
  (UPat(Ops.STORE, src=(UPat(dtype=dtypes.float16.ptr()), UPat(dtype=dtypes.float16)), name="x"),
   lambda x: x.src[0].store(x.src[1].bitcast(dtypes.int32).cast(dtypes.int16))),
  # float16 alus become float32
  (UPat(GroupOp.ALU, dtype=dtypes.float16, name="x"),
   lambda x: UOp(x.op, dtype=dtypes.float32, src=(s.cast(dtypes.float32) if s.dtype != dtypes.bool else s for s in x.src))),
  # TODO: casts from uint64 to float are complicated
  # TODO: remove extra casts by casting to max(c.dtype, float32)
  # can't cast from float16 to ints directly and vice versa
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.float16),), name="c"), lambda c: c.src[0].cast(dtypes.float32).cast(c.dtype)),
  (UPat(Ops.CAST, dtype=dtypes.float16, src=(UPat(dtype=dtypes.ints),), name="c"), lambda c: c.src[0].cast(dtypes.float32).cast(c.dtype)),
  # can't cast from float to int8/16 directly and vice versa
  (UPat(Ops.CAST, dtype=(dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16), src=(UPat(dtype=dtypes.floats),), name="c"),
    lambda c: c.src[0].cast(dtypes.int32).cast(c.dtype)),
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=(dtypes.bool, dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16)),), name="c"),
    lambda c: c.src[0].cast(dtypes.int32).cast(c.dtype)),
  # casting uint32 to float requires 64 bit register (float cast op assumes signed integers)
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uint32),), name="c"), lambda c: c.src[0].cast(dtypes.uint64).cast(c.dtype)),
  # 2 operand imul and cmov don't work with 8bit registers
  (UPat(Ops.MUL, dtype=(dtypes.uint8, dtypes.int8), name="x"),
    lambda x: UOp(Ops.MUL, dtype=dtypes.int16, src=(x.src[0].cast(dtypes.int16), x.src[1].cast(dtypes.int16))).cast(x.dtype)),
  (UPat(Ops.WHERE, dtype=(dtypes.bool, dtypes.uint8, dtypes.int8), name="x"),
    lambda x: UOp(Ops.WHERE, dtype=dtypes.int16, src=(x.src[0], x.src[1].cast(dtypes.int16), x.src[2].cast(dtypes.int16))).cast(x.dtype)),
  # *** also in ptx ***
  # cast between pointers is a noop
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  # *** also in llvmir ***
  # rewrite cast to bool to CMPNE 0
  (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
  # rewrite RECIP to FDIV
  (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
  # *** also in cstyle ***
  # gate any stores that aren't gated with ifs
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
    lambda store: UOp(Ops.STORE, src=store.src[:2]+(UOp(Ops.IF, src=(store.src[2],)),))),
  # rewrite MAX to CMPLT + WHERE
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

class X86Renderer(Renderer):
  device = "X86"
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  extra_matcher = x86_matcher

  def idiv(self, x:UOp, s:UOp) -> str:
    remainder_signex = {1:"cbw", 2: "cwd", 4: "cdq", 8: "cqo"}
    l = ""
    # if dividend is rax/rdx we don't push because pop would overwrite result
    if self.r[x] != "rax" and "rax" in self.r.values(): l += "push rax\n"
    if self.r[x] != "rdx" and "rdx" in self.r.values(): l += "push rdx\n"
    # divisor can't be rax or rdx
    if self.r[s] in ("rax", "rdx"): l += f"mov r15, {self.r[s]}\n"
    divisor = "r15" if self.r[s] in ("rax", "rdx") else self.r[s]
    l += f"mov {self.regt('rax', x.dtype)}, {self[x]}\n"
    if dtypes.is_unsigned(x.dtype): l += f"{'xor rdx, rdx' if x.dtype.itemsize > 1 else 'xor ah, ah'}\n"
    else: l += f"{remainder_signex[x.dtype.itemsize]}\n"
    l += f"{x86op[x.dtype][x.op]} {self.regt(divisor, s.dtype)}\n"
    l += f"mov {self[x]}, {self.regt('rax' if x.op is Ops.IDIV else 'rdx', x.dtype)}"
    if self.r[x] != "rdx" and "rdx" in self.r.values(): l += "\npop rdx"
    if self.r[x] != "rax" and "rax" in self.r.values(): l += "\npop rax"
    return l

  def regt(self, reg:str, dt:DType) -> str:
    if dt.itemsize == 8 or dtypes.is_float(dt) or isinstance(dt, PtrDType): return reg
    return x86_reg_map[reg][dt.itemsize]

  def __getitem__(self, key:UOp): return self.regt(self.r[key], key.dtype) if self.r[key] in self.all_regs else self.r[key]  # hacky helper
  def render(self, name:str, uops:List[UOp]) -> str:
    # 64 bit general registers, rsp/rbp not included, r15 temp register
    gen_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx", "r10", "r11", "r12", "r13", "r14"]
    float_regs = ["xmm" + str(i) for i in range(0,16)]
    self.all_regs = gen_regs + float_regs
    # can be a register, memory location or immediate value
    r: Dict[UOp, str] = {}
    self.r = r
    mem: Dict[UOp, str] = {}
    stack_size: int = 8
    kernel: List[str] = []
    self.uops = uops

    last_use: Dict[UOp, int] = {var: i for i,u in enumerate(uops) for var in (v for v in (u,) + u.src if v.dtype != dtypes.void)}

    def is_imm(u:UOp) -> bool: return u.op is Ops.CONST and not dtypes.is_float(u.dtype) and abs(u.arg) <= dtypes.max(dtypes.int32)
    def is_mem(u:UOp) -> bool: return u in r and u in mem and r[u] == mem[u]
    def is_reg(loc:str) -> bool: return loc in self.all_regs
    def free_reg(reg:str): float_regs.append(reg) if reg.startswith("xmm") else gen_regs.append(reg)

    def mov_to_reg(u:UOp, reg:str):
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or reg == "r15" else u.dtype
      kernel.append(f"{x86op[dt][Ops.LOAD]} {reg}, {r[u]}")
      r[u] = reg

    def mov_to_stack(u:UOp):
      nonlocal stack_size
      if u not in mem:
        mem[u] = f"[rbp - {stack_size}]"
        stack_size += 8
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or r[u] == "r15" else u.dtype
      kernel.append(f"{x86op[dt][Ops.STORE]} {mem[u]}, {r[u]}")
      r[u] = mem[u]

    def assign_reg(i:int, dt:DType) -> str:
      type_regs = float_regs if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else gen_regs
      if type_regs: return type_regs.pop(0)
      t = 'x' if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else 'r'
      # TODO: remove range check
      candidates = [u for u in r if r[u][0] == t and u not in (uops[i],) + uops[i].src and u.op is not Ops.RANGE]
      chosen = max(candidates, key=lambda u: last_use[u])
      reg = r[chosen]
      mov_to_stack(chosen)
      return reg

    for i,u in enumerate(uops):
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        if i < 6: r[u] = assign_reg(i, u.dtype)
        else: # value is in stack instead of register, rbp + 8 is return address
          # TODO: fix this
          r[u] = mem[u] = f"[rbp + {16 + (i-6)*8}]"
      elif u.op is Ops.CONST:
        r[u] = to_hex(u.arg, u.dtype)
        if not is_imm(u):
          mov_to_reg(u, "r15")
          mov_to_stack(u)
      # casting to <= int or src is uint32 (already zero extended) is a noop
      elif u.op is Ops.CAST and dtypes.is_int(u.dtype) and u.src[0].dtype in (dtypes.bool,) + dtypes.ints \
            and (u.dtype.itemsize <= u.src[0].dtype.itemsize or u.src[0].dtype is dtypes.uint32): r[u] = r[u.src[0]]
      else:
        for s in u.src: # mov srcs
          # these can't take imm values
          if is_imm(s) and not is_reg(r[s]) and u.op in (Ops.WHERE, Ops.IDIV, Ops.MOD): mov_to_reg(s, assign_reg(i, s.dtype))
          elif is_mem(s): mov_to_reg(s, assign_reg(i, s.dtype))
        if u.dtype != dtypes.void: # assign destination
          if u.op is Ops.ASSIGN:
            # define acc was already spilled here
            r[u] = mem[u] = mem[u.src[0]]
          else: r[u] = assign_reg(i, u.dtype)
        if u.op is Ops.RANGE: # all registers get moved to stack before loop TODO: remove range check
          for var in (v for v in r if is_reg(r[v]) and v.op is not Ops.RANGE):
            free_reg(r[var])
            mov_to_stack(var)
          # TODO?: if we do the cmp at the start of the loop we don't need this
          last_use[u.src[1]] = max(last_use[u], last_use[u.src[1]])
        # render x86 assembly
        if (l:=x86_rewrite.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))
      # free dead registers
      for loc in (r.pop(v) for v in (u,) + u.src if v in r and last_use[v] == i):
        if is_reg(loc) and loc not in r.values(): free_reg(loc)

    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp", f"sub rsp, {stack_size}"] + \
                      kernel + [f"add rsp, {stack_size}", "pop rbp", "ret", "\n"])
