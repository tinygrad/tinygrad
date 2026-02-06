import sys, struct, functools
from typing import cast
from tinygrad.dtype import dtypes, PtrDType, DType, truncate
from tinygrad.uop import Ops, X86Ops, GroupOp, X86GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext
from tinygrad.codegen.late.regalloc import Register, assign
from tinygrad.helpers import getenv, CPU_COUNT

# ***** X86 legalization *****

extra_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPEQ is XOR+XOR, bool CMPLT is XOR+AND
  # TODO: how does this work for vector dtypes?
  (UPat.var('x', dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtypes.bool).alu(Ops.CMPEQ, UPat.var('y')), lambda x,y: (x^y)^True),
  (UPat.var('x', dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # *** NOOP ***
  # cast to/from pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  (UPat.var("y").cast(name="x"), lambda y,x: x.replace(op=Ops.NOOP) if isinstance(y.dtype, PtrDType) else None),
  # zero extending scalar 32bit int is a noop
  (UPat.var("y", dtypes.uint32).cast(dtypes.int64s, name="x"), lambda y,x: x.replace(op=Ops.NOOP) if y.dtype.count == 1 else None),
  # cast between signed and unsigned int is a noop
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize else None),
  # bitcasts between scalar float and scalar int are real, rest are noops
  (UPat.var("y").bitcast().named("x"), lambda y,x: None if (y.dtype in dtypes.floats and x.dtype in dtypes.ints) or \
   (y.dtype in dtypes.ints and x.dtype in dtypes.floats) else x.replace(op=Ops.NOOP)),
  # TODO: this should be removed when bool cast is canonicalized
  # rewrite cast to bool to CMPNE 0
  (UPat.var("y").cast(dtypes.bool), lambda y: y != y.const_like(0)),
  # can't cast from float16 to ints/float64 directly and vice versa
  (UPat.var("y", dtypes.float16).cast((dtypes.float64,)+dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", (dtypes.float64,)+dtypes.ints).cast(dtypes.float16, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  # can't cast from float to int8/16 directly and vice versa
  (UPat.var("y", dtypes.floats).cast(dtypes.int8s+dtypes.int16s, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  (UPat.var("y", (dtypes.bool,)+dtypes.int8s+dtypes.int16s).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  # int/float casts only for signed int
  (UPat.var("y", dtypes.uint32).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int64).cast(x.dtype)),
  # casting uint64 to float requires special handling if msb is 1
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uint64),), name="c"),
   lambda c: ((c.src[0] >> 63) != 0).where((c.src[0] & 0x7FFFFFFFFFFFFFFF).cast(dtypes.int64).cast(c.dtype) * 2, \
                                               c.src[0].cast(dtypes.int64).cast(c.dtype))),
  # TODO: these should be removed once max is canonicalized
  # no max for scalar ints
  (UPat(Ops.MAX, dtypes.ints, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0]) if m.dtype.count == 1 else None),
  # even with Ops.MAX in decompositions this pattern still hits
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.floats), UPat.var("a")), lambda a,b: UOp(Ops.MAX, b.dtype, (a, b))),
  # no int8 mul or cmove, cast to int16
  (UPat.var("a", dtypes.int8s) * UPat.var("b"), lambda a,b: (a.cast(dtypes.int16) * b.cast(dtypes.int16)).cast(a.dtype)),
  (UPat.var("m").where(UPat.var("a", (dtypes.bool,)+dtypes.int8s), UPat.var("b")),
   lambda m,a,b: m.where(a.cast(dtypes.int16), b.cast(dtypes.int16)).cast(a.dtype) if a.dtype.count == 1 else None),
  # float16 alus are done in float32
  (UPat(GroupOp.ALU, dtypes.float16, name="x"), lambda x: UOp(x.op, dtypes.float.vec(x.dtype.count),
   tuple(s.cast(dtypes.float) if s.dtype != dtypes.bool else s for s in x.src)).cast(x.dtype)),
  (UPat(GroupOp.Comparison, src=(UPat.var("a", dtypes.float16), UPat.var("b")), name="x"),
   lambda x,a,b: UOp(x.op, x.dtype, (a.cast(dtypes.float32), b.cast(dtypes.float32))).cast(x.dtype)),
  # no cmpne for packed ints, y != x => !(y==x)
  (UPat(Ops.CMPNE, src=(UPat.var("y", dtypes.ints), UPat.var("x")), name="cmp"),
   lambda y,x,cmp: UOp(Ops.CMPEQ, cmp.dtype, (y,x))^True if y.dtype.count > 1 else None),
  # noop of a noop is removed
  (UPat(Ops.NOOP, src=(UPat(Ops.NOOP),), name="x"), lambda x: x.replace(src=x.src[0].src)),
  # cast to < scalar int is a noop
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize < y.dtype.itemsize and y.dtype.count == 1 else None),
  # float where expects a mask TODO: handle float64 cmp to float32 where
  (UPat.var("m", dtypes.bool).where(UPat.var("a", dtypes.floats), UPat.var("b")),
   lambda m,a,b: m.cast(a.dtype).ne(0).where(a, b) if m.src[0].dtype not in dtypes.floats else None),
  # TODO: do we want this? Kinda not needed if DEVECTORIZE=0. If yes make it general
  (UPat(Ops.VECTORIZE, dtypes.float16, name="x"), lambda x: x.replace(dtype=dtypes.float32.vec(x.dtype.count),
    src=tuple(s.src[0] for s in x.src)).cast(x.dtype) if all(s.op is Ops.CAST for s in x.src) else None),
  # moving elements of a single register to another without shuffling is a noop
  (UPat(Ops.VECTORIZE, src=(UPat.var("y"),), allow_any_len=True, name="x"),
   lambda y,x: UOp(Ops.NOOP, x.dtype, y.src) if all(s.op is Ops.GEP and s.src == y.src and s.arg[0] == i for i,s in enumerate(x.src)) else None),
])

# ***** X86 pre instruction selection *****

# these must be done in a separate matcher because they violate the spec
pre_isel_matcher = PatternMatcher([
  # gated index becomes a conditional move on the index, the load/store are unconditional
  (UPat.var("base").index(UPat.var("idx"), UPat.var("gate")).load(UPat.var("alt"), name="x"), lambda base,idx,gate,alt,x:
   gate.where(base.index(idx, ptr=True), (l:=UOp(Ops.DEFINE_LOCAL, base.dtype.base.ptr(x.dtype.count), arg=0)).after(l.store(alt))
              .index(UOp.const(dtypes.int32, 0), ptr=True)).load(dtype=x.dtype)),
  (UPat.var("base").index(UPat.var("idx"), UPat.var("gate")).store(UPat.var("val")), lambda base,idx,gate,val:
   gate.where(base.index(idx, ptr=True), UOp(Ops.DEFINE_LOCAL, base.dtype.base.ptr(val.dtype.count), arg=0)
              .index(UOp.const(dtypes.int32, 0), ptr=True)).store(val)),
  # TODO: remove this once we allow all flag producing ops in cmove
  # if gate in scalar int cmove is not a comparison need to add one to set the flag
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")),
   lambda m,a,b: m.ne(0).where(a,b) if m.op not in GroupOp.Comparison and a.dtype.count == 1 else None),
])

# ***** X86 registers *****

RAX = Register("rax", 0) # {4:"eax", 2:"ax", 1:"al"}
RCX = Register("rcx", 1) # {4:"ecx", 2:"cx", 1:"cl"}
RDX = Register("rdx", 2) # {4:"edx", 2:"dx", 1:"dl"}
RBX = Register("rbx", 3) # {4:"ebx", 2:"bx", 1:"bl"}
RSP = Register("rsp", 4) # {4:"esp", 2:"sp", 1:"spl"}
RBP = Register("rbp", 5) # {4:"ebp", 2:"bp", 1:"bpl"}
RSI = Register("rsi", 6) # {4:"esi", 2:"si", 1:"sil"}
RDI = Register("rdi", 7) # {4:"edi", 2:"di", 1:"dil"}
GPR = (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI) + tuple(Register(f"r{i}", i) for i in range(8, 16))
XMM = tuple(Register(f"ymm{i}", i) for i in range(16))
# gprs you can write to
WGPR = tuple(r for r in GPR if r != RSP)

# ***** X86 instruction selection *****

def def_reg(dt:DType, reg:Register|None=None) -> UOp: return UOp(X86Ops.DEFINE_REG, dt, arg=reg)
def imm(dt:DType, v:int|float) -> UOp: return UOp(X86Ops.IMM, dt, arg=v)
def to_imm(c:UOp) -> UOp|None:
  if c.op is not Ops.CONST: return None
  if c.dtype is dtypes.int64: return imm(dtypes.int32, c.arg) if not c.overflows(dtypes.int32) else None
  if c.dtype is dtypes.uint64: return imm(dtypes.uint32, c.arg) if not c.overflows(dtypes.uint32) else None
  if c.dtype in dtypes.ints+(dtypes.bool,): return imm(c.dtype, c.arg)
  return None
def cmp(x:UOp) -> UOp:
  if x.src[0].dtype is dtypes.float32: return UOp(X86Ops.VUCOMISS, src=x.src)
  if x.src[0].dtype is dtypes.float64: return UOp(X86Ops.VUCOMISD, src=x.src)
  return UOp(X86Ops.CMP, src=x.src) if (i:=to_imm(x.src[1])) is None else UOp(X86Ops.CMPi, src=(x.src[0], i))

# vshufps xmm2, xmm0, xmm1, imm
# xmm2 selects its lower 2 32 bits from xmm0 and its upper 2 32 bits from xmm1 according to imm
def vshufps(x:UOp) -> UOp:
  def _in(i:int) -> UOp: return s.src[0] if (s:=x.src[i]).op is Ops.GEP else s
  if len(x.src) != 4 or _in(0) is not _in(1) or _in(2) is not _in(3): return None
  return UOp(X86Ops.VSHUFPS, x.dtype, (_in(0), _in(2),
    imm(dtypes.uint8, sum(s.arg[0] << 2*i if s.op is Ops.GEP else 0 for i,s in enumerate(x.src)))))

# vinsertps xmm2, xmm0, xmm1, imm
# inserts any 32 bit element in xmm1 into any position in xmm0 according to immm, result is written to xmm2
# this is the fallback slow case for when you can't match more a powerful shuffle
def vinsertps(x:UOp) -> UOp:
  def _insert(ret:UOp, i:int) -> UOp:
    s, v = x.src[i], 0
    if s.op is Ops.GEP: s, v = s.src[0], s.arg[0]
    # moving the 0th element into the 0th position does nothing
    return s if i == v == 0 else UOp(X86Ops.VINSERTPS, x.dtype, (ret, s, imm(dtypes.uint8, v << 6 | i << 4)))
  return functools.reduce(_insert, range(len(x.src)), def_reg(x.dtype))

# vpinsq xmm2, xmm0, rax, imm
# inserts element in rax into any position in xmm0, result is written to xmm2 according to imm
def vpins(x:UOp) -> UOp:
  op = {1: X86Ops.VPINSRB, 2: X86Ops.VPINSRW, 4: X86Ops.VPINSRD, 8: X86Ops.VPINSRQ}[x.dtype.scalar().itemsize]
  return functools.reduce(lambda ret,i: UOp(op, x.dtype, (ret, x.src[i], imm(dtypes.uint8, i))), range(len(x.src)), def_reg(x.dtype))

def div(ctx:IselContext, x:UOp):
  # zero extend or move src[0] to x
  move1 = UOp(X86Ops.MOV, x.dtype, (x.src[0],), ctx.vreg(RAX))
  zero = UOp(X86Ops.MOVi, x.dtype, (imm(min(dtypes.uint32, x.dtype), 0),), ctx.vreg(RDX))
  move2 = UOp(X86Ops.MOV, x.dtype, (x.src[1],), ctx.vreg(tuple(r for r in WGPR if r not in (RAX, RDX))))
  div = UOp(X86Ops.DIV, x.dtype, (move2, zero, move1), ctx.vreg(RAX))
  return UOp(X86Ops.MOV, x.dtype, (div,))

def idiv(ctx:IselContext, x:UOp):
  cdq_op = {1: X86Ops.CBW, 2: X86Ops.CWD, 4: X86Ops.CDQ, 8: X86Ops.CQO}[x.dtype.itemsize]
  cdq = UOp(cdq_op, x.dtype, (UOp(X86Ops.MOV, x.dtype, (x.src[0],), ctx.vreg(RAX)),), ctx.vreg(RDX))
  move = UOp(X86Ops.MOV, x.dtype, (x.src[1],), ctx.vreg(tuple(r for r in WGPR if r not in (RAX, RDX))))
  idiv = UOp(X86Ops.IDIV, x.dtype, (move, cdq), ctx.vreg(RAX))
  # this move "cleanses" the register constraint (rax) of idiv, this is because the constraint only applies on definition and not on the uses of idiv
  return UOp(X86Ops.MOV, x.dtype, (idiv,))

def fuse_address(x:UOp) -> tuple[UOp, ...]:
  def _disp(v:int) -> UOp: return imm(dtypes.int32 if abs(v) > dtypes.max(dtypes.int8) else dtypes.int8, v)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), _disp(0))
  base, idx = x.src
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (base, _cast(idx.src[0]), _disp(idx.src[1].arg * disp_scale))
  if idx.op is Ops.CONST: return (base, UOp(Ops.NOOP), _disp(idx.arg * disp_scale))
  return (base, _cast(idx), _disp(0))

def fuse_load(ctx:IselContext, x:UOp, i:int) -> UOp|None:
  # if the load is used multiple times we don't fuse
  return x.replace(src=x.src[:i] + fuse_address(x.src[i].src[0]) + x.src[i+1:]) if len(ctx.uses[x.src[i]]) == x.src.count(x.src[i]) == 1 else None

def abi(ctx:IselContext, x:UOp):
  i = ctx.func_args.index(x)
  def _stack_arg(disp:int):
    return UOp(X86Ops.MOV, x.dtype, (def_reg(dtypes.uint64, RSP), UOp(Ops.NOOP), UOp(X86Ops.FRAME_INDEX, dtypes.int32, arg=disp)))
  if sys.platform == "win32": return def_reg(x.dtype, (RCX, RDX, GPR[8], GPR[9])[i]) if i < 4 else _stack_arg((i-3)*8+32)
  return def_reg(x.dtype, (RDI, RSI, RDX, RCX, GPR[8], GPR[9])[i]) if i < 6 else _stack_arg((i-5)*8)

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if dt.vec(l).itemsize == 2 and dt.vec(l) not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if dt.vec(l).itemsize == 4 and dt.vec(l) not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if dt.vec(l).itemsize == 8 and dt.vec(l) not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if dt.vec(l).itemsize == 16)

isel_matcher = PatternMatcher([
  # **** Op -> Op ****
  # rewrite -x -> 0 - x, this is done here because NEG is useful for MIN
  (UPat(Ops.NEG, name="x"), lambda x: UOp(Ops.SUB, x.dtype, (x.const_like(0),) + x.src)),
  # TODO: RANGE and END is tricky. Both linearizer and regalloc need them so they stay as Ops and get rewritten post regalloc
  # control flow ops need a refactor in general
  (UPat(Ops.RANGE, src=(UPat.cvar("c"),), allow_any_len=True, name="x"), lambda c,x: x.replace(src=(imm(c.dtype, c.arg),) + x.src[1:])),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(arg=ctx.vreg(WGPR)) if not isinstance(x.arg, Register) else None),
  # **** Op -> X86Op ****
  # add callee saved registers to the RET, these will be scheduled at the top of the kernel and will be saved/restored if they are used in regalloc
  # so regalloc builds the prologue/epilogue naturally
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(op=X86Ops.RET, src=x.src + tuple(def_reg(dtypes.uint64, r) for r in [RSP, RBP]))),
  # function abi constraints
  (UPat((Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), abi),
  # these are treated the same for now
  (UPat((Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), lambda ctx,x:
   x.replace(op=X86Ops.LEA, src=(def_reg(dtypes.uint64, RSP), UOp(Ops.NOOP), imm(dtypes.int32, ctx.inc_stack(x.dtype.nbytes()))), arg=None)),
  # constants that can't be immediates, move them to registers
  (UPat(Ops.CONST, dtypes.float16, name="x"), lambda x: UOp(X86Ops.VPINSRW, x.dtype, (def_reg(x.dtype), UOp(X86Ops.MOVi, dtypes.int16, (imm(x.dtype, x.arg),)), imm(dtypes.uint8, 0)))), # noqa: E501
  (UPat(Ops.CONST, dtypes.float32, name="x"), lambda x: UOp(X86Ops.VMOVD, x.dtype, (UOp(X86Ops.MOVi, dtypes.int32, (imm(x.dtype, x.arg),)),))),
  (UPat(Ops.CONST, dtypes.float64, name="x"), lambda x: UOp(X86Ops.VMOVQ, x.dtype, (UOp(X86Ops.MOVABS, dtypes.int64, (imm(x.dtype, x.arg),)),))),
  (UPat(Ops.CONST, dtypes.int64s, name="x"), lambda x: UOp(X86Ops.MOVABS, x.dtype, (imm(x.dtype, x.arg),))),
  (UPat(Ops.CONST, dtypes.ints+(dtypes.bool,), name="x"), lambda x: UOp(X86Ops.MOVi, x.dtype, (imm(x.dtype, x.arg),))),
  # conditional moves that use masks NOTE: these currently assume a mask producing cmp exists
  (UPat.var("m").where(UPat.var("a", dtypes.ints), UPat.var("b")), lambda m,a,b: UOp(X86Ops.VPBLENDVB, a.dtype, (b, a, m.replace(dtype=m.src[0].dtype))) if a.dtype.count > 1 else None), # noqa: E501
  (UPat.var("m").where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda m,a,b: UOp(X86Ops.VBLENDVPS, a.dtype, (b, a, m.replace(dtype=m.src[0].dtype)))), # noqa: E501
  (UPat.var("m").where(UPat.var("a", dtypes.float64), UPat.var("b")), lambda m,a,b: UOp(X86Ops.VBLENDVPD, a.dtype, (b, a, m.replace(dtype=m.src[0].dtype)))), # noqa: E501
  # in this case we have a mask producing comparison whose user expects a bool, so we convert to bool
  (UPat(GroupOp.Comparison, dtypes.bool, (UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.replace(dtype=x.src[0].dtype).bitcast(dtypes.int32).bitwise_and(1).f(Ops.NOOP, dtype=dtypes.bool)), # noqa: E501
  (UPat(GroupOp.Comparison, dtypes.bool, (UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.replace(dtype=x.src[0].dtype).bitcast(dtypes.int64).bitwise_and(1).f(Ops.NOOP, dtype=dtypes.bool)), # noqa: E501
  # conditional moves that use flags
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.sints), UPat()), name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: UOp(X86Ops.CMOVL, a.dtype, src=(b, a, cmp(m)))), # noqa: E501
  (UPat(Ops.CMPLT, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: UOp(X86Ops.CMOVB, a.dtype, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPEQ, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: UOp(X86Ops.CMOVE, a.dtype, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPNE, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: UOp(X86Ops.CMOVNE, a.dtype, src=(b, a, cmp(m)))),
  # jumps, use flags
  (UPat(Ops.IF, src=(UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.uints), UPat()), name="y"),), name="x"), lambda y,x: UOp(X86Ops.JB, x.dtype, (cmp(y),))), # noqa: E501
  (UPat(Ops.IF, src=(UPat(Ops.CMPLT, name="y"),)), lambda y: UOp(X86Ops.JL, src=(cmp(y),))),
  (UPat(Ops.IF, src=(UPat(Ops.CMPEQ, name="y"),)), lambda y: UOp(X86Ops.JE, src=(cmp(y),))),
  (UPat(Ops.IF, src=(UPat(Ops.CMPNE, name="y"),)), lambda y: UOp(X86Ops.JNE, src=(cmp(y),))),
  # comparisons whose user doesn't use the flag, move flag result to register
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.uints), UPat()), name="x"), lambda x: UOp(X86Ops.SETB, x.dtype, (cmp(x),))),
  (UPat(Ops.CMPLT, dtypes.bool, name="x"), lambda x: UOp(X86Ops.SETL, x.dtype, (cmp(x),))),
  (UPat(Ops.CMPEQ, dtypes.bool, name="x"), lambda x: UOp(X86Ops.SETE, x.dtype, (cmp(x),))),
  (UPat(Ops.CMPNE, dtypes.bool, name="x"), lambda x: UOp(X86Ops.SETNE, x.dtype, (cmp(x),))),
  # comparisons that produce masks (these aren't bool dtype)
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSS if x.dtype.count == 1 else X86Ops.VCMPPS, src=x.src + (imm(dtypes.uint8, 1),))), # noqa: E501
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSD if x.dtype.count == 1 else X86Ops.VCMPPD, src=x.src + (imm(dtypes.uint8, 1),))), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSS if x.dtype.count == 1 else X86Ops.VCMPPS, src=x.src + (imm(dtypes.uint8, 4),))), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSD if x.dtype.count == 1 else X86Ops.VCMPPD, src=x.src + (imm(dtypes.uint8, 4),))), # noqa: E501
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSS if x.dtype.count == 1 else X86Ops.VCMPPS, src=x.src + (imm(dtypes.uint8, 0),))), # noqa: E501
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VCMPSD if x.dtype.count == 1 else X86Ops.VCMPPD, src=x.src + (imm(dtypes.uint8, 0),))), # noqa: E501
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int8s), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VPCMPEQB)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int16s), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VPCMPEQW)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int32s), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VPCMPEQD)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int64s), UPat()), name="x"), lambda x: x.replace(op=X86Ops.VPCMPEQQ)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int8s), UPat.var("b")), name="x"), lambda a,b,x: x.replace(op=X86Ops.VPCMPGTB, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int16s), UPat.var("b")), name="x"), lambda a,b,x: x.replace(op=X86Ops.VPCMPGTW, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int32s), UPat.var("b")), name="x"), lambda a,b,x: x.replace(op=X86Ops.VPCMPGTD, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int64s), UPat.var("b")), name="x"), lambda a,b,x: x.replace(op=X86Ops.VPCMPGTQ, src=(b, a))),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt().named("x"), lambda y,x: UOp(X86Ops.VSQRTSS, x.dtype, (y, y)) if x.dtype.count == 1 else x.replace(op=X86Ops.VSQRTPS)), # noqa: E501
  (UPat.var("y", dtypes.float64).sqrt().named("x"), lambda y,x: UOp(X86Ops.VSQRTSD, x.dtype, (y, y)) if x.dtype.count == 1 else x.replace(op=X86Ops.VSQRTPD)), # noqa: E501
  (UPat.var("y", dtypes.float32).trunc().named("x"), lambda y,x: UOp(X86Ops.VROUNDSS, x.dtype, (y, y, imm(dtypes.uint8, 3))) if x.dtype.count == 1 else None), # noqa: E501
  (UPat.var("y", dtypes.float64).trunc().named("x"), lambda y,x: UOp(X86Ops.VROUNDSD, x.dtype, (y, y, imm(dtypes.uint8, 3))) if x.dtype.count == 1 else None), # noqa: E501
  (UPat.var("y", dtypes.float32).trunc().named("x"), lambda y,x: UOp(X86Ops.VROUNDPS, x.dtype, (y, imm(dtypes.uint8, 3)))),
  (UPat.var("y", dtypes.float64).trunc().named("x"), lambda y,x: UOp(X86Ops.VROUNDPD, x.dtype, (y, imm(dtypes.uint8, 3)))),
  # broadcasts TODO: not quite right, what about load fusion? Also, bitcast should be x86op and reg is xmm?
  (UPat.var("y", dtypes.int8s+(dtypes.bool,)).broadcast(name="x"), lambda y,x: UOp(X86Ops.VPBROADCASTB, x.dtype, (y.bitcast(dtypes.float32),))),
  (UPat.var("y", dtypes.int16s).broadcast(name="x"), lambda y,x: UOp(X86Ops.VPBROADCASTW, x.dtype, (y.bitcast(dtypes.float32),))),
  (UPat.var("y", dtypes.int32s).broadcast(name="x"), lambda y,x: UOp(X86Ops.VPBROADCASTD, x.dtype, (y.bitcast(dtypes.float32),))),
  (UPat.var("y", dtypes.int64s).broadcast(name="x"), lambda y,x: UOp(X86Ops.VPBROADCASTQ, x.dtype, (y.bitcast(dtypes.float64),))),
  (UPat.var("y", dtypes.float32).broadcast(name="x"), lambda y,x: UOp(X86Ops.VBROADCASTSS, x.dtype, (y,))),
  # shufles
  (UPat.var("y", dtypes.int16s).bitcast(dtypes.float16).named("x"), lambda y,x: UOp(X86Ops.VPINSRW, x.dtype, (def_reg(x.dtype), y, imm(dtypes.uint8, 0)))), # noqa: E501
  (UPat(Ops.VECTORIZE, dtypes.ints+(dtypes.bool,), name="x"), vpins),
  (UPat(Ops.VECTORIZE, dtypes.float32, name="x"), vshufps),
  (UPat(Ops.VECTORIZE, dtypes.float32, name="x"), vinsertps),
  (UPat.var("y", dtypes.float32).gep(name="x"), lambda y,x: UOp(X86Ops.VINSERTPS, x.dtype, (y, y, imm(dtypes.uint8, x.arg[0] << 6)))),
  # extract
  (UPat.var("y", dtypes.float16).bitcast(dtypes.int16s).named("x"), lambda y,x: UOp(X86Ops.VPEXTRW, x.dtype, (y, imm(dtypes.uint8, 0)))),
  (UPat.var("y", dtypes.int8s).gep(name="x"), lambda y,x: UOp(X86Ops.VPEXTRB, x.dtype, (y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int16s).gep(name="x"), lambda y,x: UOp(X86Ops.VPEXTRW, x.dtype, (y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int32s).gep(name="x"), lambda y,x: UOp(X86Ops.VPEXTRD, x.dtype, (y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int64s).gep(name="x"), lambda y,x: UOp(X86Ops.VPEXTRQ, x.dtype, (y, imm(dtypes.uint8, x.arg[0])))),
  # fused multiply add TODO: don't fuse if mul used several times
  (UPat.var('a', dtypes.float32) * UPat.var('b') + UPat.var('c'), lambda a,b,c: a.alu(X86Ops.VFMADD213SS if a.dtype.count == 1 else X86Ops.VFMADD213PS, b, c)), # noqa: E501
  (UPat.var('a', dtypes.float64) * UPat.var('b') + UPat.var('c'), lambda a,b,c: a.alu(X86Ops.VFMADD213SD if a.dtype.count == 1 else X86Ops.VFMADD213PD, b, c)), # noqa: E501
  # packed bitwise
  ((UPat() & UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPAND) if x.dtype.count > 1 else None),
  ((UPat() | UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPOR) if x.dtype.count > 1 else None),
  ((UPat() ^ UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPXOR) if x.dtype.count > 1 else None),
  # packed int binary
  ((UPat(dtype=dtypes.int32s) << UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPSLLVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int64s) << UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPSLLVQ) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.uint32) >> UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPSRLVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.uint64) >> UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPSRLVQ) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int32) >> UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPSRAVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int8s) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPADDB) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int16s) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPADDW) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int32s) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPADDD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int64s) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VPADDQ) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int8s, name="x"), lambda x: x.replace(op=X86Ops.VPSUBB) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int16s, name="x"), lambda x: x.replace(op=X86Ops.VPSUBW) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPSUBD) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPSUBQ) if x.dtype.count > 1 else None),
  (UPat(Ops.MUL, dtypes.int16s, name="x"), lambda x: x.replace(op=X86Ops.VPMULLW) if x.dtype.count > 1 else None),
  (UPat(Ops.MUL, dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPMULLD) if x.dtype.count > 1 else None),
  # scalar int binary
  ((UPat(dtype=dtypes.uints) // UPat()).named("x"), div),
  ((UPat(dtype=dtypes.sints) // UPat()).named("x"), idiv),
  ((UPat.var("a", dtypes.ints) << UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.SHLi, src=(a, imm(dtypes.uint8, b.arg))) if b.op is Ops.CONST else x.replace(op=X86Ops.SHL)), # noqa: E501
  ((UPat.var("a", dtypes.uints) >> UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.SHRi, src=(a, imm(dtypes.uint8, b.arg))) if b.op is Ops.CONST else x.replace(op=X86Ops.SHR)), # noqa: E501
  ((UPat.var("a", dtypes.sints) >> UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.SARi, src=(a, imm(dtypes.uint8, b.arg))) if b.op is Ops.CONST else x.replace(op=X86Ops.SHR)), # noqa: E501
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.AND) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.ANDi, src=(a, i))), # noqa: E501
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.OR) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.ORi, src=(a, i))), # noqa: E501
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.XOR) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.XORi, src=(a, i))), # noqa: E501
  ((UPat.var("a", dtypes.ints) * UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.IMUL) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.IMULi, src=(a, i))), # noqa: E501
  ((UPat.var("a", dtypes.ints) + UPat.var("b")).named("x"), lambda a,b,x: x.replace(op=X86Ops.ADD) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.ADDi, src=(a, i))), # noqa: E501
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.var("b")), name="x"), lambda a,b,x: x.replace(op=X86Ops.SUB) if (i:=to_imm(b)) is None else x.replace(op=X86Ops.SUBi, src=(a, i))), # noqa: E501
  # float binary
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VADDSS if x.dtype.count == 1 else X86Ops.VADDPS)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: x.replace(op=X86Ops.VADDSD if x.dtype.count == 1 else X86Ops.VADDPD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.replace(op=X86Ops.VMULSS if x.dtype.count == 1 else X86Ops.VMULPS)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: x.replace(op=X86Ops.VMULSD if x.dtype.count == 1 else X86Ops.VMULPD)),
  (UPat(Ops.SUB, dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VSUBSS if x.dtype.count == 1 else X86Ops.VSUBPS)),
  (UPat(Ops.SUB, dtypes.float64, name="x"), lambda x: x.replace(op=X86Ops.VSUBSD if x.dtype.count == 1 else X86Ops.VSUBPD)),
  (UPat(Ops.FDIV, dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VDIVSS if x.dtype.count == 1 else X86Ops.VDIVPS)),
  (UPat(Ops.FDIV, dtypes.float64, name="x"), lambda x: x.replace(op=X86Ops.VDIVSD if x.dtype.count == 1 else X86Ops.VDIVPD)),
  # TODO: these should use a.maximum(b) / a.minimum(b)
  (UPat(Ops.MAX, dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VMAXSS if x.dtype.count == 1 else X86Ops.VMAXPS)),
  (UPat(Ops.MAX, dtypes.float64, name="x"), lambda x: x.replace(op=X86Ops.VMAXSD if x.dtype.count == 1 else X86Ops.VMAXPD)),
  ((UPat.var("a", dtypes.float32) < UPat.var("b")).where(UPat.var("a"), UPat.var("b")), lambda a,b: UOp(X86Ops.VMINSS if a.dtype.count == 1 else X86Ops.VMINPS, a.dtype, (a, b))), # noqa: E501
  ((UPat.var("a", dtypes.float64) < UPat.var("b")).where(UPat.var("a"), UPat.var("b")), lambda a,b: UOp(X86Ops.VMINSD if a.dtype.count == 1 else X86Ops.VMINPD, a.dtype, (a, b))), # noqa: E501
  # casts
  (UPat(dtype=dtypes.int32).cast(dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VCVTDQ2PS) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.float64, name="x"), lambda x: x.replace(op=X86Ops.VCVTDQ2PD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VCVTTPS2DQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VCVTTPD2DQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.float64, name="x"), lambda x: x.replace(op=X86Ops.VCVTPS2PD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float64).cast(dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VCVTPD2PS) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: x.replace(op=X86Ops.VCVTPS2PH, src=x.src + (imm(dtypes.uint8, 4),))),
  (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: x.replace(op=X86Ops.VCVTPH2PS)),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VCVTTSS2SI)),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VCVTTSD2SI)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda y,x: x.replace(op=X86Ops.VCVTSS2SD, src=(y, y))),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda y,x: x.replace(op=X86Ops.VCVTSD2SS, src=(y, y))),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float32, name="x"), lambda y,x: x.replace(op=X86Ops.VCVTSI2SS, src=(def_reg(x.dtype), y))), # noqa: E501
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float64, name="x"), lambda y,x: x.replace(op=X86Ops.VCVTSI2SD, src=(def_reg(x.dtype), y))), # noqa: E501
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int16s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXBW) if x.dtype.count > 1 else None),
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXBD) if x.dtype.count > 1 else None),
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXBQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.uint16).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXWD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.uint16).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXWQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.uint32).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVZXDQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int8).cast(dtypes.int16s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXBW) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int8).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXBD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int8).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXBQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int16).cast(dtypes.int32s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXWD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int16).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXWQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.VPMOVSXDQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.uints+(dtypes.bool,)).cast(dtypes.ints, name="x"), lambda x: x.replace(op=X86Ops.MOVZX)),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=X86Ops.MOVSXD)),
  (UPat(dtype=dtypes.sints).cast(dtypes.ints, name="x"), lambda x: x.replace(op=X86Ops.MOVSX)),
  # bitcasts
  (UPat(dtype=dtypes.int32s).bitcast(dtypes.float32).named("x"), lambda x: x.replace(op=X86Ops.VMOVD)),
  (UPat(dtype=dtypes.int64s).bitcast(dtypes.float64).named("x"), lambda x: x.replace(op=X86Ops.VMOVQ)),
  (UPat(dtype=dtypes.float32).bitcast(dtypes.int32s).named("x"), lambda x: x.replace(op=X86Ops.VMOVDm)),
  (UPat(dtype=dtypes.float64).bitcast(dtypes.int64s).named("x"), lambda x: x.replace(op=X86Ops.VMOVQm)),
  # index
  (UPat(Ops.INDEX, name="x"), lambda x: x.replace(op=X86Ops.LEA, src=fuse_address(x))),
  # TODO: fuse stores, very few cases -- store cmp becomes setcc, store gep int becomes vpextr, store bitcast to int becomes vmovd/q
  # assign, load, store
  # NOTE: assign here violates the spec, it only happens in register allocation when a reg to reg move needs to be inserted
  (UPat(Ops.ASSIGN, dt_128bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVUPS)),
  (UPat(Ops.ASSIGN, dt_64bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVSD)),
  (UPat(Ops.ASSIGN, dt_32bit+dt_16bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVSS)),
  (UPat(Ops.ASSIGN, dtypes.ints+(dtypes.bool,), name="x"), lambda x: x.replace(op=X86Ops.MOV)),
  (UPat(Ops.LOAD, dt_128bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVUPS, src=fuse_address(x.src[0]))),
  (UPat(Ops.LOAD, dt_64bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVSD, src=fuse_address(x.src[0]))),
  (UPat(Ops.LOAD, dt_32bit, name="x"), lambda x: x.replace(op=X86Ops.VMOVSS, src=fuse_address(x.src[0]))),
  (UPat(Ops.LOAD, dt_16bit, name="x"), lambda x: x.replace(op=X86Ops.VPINSRW, src=(def_reg(x.dtype, x.arg),) + fuse_address(x.src[0]) + (imm(dtypes.uint8, 0),))), # noqa: E501
  (UPat(Ops.LOAD, dtypes.ints+(dtypes.bool,), name="x"), lambda x: x.replace(op=X86Ops.MOV, src=fuse_address(x.src[0]))),
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dt_128bit)), name="x"), lambda x: x.replace(op=X86Ops.VMOVUPSm, src=fuse_address(x.src[0]) + (x.src[1],))), # noqa: E501
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dt_64bit)), name="x"), lambda x: x.replace(op=X86Ops.VMOVSDm, src=fuse_address(x.src[0]) + (x.src[1],))),
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dt_32bit)), name="x"), lambda x: x.replace(op=X86Ops.VMOVSSm, src=fuse_address(x.src[0]) + (x.src[1],))),
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dt_16bit)), name="x"), lambda x: x.replace(op=X86Ops.VPEXTRW, src=fuse_address(x.src[0]) + (x.src[1], imm(dtypes.uint8, 0)))), # noqa: E501
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.ints+(dtypes.bool,))), name="x"),
   lambda x: x.replace(op=X86Ops.MOVm, src=fuse_address(x.src[0]) + (x.src[1],)) if (i:=to_imm(x.src[1])) is None else x.replace(op=X86Ops.MOVi, src=fuse_address(x.src[0]) + (i,))), # noqa: E501
  # **** X86Op -> X86Op ****
  # fuse loads into X86Ops that allow it, if beneficial
  (UPat(X86GroupOp.ReadMem1st, src=(UPat(Ops.LOAD),), allow_any_len=True, name="x"), lambda ctx,x: fuse_load(ctx, x, 0)),
  (UPat(X86GroupOp.ReadMem2nd, src=(UPat(), UPat(Ops.LOAD)), allow_any_len=True, name="x"), lambda ctx,x: fuse_load(ctx, x, 1)),
  (UPat(X86GroupOp.ReadMem3rd, src=(UPat(), UPat(), UPat(Ops.LOAD)), name="x"), lambda ctx,x: fuse_load(ctx, x, 2)),
  # allocate virtual register to X86Op, ones with specific constraints have already been allocated
  (UPat(X86GroupOp.All, name="x"), lambda ctx,x:
   x.replace(arg=ctx.vreg(XMM if x.dtype in dtypes.floats or x.dtype.count > 1 else WGPR)) if x.arg is None and x.dtype != dtypes.void else None),
])

# ***** post register allocation *****

# TODO: rm after,group
# final rewrite to match the isa spec
post_regalloc_matcher = PatternMatcher([
  # alloc stack space
  (UPat(X86Ops.DEFINE_REG, dtypes.uint64, arg=RSP, name="x"), lambda ctx,x:
   (x, [x, UOp(X86Ops.SUBi, dtypes.uint64, (imm(dtypes.uint32, ctx.stack_size),), RSP)]) if ctx.stack_size > 0 else None),
  # dealloc stack space
  (UPat(X86Ops.RET, name="x"), lambda ctx,x:
   (x, [UOp(X86Ops.ADDi, dtypes.uint64, (imm(dtypes.uint32, ctx.stack_size),), RSP), x]) if ctx.stack_size > 0 else None),
  # rewrite FRAME_INDEX to IMM now that the stack size is known
  (UPat(X86Ops.FRAME_INDEX, name="x"), lambda ctx,x: (nx:=x.replace(op=X86Ops.IMM, arg=ctx.stack_size + x.arg), [nx])),
  # rewrite RANGE to MOV reg, 0. Terrible HACK to pass the CONST to the END
  (UPat(Ops.RANGE, name="x"), lambda x: (nx:=x.replace(op=X86Ops.MOVi, src=(imm(x.dtype, 0),), tag=x.src[0].arg), [nx])),
  # rewrite END to ADD 1 -> CMPLT -> JUMP
  (UPat(Ops.END, name="x"), lambda x:
   (jl:=x.replace(op=X86Ops.JL, src=(x.src[1], cmp:=UOp(X86Ops.CMPi if isinstance(x.src[1].tag, int) else X86Ops.CMP,
    src=(add:=UOp(X86Ops.ADDi, x.src[1].dtype, (imm(x.src[1].dtype, 1),), x.src[1].arg),
         imm(x.src[1].dtype, x.src[1].tag) if isinstance(x.src[1].tag, int) else def_reg(x.src[1].dtype, x.src[1].tag))))), [add, cmp, jl])),
  # TODO: need a generic way to model clobbers, idiv and flags should be handled the same way, maybe add clobber field to Register?
  # fixup div, zero rdx again because scheduling constraint isn't being respected
  (UPat(X86Ops.DIV, name="x"), lambda x:
   (nx:=x.replace(src=x.src[:1]), [UOp(X86Ops.MOVi, x.dtype, (imm(min(dtypes.uint32, x.dtype), 0),), RDX), nx])),
  # remove cdq from idiv
  (UPat(X86Ops.IDIV, name="x"), lambda x: (nx:=x.replace(src=x.src[:-1]), [nx])),
  # rewrite two address instructions to two address form, if reused src wasn't coalesced insert a move
  (UPat(X86GroupOp.TwoAddress1st, name="x"), lambda ctx,x:
   (nx:=x.replace(src=x.src[1:]), [assign(ctx, x.src[0], x.arg), nx] if x.arg != x.src[0].arg else [nx])),
])

# ***** X86 spec *****
# TODO: do we even want this?
isa_spec = PatternMatcher([
  # these are the only non X86Ops allowed
  (UPat((Ops.NOOP, Ops.GROUP, Ops.AFTER, Ops.BARRIER)), lambda: True),
  # vblends take a mask which is float or int dtype
  (UPat((X86Ops.VPBLENDVB, X86Ops.VBLENDVPS, X86Ops.VBLENDVPD), src=(UPat.var("a"), UPat.var("b"), UPat.var("m")), name="x"),
   lambda a,b,m,x: x.dtype == a.dtype == b.dtype and x.dtype.itemsize == m.dtype.itemsize),
  # cmoves take a flag producing instruction
  (UPat((X86Ops.CMOVB, X86Ops.CMOVL, X86Ops.CMOVE, X86Ops.CMOVNE), dtypes.bool, (UPat(), UPat(), UPat(X86GroupOp.WriteFlags))), lambda: True),
  (UPat(X86GroupOp.All), lambda: True),
])

# ***** X86 instruction encoding *****

def to_bytes(dt:DType, v:int|float):
  v = truncate[dt](v)
  if dt in dtypes.floats: return struct.pack({dtypes.float16: "<e", dtypes.float32: "<f", dtypes.float64: "<d"}[dt], v)
  return v.to_bytes(dt.itemsize, 'little', signed=dt in dtypes.sints)

def encode(x:UOp, opc:int, reg:int|None=None, pp:int=0, sel:int=0, we:int=0):
  # get the encoding structure of the uop
  reg_uop, vvvv_uop, rm_uop, idx_uop, disp_uop, imm_uop = None, None, None, None, None, None
  # when a uop writes to memory it takes the form of a store, dtype is void, no definition
  if x.op in X86GroupOp.WriteMem:
    if len(x.src) > 3: rm_uop, idx_uop, disp_uop = x.src[0], x.src[1], x.src[2]
    else: rm_uop = x
    if reg is None:
      reg_uop = x.src[3] if len(x.src) > 3 else x.src[0]
      imm_uop = x.src[4] if len(x.src) == 5 else x.src[1] if len(x.src) == 2 else None
    else: imm_uop = x.src[3] if len(x.src) > 3 and x.src[3].arg is not None else x.src[0] if x.src[0].arg is not None else None

  elif x.op in X86GroupOp.ReadMem1st or x.op in X86GroupOp.ReadMem2nd and x.op in X86GroupOp.TwoAddress1st:
    if len(x.src) > 2: idx_uop, disp_uop = x.src[1], x.src[2]
    if reg is None: reg_uop = x
    if x.src[-1].dtype != dtypes.void: imm_uop = x.src[3] if len(x.src) == 4 else x.src[1] if len(x.src) == 2 else None
    rm_uop = x.src[0]

  elif x.op in X86GroupOp.ReadMem2nd or x.op in X86GroupOp.ReadMem3rd and x.op in X86GroupOp.TwoAddress1st:
    if len(x.src) > 3: idx_uop, disp_uop = x.src[2], x.src[3]
    reg_uop = x if x.dtype != dtypes.void else x.src[0]
    vvvv_uop = x.src[0] if x.dtype != dtypes.void else None
    imm_uop = x.src[4] if len(x.src) == 5 else x.src[2] if len(x.src) == 3 else None
    rm_uop = x.src[1]

  assert rm_uop is not None
  assert reg_uop is None if reg is not None else reg_uop is not None
  if imm_uop is not None: assert imm_uop.op is X86Ops.IMM or x.op in {X86Ops.VPBLENDVB, X86Ops.VBLENDVPS, X86Ops.VBLENDVPD}, x.op
  # now get the encoding values of the different fields
  rm = cast(Register, rm_uop.arg).index
  reg = cast(Register, reg_uop.arg).index if reg_uop is not None else reg
  vvvv = cast(Register, vvvv_uop.arg).index if vvvv_uop is not None else 0
  # index == 4 (rsp) indicates no index is present
  idx = cast(Register, idx_uop.arg).index if idx_uop is not None and idx_uop.arg is not None else 4
  reg_sz = (reg_uop.dtype.itemsize if not isinstance(reg_uop.dtype, PtrDType) else 8) if reg_uop is not None else 0
  # TODO: another reason to get rid of ptrs, if we access memory the size should be in scale uop otherwise size is in rm
  rm_sz = 8 if isinstance(rm_uop.dtype, PtrDType) and disp_uop is None else rm_uop.dtype.itemsize

  # encode instruction
  inst = bytes([])
  # PREFIX byte
  # there's other uses for this like atomic operations but setting 16bit variant of legacy op is currently the only one
  if sel == 0 and (reg_sz == 2 if reg_sz != 0 else rm_sz == 2): inst += bytes([0x66])
  # VEX bytes
  assert 0 <= reg <= 15 and 0 <= idx <= 15 and 0 <= rm <= 15
  # r extends reg field, x extends index field, b extends rm or base field
  r, _x, b = reg >> 3, idx >> 3, rm >> 3
  if sel:
    assert reg_uop is not None
    l = (max(reg_sz, rm_sz) > 16) & 0b1
    if sel == 1 and _x == b == we == 0: inst += bytes([0xC5, (~r & 0b1) << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
    else: inst += bytes([0xC4, (~r & 0b1) << 7 | (~_x & 0b1) << 6 | (~b & 0b1) << 5 | sel, we << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
  # REX byte
  else:
    # bit signaling 64 bit variant of instruction
    w = reg_sz == 8 if reg_sz != 0 else rm_sz == 8
    # rex prefix is required when an extended reg is used (index 8 - 15) or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
    if w | r | _x | b | (reg_sz == 1 & reg >> 2) | (rm_sz == 1 & rm >> 2): inst += bytes([0b0100 << 4 | w << 3 | r << 2 | _x << 1 | b])
  # OPCODE byte
  # legacy 8bit opcodes are 1 less than 16-64bit versions, with these exceptions
  real_opc = opc-1 if (rm_sz == 1 or reg_sz == 1) and x.op not in {X86Ops.SETB, X86Ops.SETE, X86Ops.SETL, X86Ops.SETNE, X86Ops.LEA} else opc
  inst += real_opc.to_bytes((real_opc.bit_length() + 7) // 8, 'big')
  # MODRM byte
  # now we only care about the lower 3 bits
  idx, rm, reg = idx & 0b111, rm & 0b111, reg & 0b111
  # 0b00 -- signals memory access with no displacement
  # 0b01 -- signals memory access with 8bit displacement
  # 0b10 -- signals memory access with 32bit displacement
  # 0b11 -- signals no memory access
  if disp_uop is not None:
    assert disp_uop.dtype in (dtypes.int8, dtypes.int32), "displacement can only be 1 or 4 byte signed int"
    # rbp/r13 always require a displacement
    if disp_uop.arg != 0 or rm == 0b101: mod = 0b01 if disp_uop.dtype.itemsize == 1 else 0b10
    else: mod = 0b00
  else: mod = 0b11
  # x 0b0 and idx 0b100 means rsp which means no index exists
  # rm 0b100 (rsp/r12) signals a sib byte is required, rm then is encoded in the base field of SIB
  _rm = rm if idx == 0b100 and _x == 0b0 else 0b100
  inst += bytes([mod << 6 | reg << 3 | _rm])
  # SIB byte
  if _rm == 0b100 and mod != 0b11:
    scale = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[1 if idx == 0b100 and _x == 0b0 else rm_sz]
    inst += bytes([scale << 6 | idx << 3 | rm])
  # DISP byte
  if mod == 0b01 or mod == 0b10:
    assert disp_uop is not None
    inst += disp_uop.arg.to_bytes(disp_uop.dtype.itemsize, 'little', signed=True)
  # IMM byte
  if imm_uop is not None:
    if isinstance(imm_uop.arg, Register): inst += bytes([(imm_uop.arg.index & 0b1111) << 4 | 0b0000])
    else: inst += to_bytes(imm_uop.dtype, imm_uop.arg)
  return inst

# https://www.felixcloutier.com/x86/
# NOTE: LEGACY prefix == VEX prefix
# pp field: None == 0, 66 == 1, F3 == 2, F2 == 3
# map select: 0F == 1, 0F38 == 2, 0F3A == 3
encodings = PatternMatcher([
  # moves
  (UPat(X86Ops.MOVABS, name="x"), lambda x:
   bytes([0b0100 << 4 | 0b1 << 3 | 0b00 << 2 | x.arg.index >> 3, 0xB8 + (x.arg.index & 0b111)]) + to_bytes(x.src[0].dtype, x.src[0].arg)),
  (UPat(X86Ops.MOV, name="x"), lambda x: encode(x, 0x8B)), (UPat(X86Ops.MOVi, name="x"), lambda x: encode(x, 0xC7, reg=0)),
  (UPat(X86Ops.MOVm, name="x"), lambda x: encode(x, 0x89)), (UPat(X86Ops.LEA, name="x"), lambda x: encode(x, 0x8D)),
  (UPat(X86Ops.VMOVSS, name="x"), lambda x: encode(x, 0x10, pp=2, sel=1)), (UPat(X86Ops.VMOVSSm, name="x"), lambda x: encode(x, 0x11, pp=2, sel=1)),
  (UPat(X86Ops.VMOVSD, name="x"), lambda x: encode(x, 0x10, pp=3, sel=1)), (UPat(X86Ops.VMOVSDm, name="x"), lambda x: encode(x, 0x11, pp=3, sel=1)),
  (UPat(X86Ops.VMOVUPS, name="x"), lambda x: encode(x, 0x10, pp=0, sel=1)), (UPat(X86Ops.VMOVUPSm, name="x"), lambda x: encode(x, 0x11, pp=0, sel=1)), # noqa: E501
  (UPat(X86Ops.VMOVD, name="x"), lambda x: encode(x, 0x6E, pp=1, sel=1)), (UPat(X86Ops.VMOVQ, name="x"), lambda x: encode(x, 0x6E, pp=1, sel=1, we=1)), # noqa: E501
  (UPat(X86Ops.VMOVDm, name="x"), lambda x: encode(x, 0x7E, pp=1, sel=1)), (UPat(X86Ops.VMOVQm, name="x"), lambda x: encode(x, 0x7E, pp=1, sel=1, we=1)), # noqa: E501
  # casts
  (UPat(X86Ops.MOVZX, name="x"), lambda x: encode(x, 0x0FB7)),
  (UPat(X86Ops.MOVSX, name="x"), lambda x: encode(x, 0x0FBF)), (UPat(X86Ops.MOVSXD, name="x"), lambda x: encode(x, 0x63)),
  (UPat(X86Ops.VPMOVZXBW, name="x"), lambda x: encode(x, 0x30, pp=1, sel=2)), (UPat(X86Ops.VPMOVZXBD, name="x"), lambda x: encode(x, 0x31, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMOVZXBQ, name="x"), lambda x: encode(x, 0x32, pp=1, sel=2)), (UPat(X86Ops.VPMOVZXWD, name="x"), lambda x: encode(x, 0x33, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMOVZXWQ, name="x"), lambda x: encode(x, 0x34, pp=1, sel=2)), (UPat(X86Ops.VPMOVZXDQ, name="x"), lambda x: encode(x, 0x35, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMOVSXBW, name="x"), lambda x: encode(x, 0x20, pp=1, sel=2)), (UPat(X86Ops.VPMOVSXBD, name="x"), lambda x: encode(x, 0x21, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMOVSXBQ, name="x"), lambda x: encode(x, 0x22, pp=1, sel=2)), (UPat(X86Ops.VPMOVSXWD, name="x"), lambda x: encode(x, 0x23, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMOVSXWQ, name="x"), lambda x: encode(x, 0x24, pp=1, sel=2)), (UPat(X86Ops.VPMOVSXDQ, name="x"), lambda x: encode(x, 0x25, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VCVTSS2SD, name="x"), lambda x: encode(x, 0x5A, pp=2, sel=1)), (UPat(X86Ops.VCVTSD2SS, name="x"), lambda x: encode(x, 0x5A, pp=3, sel=1)), # noqa: E501
  (UPat(X86Ops.VCVTPH2PS, name="x"), lambda x: encode(x, 0x13, pp=1, sel=2)), (UPat(X86Ops.VCVTPS2PH, name="x"), lambda x: encode(x, 0x1D, pp=1, sel=3)), # noqa: E501
  (UPat(X86Ops.VCVTDQ2PS, name="x"), lambda x: encode(x, 0x5B, pp=0, sel=1)), (UPat(X86Ops.VCVTDQ2PD, name="x"), lambda x: encode(x, 0xE6, pp=2, sel=1)), # noqa: E501
  (UPat(X86Ops.VCVTPS2PD, name="x"), lambda x: encode(x, 0x5A, pp=0, sel=1)), (UPat(X86Ops.VCVTPD2PS, name="x"), lambda x: encode(x, 0x5A, pp=1, sel=1)), # noqa: E501
  (UPat(X86Ops.VCVTTPS2DQ, name="x"), lambda x: encode(x, 0x5B, pp=2, sel=1)), (UPat(X86Ops.VCVTTPD2DQ, name="x"), lambda x: encode(x, 0xE6, pp=1, sel=1)), # noqa: E501
  (UPat(X86Ops.VCVTSI2SS, name="x"), lambda x: encode(x, 0x2A, pp=2, sel=1, we=x.src[1].dtype.base is dtypes.int64)),
  (UPat(X86Ops.VCVTSI2SD, name="x"), lambda x: encode(x, 0x2A, pp=3, sel=1, we=x.src[1].dtype.base is dtypes.int64)),
  (UPat(X86Ops.VCVTTSS2SI, name="x"), lambda x: encode(x, 0x2C, pp=2, sel=1, we=x.dtype in dtypes.int64s)),
  (UPat(X86Ops.VCVTTSD2SI, name="x"), lambda x: encode(x, 0x2C, pp=3, sel=1, we=x.dtype in dtypes.int64s)),
  # int division
  (UPat(X86Ops.CBW), lambda: bytes([0x66, 0x98])), (UPat(X86Ops.CWD), lambda: bytes([0x66, 0x99])),
  (UPat(X86Ops.CDQ), lambda: bytes([0x99])), (UPat(X86Ops.CQO), lambda: bytes([0x48, 0x99])),
  (UPat(X86Ops.IDIV, name="x"), lambda x: encode(x, 0xF7, reg=7)), (UPat(X86Ops.DIV, name="x"), lambda x: encode(x, 0xF7, reg=6)),
  # scalar int binary
  (UPat(X86Ops.SHLi, name="x"), lambda x: encode(x, 0xC1, reg=4)),
  (UPat(X86Ops.SHRi, name="x"), lambda x: encode(x, 0xC1, reg=5)), (UPat(X86Ops.SARi, name="x"), lambda x: encode(x, 0xC1, reg=7)),
  (UPat(X86Ops.ADD, name="x"), lambda x: encode(x, 0x03)), (UPat(X86Ops.ADDi, name="x"), lambda x: encode(x, 0x81, reg=0)),
  (UPat(X86Ops.SUB, name="x"), lambda x: encode(x, 0x2B)), (UPat(X86Ops.SUBi, name="x"), lambda x: encode(x, 0x81, reg=5)),
  (UPat(X86Ops.AND, name="x"), lambda x: encode(x, 0x23)), (UPat(X86Ops.ANDi, name="x"), lambda x: encode(x, 0x81, reg=4)),
  (UPat(X86Ops.XOR, name="x"), lambda x: encode(x, 0x33)), (UPat(X86Ops.XORi, name="x"), lambda x: encode(x, 0x81, reg=6)),
  (UPat(X86Ops.OR, name="x"), lambda x: encode(x, 0x0B)), (UPat(X86Ops.ORi, name="x"), lambda x: encode(x, 0x81, reg=1)),
  (UPat(X86Ops.CMP, name="x"), lambda x: encode(x, 0x3B)), (UPat(X86Ops.CMPi, name="x"), lambda x: encode(x, 0x81, reg=7)),
  (UPat(X86Ops.IMUL, name="x"), lambda x: encode(x, 0x0FAF)), (UPat(X86Ops.IMULi, name="x"), lambda x: encode(x, 0x69)),
  (UPat(X86Ops.SETB, name="x"), lambda x: encode(x, 0x0F92, reg=0)), (UPat(X86Ops.SETL, name="x"), lambda x: encode(x, 0x0F9C, reg=0)),
  (UPat(X86Ops.SETE, name="x"), lambda x: encode(x, 0x0F94, reg=0)), (UPat(X86Ops.SETNE, name="x"), lambda x: encode(x, 0x0F95, reg=0)),
  # packed bitwise NOTE: only bitwise and packed
  (UPat(X86Ops.VPAND, name="x"), lambda x: encode(x, 0xDB, pp=1, sel=1)), (UPat(X86Ops.VPXOR, name="x"), lambda x: encode(x, 0xEF, pp=1, sel=1)),
  (UPat(X86Ops.VPOR, name="x"), lambda x: encode(x, 0xEB, pp=1, sel=1)),
  # unary
  (UPat(X86Ops.VSQRTSS, name="x"), lambda x: encode(x, 0x51, pp=2, sel=1)), (UPat(X86Ops.VSQRTPS, name="x"), lambda x: encode(x, 0x51, pp=0, sel=1)),
  (UPat(X86Ops.VSQRTSD, name="x"), lambda x: encode(x, 0x51, pp=3, sel=1)), (UPat(X86Ops.VSQRTPD, name="x"), lambda x: encode(x, 0x51, pp=1, sel=1)),
  (UPat(X86Ops.VROUNDSS, name="x"), lambda x: encode(x, 0x0A, pp=1, sel=3)), (UPat(X86Ops.VROUNDPS, name="x"), lambda x: encode(x, 0x08, pp=1, sel=3)), # noqa: E501
  (UPat(X86Ops.VROUNDSD, name="x"), lambda x: encode(x, 0x0B, pp=1, sel=3)), (UPat(X86Ops.VROUNDPD, name="x"), lambda x: encode(x, 0x09, pp=1, sel=3)), # noqa: E501
  # packed int binary
  (UPat(X86Ops.VPSLLVD, name="x"), lambda x: encode(x, 0x47, pp=1, sel=2)), (UPat(X86Ops.VPSLLVQ, name="x"), lambda x: encode(x, 0x47, pp=1, sel=2, we=1)), # noqa: E501
  (UPat(X86Ops.VPSRLVD, name="x"), lambda x: encode(x, 0x45, pp=1, sel=2)), (UPat(X86Ops.VPSRLVQ, name="x"), lambda x: encode(x, 0x45, pp=1, sel=2, we=1)), # noqa: E501
  (UPat(X86Ops.VPCMPGTB, name="x"), lambda x: encode(x, 0x64, pp=1, sel=1)), (UPat(X86Ops.VPCMPGTW, name="x"), lambda x: encode(x, 0x65, pp=1, sel=1)), # noqa: E501
  (UPat(X86Ops.VPCMPGTD, name="x"), lambda x: encode(x, 0x66, pp=1, sel=1)), (UPat(X86Ops.VPCMPGTQ, name="x"), lambda x: encode(x, 0x37, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPCMPEQB, name="x"), lambda x: encode(x, 0x74, pp=1, sel=1)), (UPat(X86Ops.VPCMPEQW, name="x"), lambda x: encode(x, 0x75, pp=1, sel=1)), # noqa: E501
  (UPat(X86Ops.VPCMPEQD, name="x"), lambda x: encode(x, 0x76, pp=1, sel=1)), (UPat(X86Ops.VPCMPEQQ, name="x"), lambda x: encode(x, 0x29, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPMULLW, name="x"), lambda x: encode(x, 0xD5, pp=1, sel=1)), (UPat(X86Ops.VPMULLD, name="x"), lambda x: encode(x, 0x40, pp=1, sel=2)),
  (UPat(X86Ops.VPADDB, name="x"), lambda x: encode(x, 0xFC, pp=1, sel=1)), (UPat(X86Ops.VPADDW, name="x"), lambda x: encode(x, 0xFD, pp=1, sel=1)),
  (UPat(X86Ops.VPADDD, name="x"), lambda x: encode(x, 0xFE, pp=1, sel=1)), (UPat(X86Ops.VPADDQ, name="x"), lambda x: encode(x, 0xD4, pp=1, sel=1)),
  (UPat(X86Ops.VPSUBB, name="x"), lambda x: encode(x, 0xF8, pp=1, sel=1)), (UPat(X86Ops.VPSUBW, name="x"), lambda x: encode(x, 0xF9, pp=1, sel=1)),
  (UPat(X86Ops.VPSUBD, name="x"), lambda x: encode(x, 0xFA, pp=1, sel=1)), (UPat(X86Ops.VPSUBQ, name="x"), lambda x: encode(x, 0xFB, pp=1, sel=1)),
  (UPat(X86Ops.VPSRAVD, name="x"), lambda x: encode(x, 0x46, pp=1, sel=2)),
  # float cmp
  (UPat(X86Ops.VUCOMISS, name="x"), lambda x: encode(x, 0x2E, pp=0, sel=1)), (UPat(X86Ops.VUCOMISD, name="x"), lambda x: encode(x, 0x2E, pp=1, sel=1)), # noqa: E501
  # scalar / packed float binary
  (UPat(X86Ops.VADDSS, name="x"), lambda x: encode(x, 0x58, pp=2, sel=1)), (UPat(X86Ops.VADDPS, name="x"), lambda x: encode(x, 0x58, pp=0, sel=1)),
  (UPat(X86Ops.VADDSD, name="x"), lambda x: encode(x, 0x58, pp=3, sel=1)), (UPat(X86Ops.VADDPD, name="x"), lambda x: encode(x, 0x58, pp=1, sel=1)),
  (UPat(X86Ops.VSUBSS, name="x"), lambda x: encode(x, 0x5C, pp=2, sel=1)), (UPat(X86Ops.VSUBPS, name="x"), lambda x: encode(x, 0x5C, pp=0, sel=1)),
  (UPat(X86Ops.VSUBSD, name="x"), lambda x: encode(x, 0x5C, pp=3, sel=1)), (UPat(X86Ops.VSUBPD, name="x"), lambda x: encode(x, 0x5C, pp=1, sel=1)),
  (UPat(X86Ops.VMULSS, name="x"), lambda x: encode(x, 0x59, pp=2, sel=1)), (UPat(X86Ops.VMULPS, name="x"), lambda x: encode(x, 0x59, pp=0, sel=1)),
  (UPat(X86Ops.VMULSD, name="x"), lambda x: encode(x, 0x59, pp=3, sel=1)), (UPat(X86Ops.VMULPD, name="x"), lambda x: encode(x, 0x59, pp=1, sel=1)),
  (UPat(X86Ops.VDIVSS, name="x"), lambda x: encode(x, 0x5E, pp=2, sel=1)), (UPat(X86Ops.VDIVPS, name="x"), lambda x: encode(x, 0x5E, pp=0, sel=1)),
  (UPat(X86Ops.VDIVSD, name="x"), lambda x: encode(x, 0x5E, pp=3, sel=1)), (UPat(X86Ops.VDIVPD, name="x"), lambda x: encode(x, 0x5E, pp=1, sel=1)),
  (UPat(X86Ops.VCMPSS, name="x"), lambda x: encode(x, 0xC2, pp=2, sel=1)), (UPat(X86Ops.VCMPPS, name="x"), lambda x: encode(x, 0xC2, pp=0, sel=1)),
  (UPat(X86Ops.VCMPSD, name="x"), lambda x: encode(x, 0xC2, pp=3, sel=1)), (UPat(X86Ops.VCMPPD, name="x"), lambda x: encode(x, 0xC2, pp=1, sel=1)),
  (UPat(X86Ops.VMAXSS, name="x"), lambda x: encode(x, 0x5F, pp=2, sel=1)), (UPat(X86Ops.VMAXPS, name="x"), lambda x: encode(x, 0x5F, pp=0, sel=1)),
  (UPat(X86Ops.VMAXSD, name="x"), lambda x: encode(x, 0x5F, pp=3, sel=1)), (UPat(X86Ops.VMAXPD, name="x"), lambda x: encode(x, 0x5F, pp=1, sel=1)),
  (UPat(X86Ops.VMINSS, name="x"), lambda x: encode(x, 0x5D, pp=2, sel=1)), (UPat(X86Ops.VMINPS, name="x"), lambda x: encode(x, 0x5D, pp=0, sel=1)),
  (UPat(X86Ops.VMINSD, name="x"), lambda x: encode(x, 0x5D, pp=3, sel=1)), (UPat(X86Ops.VMINPD, name="x"), lambda x: encode(x, 0x5D, pp=1, sel=1)),
  # ternary
  (UPat(X86Ops.CMOVB, name="x"), lambda x: encode(x, 0x0F42)), (UPat(X86Ops.CMOVL, name="x"), lambda x: encode(x, 0x0F4C)),
  (UPat(X86Ops.CMOVE, name="x"), lambda x: encode(x, 0x0F44)), (UPat(X86Ops.CMOVNE, name="x"), lambda x: encode(x, 0x0F45)),
  (UPat(X86Ops.VFMADD213SS, name="x"), lambda x: encode(x, 0xA9, pp=1, sel=2)), (UPat(X86Ops.VFMADD213SD, name="x"), lambda x: encode(x, 0xA9, pp=1, sel=2, we=1)), # noqa: E501
  (UPat(X86Ops.VFMADD213PS, name="x"), lambda x: encode(x, 0xA8, pp=1, sel=2)), (UPat(X86Ops.VFMADD213PD, name="x"), lambda x: encode(x, 0xA8, pp=1, sel=2, we=1)), # noqa: E501
  (UPat(X86Ops.VBLENDVPS, name="x"), lambda x: encode(x, 0x4A, pp=1, sel=3)), (UPat(X86Ops.VBLENDVPD, name="x"), lambda x: encode(x, 0x4B, pp=1, sel=3)), # noqa: E501
  (UPat(X86Ops.VPBLENDVB, name="x"), lambda x: encode(x, 0x4C, pp=1, sel=3)),
  # shuffles
  (UPat(X86Ops.VPBROADCASTB, name="x"), lambda x: encode(x, 0x78, pp=1, sel=2)), (UPat(X86Ops.VPBROADCASTW, name="x"), lambda x: encode(x, 0x79, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VPBROADCASTD, name="x"), lambda x: encode(x, 0x58, pp=1, sel=2)), (UPat(X86Ops.VPBROADCASTQ, name="x"), lambda x: encode(x, 0x59, pp=1, sel=2)), # noqa: E501
  (UPat(X86Ops.VBROADCASTSS, name="x"), lambda x: encode(x, 0x18, pp=1, sel=2)),
  (UPat(X86Ops.VPINSRB, name="x"), lambda x: encode(x, 0x20, pp=1, sel=3)), (UPat(X86Ops.VPINSRW, name="x"), lambda x: encode(x, 0xC4, pp=1, sel=1)),
  (UPat(X86Ops.VPINSRD, name="x"), lambda x: encode(x, 0x22, pp=1, sel=3)), (UPat(X86Ops.VPINSRQ, name="x"), lambda x: encode(x, 0x22, pp=1, sel=3, we=1)), # noqa: E501
  (UPat(X86Ops.VSHUFPS, name="x"), lambda x: encode(x, 0xC6, pp=0, sel=1)), (UPat(X86Ops.VINSERTPS, name="x"), lambda x: encode(x, 0x21, pp=1, sel=3)), # noqa: E501
  # extract
  (UPat(X86Ops.VPEXTRB, name="x"), lambda x: encode(x, 0x14, pp=1, sel=3)), (UPat(X86Ops.VPEXTRW, name="x"), lambda x: encode(x, 0x15, pp=1, sel=3)),
  (UPat(X86Ops.VPEXTRD, name="x"), lambda x: encode(x, 0x16, pp=1, sel=3)), (UPat(X86Ops.VPEXTRQ, name="x"), lambda x: encode(x, 0x16, pp=1, sel=3, we=1)), # noqa: E501
  # jumps are encoded with a placeholder which gets patched later once the real offset is known
  (UPat(X86Ops.JE), lambda: bytes([0x0F, 0x84]) + int(0).to_bytes(4, 'little', signed=True)),
  (UPat(X86Ops.JNE), lambda: bytes([0x0F, 0x85]) + int(0).to_bytes(4, 'little', signed=True)),
  (UPat(X86Ops.JL), lambda: bytes([0x0F, 0x8C]) + int(0).to_bytes(4, 'little', signed=True)),
  (UPat(X86Ops.JB), lambda: bytes([0x0F, 0x82]) + int(0).to_bytes(4, 'little', signed=True)),
  # return
  (UPat(X86Ops.RET), lambda: bytes([0xC3])),
])

class X86Renderer(ISARenderer):
  device = "CPU"
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (CPU_COUNT.value, 0, 0)
  extra_matcher = extra_matcher
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  isa_spec = isa_spec
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.FDIV, Ops.CMPLT, Ops.CMPEQ, Ops.MAX)}

  def stack_pointer(self) -> UOp: return UOp(X86Ops.DEFINE_REG, dtypes.uint64, arg=RSP)
  def render(self, uops:list[UOp], lower:bool=True) -> str:
    if lower: uops = self.lower(uops[-1])
    targets: set[UOp] = set()
    target_loc: list[UOp, int] = []
    binary = bytearray()
    for u in uops:
      if u.op in (X86Ops.JL, X86Ops.JB, X86Ops.JE, X86Ops.JNE): targets.add(u.src[0])
    for u in uops:
      if u.op in (Ops.GROUP, Ops.NOOP, Ops.AFTER, Ops.BARRIER): continue
      if u.op in (X86Ops.IMM, X86Ops.DEFINE_REG): continue
      if (l:=cast(bytes|None, encodings.rewrite(u))) is None:
        raise RuntimeError(f"failed to encode {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      binary.extend(l)
      if u in targets: target_loc.append(len(binary))
      elif u.op in (X86Ops.JL, X86Ops.JB, X86Ops.JE, X86Ops.JNE):
        binary[-4:] = (target_loc.pop() - len(binary)).to_bytes(4, 'little', signed=True)
    return binary.hex()