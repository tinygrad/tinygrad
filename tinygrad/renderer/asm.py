import struct
from typing import cast
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType, truncate
from tinygrad.muop import Register, Memory, Immediate, Label, Operand, MUOp, MUOpX86, GPR, VEC
from tinygrad.uop.ops import UPat, UOp, Ops, GroupOp, PatternMatcher
from tinygrad.codegen.devectorizer import no_vectorized_alu
from tinygrad.renderer import Renderer
from tinygrad.helpers import DEBUG

def x86_load_consts(x:UOp) -> UOp|None:
  if x.op is Ops.LOAD and x.src[0].op is Ops.CONST: return None
  nsrc = []
  for s in x.src:
    if s.op is Ops.CONST:
      if s.dtype is dtypes.float16: s = s.load(dtype=dtypes.int16).bitcast(dtypes.float16)
      elif s.dtype is dtypes.float32: s = s.load(dtype=dtypes.int32).bitcast(dtypes.float32)
      elif s.dtype is dtypes.float64: s = s.load(dtype=dtypes.int64).bitcast(dtypes.float64)
      elif x.dtype in dtypes.masks: s = s.load()
      elif (x.dtype.count > 1 and x.op not in (Ops.LOAD,)) or abs(s.arg) > dtypes.max(dtypes.int32):
        s = UOp.const(s.dtype, truncate[s.dtype](s.arg)).load()
    nsrc.append(s)
  return x.replace(src=tuple(nsrc)) if tuple(nsrc) != x.src else None

x86_matcher = PatternMatcher([
  # TODO: add negate and rewrite to xor so we can have sub, (const won't be loaded)
  # TODO: probably should go in pre matcher
  # *** vector dtypes ***
  # no packed int64 to floats cast and vice versa
  (UPat(dtype=dtypes.ints64).cast(dtypes.floats, name="alu"), no_vectorized_alu),
  (UPat(dtype=dtypes.floats).cast(dtypes.ints64, name="alu"), no_vectorized_alu),
  # no casts to smaller packed int
  (UPat.var("x", dtypes.ints).cast(dtypes.ints, name="alu"), lambda x,alu: no_vectorized_alu(alu) if x.dtype > alu.dtype else None),
  # no packed int64 mul
  (UPat(Ops.MUL, dtypes.ints64, name="alu"), no_vectorized_alu),
  # no packed idiv
  (UPat(Ops.IDIV, name="alu"), no_vectorized_alu),
  # *** MASKS ***
  # bool CMPNE is XOR, bool CMPLT is XOR+AND, NOTE: cmp of masks is not valid (true mask == nan)
  (UPat.var('x', dtype=(dtypes.bool,)+dtypes.masks).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtype=(dtypes.bool,)+dtypes.masks)<UPat.var('y'), lambda x,y: (x^True)&y),
  # cmp/bitwise of floats/masks are masks
  (UPat(GroupOp.Binary, dtypes.bool, (UPat(dtype=dtypes.floats+dtypes.masks), UPat()), name="x"), lambda x: x.replace(dtype=dtypes.mask32)),
  # convert bools to masks in bitwise source
  (UPat((Ops.CMPNE, Ops.CMPLT, Ops.AND, Ops.OR, Ops.XOR), src=(UPat.var("a", dtypes.bool), UPat.var("b", dtypes.mask32)), name="x"),
   lambda a,b,x: x.replace(dtype=dtypes.mask32, src=(a.cast(dtypes.int32).mul(-1).bitcast(dtypes.mask32), b))),
  (UPat((Ops.CMPNE, Ops.CMPLT, Ops.AND, Ops.OR, Ops.XOR), src=(UPat.var("a", dtypes.mask32), UPat.var("b", dtypes.bool)), name="x"),
   lambda a,b,x: x.replace(dtype=dtypes.mask32, src=(a, b.cast(dtypes.int32).mul(-1).bitcast(dtypes.mask32)))),
  # convert bool to mask in float/packed where
  (UPat.var("m", dtypes.bool).where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda m,a,b: m.cast(dtypes.int32).mul(-1).bitcast(dtypes.mask32).where(a, b)),
  # convert mask to bool in scalar int where
  (UPat.var("m", dtypes.mask32).where(UPat.var("a", dtypes.ints), UPat.var("b")), lambda m,a,b: m.bitcast(dtypes.int32).ne(0).where(a, b)),
  (UPat.var("m", dtypes.mask64).where(UPat.var("a", dtypes.ints), UPat.var("b")), lambda m,a,b: m.bitcast(dtypes.int64).ne(0).where(a, b)),
  # cast from mask is 1 if True, 0 if False
  (UPat.var("y", dtypes.masks).cast(dtypes.ints, name="x"), lambda y,x: y.bitcast(x.dtype).mul(-1)),
  (UPat.var("y", dtypes.masks).cast(dtypes.floats, name="x"), lambda y,x: y.where(x.const_like(1), x.const_like(0))),
  # mask is converted to bool in store
  (UPat.var("a").store(UPat.var("b", dtypes.mask32), allow_any_len=True), lambda a,b: a.store(b.bitcast(dtypes.int32).ne(0))),
  (UPat.var("a").store(UPat.var("b", dtypes.mask64), allow_any_len=True), lambda a,b: a.store(b.bitcast(dtypes.int64).ne(0))),
  # no cmplt for packed ints, y < x => x > y
  ((UPat.var("y", dtypes.ints) < UPat.var("x")).named("cmp"), lambda y,x,cmp: UOp(Ops.CMPGT, cmp.dtype, (x, y)) if y.dtype.count > 1 else None),
  # no cmpne for packed ints, y != x => !(y==x)
  ((UPat.var("y", dtypes.ints) != UPat.var("x")).named("cmp"), lambda y,x,cmp: UOp(Ops.CMPEQ, cmp.dtype, (y,x))^True if y.dtype.count > 1 else None),
  # *** IMMEDIATES ***
  # some consts can't be immediates
  (UPat(GroupOp.All, name="x"), x86_load_consts),
  # some ops can't take imm in srcs
  (UPat((Ops.IDIV, Ops.MOD, Ops.WHERE, Ops.STORE), name="x"),
   lambda x: x.replace(src=nsrc) if (nsrc:=tuple(s.load(dtype=s.dtype) if s.op is Ops.CONST else s for s in x.src)) != x.src else None),
  # TODO: cmpne, add shouldn't have consts on the left to begin with
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.ADD), src=(UPat.cvar("c", dtypes.ints), UPat()), name="x"), lambda x,c: x.replace(src=(c.load(dtype=c.dtype), x.src[1]))),
  # *** CASTS ***
  # rewrite cast to bool to CMPNE 0
  (UPat.var("y").cast(dtypes.bool), lambda y: y != y.const_like(0)),
  # can't cast from float16 to ints/float64 directly and vice versa
  (UPat.var("y", dtypes.float16).cast((dtypes.float64,)+dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", (dtypes.float64,)+dtypes.ints).cast(dtypes.float16, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  # can't cast from float to int8/16 directly and vice versa
  (UPat.var("y", dtypes.floats).cast(dtypes.ints8+dtypes.ints16, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  (UPat.var("y", (dtypes.bool,)+dtypes.ints8+dtypes.ints16).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  # int/float casts only for signed int
  (UPat.var("y", dtypes.uint32).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int64).cast(x.dtype)),
  # casting uint64 to float requires special handling if msb is 1
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uint64),), name="c"),
   lambda c: ((c.src[0] >> 63) != 0).where((c.src[0] & 0x7FFFFFFFFFFFFFFF).cast(dtypes.int64).cast(c.dtype) * 2, \
                                               c.src[0].cast(dtypes.int64).cast(c.dtype))),
  # *** NOOPS ***
  # cast to pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  # cast from pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: x.replace(op=Ops.NOOP) if isinstance(y.dtype, PtrDType) else None),
  # cast to <= scalar int is a noop
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize <= y.dtype.itemsize and y.dtype.count == 1 else None),
  # zero extending scalar 32bit int is a noop
  (UPat.var("y", dtypes.uint32).cast(dtypes.ints64, name="x"), lambda y,x: x.replace(op=Ops.NOOP) if y.dtype.count == 1 else None),
  # vector cast between signed and unsigned is a noop
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize and y.dtype.count > 1 else None),
  # bitcast between mask and float is a noop
  (UPat(dtype=dtypes.masks).bitcast(dtypes.floats).named("x"), lambda x: x.replace(op=Ops.NOOP)),
  # bitcast between signed and unsigned is a noop
  (UPat.var("y", dtypes.ints).bitcast(dtypes.ints).named("x"), lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize else None),
  # bitcast between vectors is a noop
  (UPat.var("y").bitcast().named("x"), lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.count > 1 and y.dtype.count > 1 else None),
  # a gep in a float32 vectorize is a noop and its arg is part of the imm of the instruction
  (UPat(Ops.VECTORIZE, dtypes.float32, name="x"),
   lambda x: x.replace(src=nsrc) if (nsrc:=tuple(s.replace(op=Ops.NOOP) if s.op is Ops.GEP else s for s in x.src)) != x.src else None),
  # loading from register is a noop
  #(UPat(Ops.DEFINE_REG).load(allow_any_len=True, name="x"), lambda x: x.replace(op=Ops.NOOP)),
  # load/store use pointer arithmetic and we get rid of the buf pointer
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))),
   lambda buf,idx: (buf.cast(dtypes.uint64) + idx.cast(dtypes.uint64)*buf.dtype.itemsize)),
  # move mask from INDEX to the load/store
  (UPat.var("buf").index(UPat.var("idx"), UPat.var("gate")).load(UPat.var("alt")),
   lambda buf,idx,gate,alt: buf.index(idx).load(alt, gate, dtype=alt.dtype)),
  (UPat.var("buf").index(UPat.var("idx"), UPat()).store(UPat.var("val"), UPat.var("gate"), allow_any_len=True),
   lambda buf,idx,val,gate: buf.index(idx).store(val, gate)),
  # TODO: replacing recip with fdiv very bad for perf but required for precision, ideally you replace recip + mul with fdiv but imm gets loaded
  # x * (1/b) => x / b, float64 doesn't have recip
  #(UPat.var("a", dtypes.float64) * UPat(Ops.RECIP, name="b"), lambda a,b: UOp(Ops.FDIV, a.dtype, (a, b.src[0]))),
  (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
  # mulacc only available for floats
  (UPat.var('a', dtypes.floats)*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c)),
  # rewrite MAX to CMPLT + WHERE, no max for scalar int and CMPLT + WHERE nearly as fast
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  # no int8 mul or cmove, cast to int16
  (UPat.var("a", dtypes.ints8) * UPat.var("b"), lambda a,b: (a.cast(dtypes.int16) * b.cast(dtypes.int16)).cast(a.dtype)),
  (UPat.var("m").where(UPat.var("a", (dtypes.bool,)+dtypes.ints8), UPat.var("b")),
   lambda m,a,b: m.where(a.cast(dtypes.int16), b.cast(dtypes.int16)).cast(a.dtype) if a.dtype.count == 1 else None),
])

def gep_imm(s,d) -> int: return (s << 6) | (d << 4)
def shuf_imm(x:UOp) -> int: return sum((s.arg[0] if isinstance(s.arg, tuple) else 0) << (2 * i) for i,s in enumerate(x.src))
def x86_setcc(dt:DType, op:Ops):
  if op is Ops.CMPNE: return ("setne", 0x0F95)
  if op is Ops.CMPLT: return ("setl", 0x0F9C) if dt in dtypes.sints else ("setb", 0x0F92)
def x86_mov(dt:DType): return ("mov", 0x8A) if dt in dtypes.ints8+(dtypes.bool,) else ("mov", 0x8B)
def x86_pre(x:UOp):
  if x.dtype in dtypes.ints16: return (0, 0x66)
  if x.dtype in dtypes.ints64: return (1,)
  if x.dtype in (dtypes.bool,)+dtypes.ints8+dtypes.ints32: return ()
  raise RuntimeError("invalid dtype")

#https://www.felixcloutier.com/x86/
# NOTE: LEGACY prefix == VEX prefix
# pp field: None == 0, 66 == 1, F3 == 2, F2 == 3
# map select: 0F == 1, 0F38 == 2, 0F3A == 3
x86_vec_lowerer = PatternMatcher([
  # int binary
  ((UPat.var("a", dtypes.ints32) * UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpmulld", 0x40, ctx[x], ctx[a], ctx[b], 1, 2)),
  ((UPat.var("a", dtypes.ints32) << UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpsllvd", 0x47, ctx[x], ctx[a], ctx[b], 1, 2)),
  ((UPat.var("a", dtypes.ints64) << UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpsllvq", 0x47, ctx[x], ctx[a], ctx[b], 1, 2, 1)),
  ((UPat.var("a", dtypes.uint32) >> UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpsrlvd", 0x45, ctx[x], ctx[a], ctx[b], 1, 2)),
  ((UPat.var("a", dtypes.uint64) >> UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpsrlvq", 0x45, ctx[x], ctx[a], ctx[b], 1, 2, 1)),
  ((UPat.var("a", dtypes.int32) >> UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpsravd", 0x46, ctx[x], ctx[a], ctx[b], 1, 2)),
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpand", 0xDB, ctx[x], ctx[a], ctx[b], 1, 1)), # noqa: E501
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpor", 0xEB, ctx[x], ctx[a], ctx[b], 1, 1)), # noqa: E501
  ((UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpxor", 0xEF, ctx[x], ctx[a], ctx[b], 1, 1)), # noqa: E501
  (UPat(Ops.CMPGT, src=(UPat.var("a", dtypes.ints64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpcmpgtq", 0x37, ctx[x], ctx[a], ctx[b], 1, 2)), # noqa: E501
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.ints64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vpcmpeqq", 0x29, ctx[x], ctx[a], ctx[b], 1, 2)), # noqa: E501
  (UPat(GroupOp.Binary, src=(UPat.var("a", dtypes.ints), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM(*MUOpX86.ivec_ops(x.op, x.dtype.scalar().itemsize), ctx[x], ctx[a], ctx[b], 1, 1)), # noqa: E501
  # int ternary # NOTE: all ints use same cmove with single byte mask granularity
  (UPat.var("m").where(UPat.var("a", dtypes.ints), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: MUOpX86.V_V_VM_V("vpblendvb", 0x4C, ctx[x], ctx[b], ctx[a], ctx[m], 1, 3)), # noqa: E501
  # int moves
  # int shuffles TODO: broadcasts need move
  #(UPat.var("y", dtypes.ints8+(dtypes.bool,)).broadcast(name="x"), lambda ctx,y,x: MUOpX86.X66_0F38("vpbroadcastb", 0x78, ctx[x], (ctx[x],))),
  #(UPat.var("y", dtypes.ints16).broadcast(name="x"), lambda ctx,y,x: MUOpX86.X66_0F38("vpbroadcastw", 0x79, ctx[x], (ctx[x],))),
  #(UPat.var("y", dtypes.ints32).broadcast(name="x"), lambda ctx,y,x: MUOpX86.X66_0F38("vpbroadcastd", 0x58, ctx[x], (ctx[x],))),
  #(UPat.var("y", dtypes.ints64).broadcast(name="x"), lambda ctx,y,x: MUOpX86.X66_0F38("vpbroadcastq", 0x59, ctx[x], (ctx[x],))),
  (UPat(Ops.VECTORIZE, dtypes.ints8+(dtypes.bool,), name="x"), lambda ctx,x: [MUOpX86.V_V_RM_I("vpinsrb", 0x20, ctx[x], (ctx[x], ctx[s], Immediate(i, 1)), 1, 3) for i,s in enumerate(x.src)]), # noqa: E501
  (UPat(Ops.VECTORIZE, dtypes.ints16, name="x"), lambda ctx,x: [MUOpX86.V_V_RM_I("vpinsrw", 0xC4, ctx[x], (ctx[x], ctx[s], Immediate(i, 1)), 1, 1) for i,s in enumerate(x.src)]), # noqa: E501
  (UPat(Ops.VECTORIZE, dtypes.ints32, name="x"), lambda ctx,x: [MUOpX86.V_V_RM_I("vpinsrd", 0x22, ctx[x], (ctx[x], ctx[s], Immediate(i, 1)), 1, 3) for i,s in enumerate(x.src)]), # noqa: E501
  (UPat(Ops.VECTORIZE, dtypes.ints64, name="x"), lambda ctx,x: [MUOpX86.V_V_RM_I("vpinsrq", 0x22, ctx[x], (ctx[x], ctx[s], Immediate(i, 1)), 1, 3, 1) for i,s in enumerate(x.src)]), # noqa: E501
  # int casts from unsigned
  (UPat.var("y", dtypes.uint8).cast(dtypes.ints16, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxbw", 0x30, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.uint8).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxbd", 0x31, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.uint8).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxbq", 0x32, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.uint16).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxwd", 0x33, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.uint16).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxwq", 0x34, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.uint32).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovzxdq", 0x35, ctx[x], ctx[y], 1, 2)),
  # int casts from signed
  (UPat.var("y", dtypes.int8).cast(dtypes.ints16, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxbw", 0x20, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.int8).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxbd", 0x21, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.int8).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxbq", 0x22, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.int16).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxwd", 0x23, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.int16).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxwq", 0x24, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.int32).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vpmovsxdq", 0x25, ctx[x], ctx[y], 1, 2)),
  # int/float casts
  (UPat.var("y", dtypes.int32).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtdq2ps", 0x5B, ctx[x], ctx[y], 0, 1)),
  (UPat.var("y", dtypes.int32).cast(dtypes.float64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtdq2pd", 0xE6, ctx[x], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.float32).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvttps2dq", 0x5B, ctx[x], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.float64).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvttpd2dq", 0xE6, ctx[x], ctx[y], 1, 1)),
  # float casts
  (UPat.var("y", dtypes.float16).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtph2ps", 0x13, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float16, name="x"), lambda ctx,y,x: MUOpX86.VM_V_I("vcvtps2ph", 0x1D, ctx[x], ctx[y], Immediate(4, 1), 1, 3)), # noqa: E501
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtps2pd", 0x5A, ctx[x], ctx[y], 0, 1)),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtpd2ps", 0x5A, ctx[x], ctx[y], 1, 1)),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt().named("x"), lambda ctx,y,x: MUOpX86.V_VM("vsqrtps", 0x51, ctx[x], ctx[y], 0, 1)),
  (UPat.var("y", dtypes.float64).sqrt().named("x"), lambda ctx,y,x: MUOpX86.V_VM("vsqrtpd", 0x51, ctx[x], ctx[y], 1, 1)),
  (UPat.var("y", dtypes.float32).reciprocal().named("x"), lambda ctx,y,x: MUOpX86.V_VM("vrcpss", 0x53, ctx[x], ctx[y], 0, 1)),
  (UPat(Ops.TRUNC, dtypes.float32, (UPat.var("y"),), name="x"), lambda ctx,y,x: MUOpX86.V_VM_I("vroundps", 0x08, ctx[x], ctx[y], Immediate(3, 1), 1, 3)), # noqa: E501
  (UPat(Ops.TRUNC, dtypes.float64, (UPat.var("y"),), name="x"), lambda ctx,y,x: MUOpX86.V_VM_I("vroundpd", 0x09, ctx[x], ctx[y], Immediate(3, 1), 1, 3)), # noqa: E501
  # float cmp
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpltps", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(1, 1), 0, 1)), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpneqps", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(4, 1), 0, 1)), # noqa: E501
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpltpd", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(1, 1), 1, 1)), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpneqpd", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(4, 1), 1, 1)), # noqa: E501
  # float binary
  (UPat(GroupOp.Binary, dtypes.float32, (UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM(*MUOpX86.f_ops(x.op, x.dtype.scalar().itemsize), ctx[x], ctx[a], ctx[b], 0, 1)), # noqa: E501
  (UPat(GroupOp.Binary, dtypes.float64, (UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM(*MUOpX86.f_ops(x.op, x.dtype.scalar().itemsize), ctx[x], ctx[a], ctx[b], 1, 1)), # noqa: E501
  # float ternary TODO: can share with scalar pm
  (UPat.var("m").where(UPat.var("a", dtypes.float32), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: MUOpX86.V_V_VM_V("vblendvps", 0x4A, ctx[x], ctx[b], ctx[a], ctx[m], 1, 3)), # noqa: E501
  (UPat.var("m").where(UPat.var("a", dtypes.float64), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: MUOpX86.V_V_VM_V("vblendvpd", 0x4B, ctx[x], ctx[b], ctx[a], ctx[m], 1, 3)), # noqa: E501
  (UPat(Ops.MULACC, dtypes.float32, name="x"), lambda ctx,x: [MUOpX86.V_VM("vmovups", 0x10, ctx[x], ctx[x.src[0]], 0, 1),
                                                              MUOpX86.V_V_VM("vfmadd213ps", 0xA8, ctx[x], ctx[x.src[1]], ctx[x.src[2]], 1, 2)]),
  (UPat(Ops.MULACC, dtypes.float64, name="x"), lambda ctx,x: [MUOpX86.V_VM("vmovupd", 0x10, ctx[x], ctx[x.src[0]], 1, 1),
                                                              MUOpX86.V_V_VM("vfmadd213pd", 0xA8, ctx[x], ctx[x.src[1]], ctx[x.src[2]], 1, 2, 1)]),
  # float moves TODO: can share with scalar pm
  (UPat.var("a").load(UPat.var("b"), UPat.var("m", dtypes.bool), dtype=dtypes.float32.vec(2), name="x"), lambda ctx,a,b,m,x:
   [MUOpX86.V_VM("vmovq", 0x7E, ctx[x], ctx[b], 2, 1), MUOpX86._RM_I("test", 0xF6, 0, ctx[m], Immediate(1, 1)),
    MUOpX86("je", 0x0F84, ins=(Label(f".IF_{ctx.uops.index(x)}:"),), ins_con=((),)),
    MUOpX86.V_VM("vmovq", 0x7E, ctx[x], Memory(ctx[x].size, ctx[a]), 2, 1), MUOpX86("", -1, Label(f".IF_{ctx.uops.index(x)}:"))]),
  (UPat.var("a").load(UPat.var("b"), UPat.var("m", dtypes.bool), dtype=dtypes.float32.vec(4), name="x"), lambda ctx,a,b,m,x:
   [MUOpX86.V_VM("vmovups", 0x10, ctx[x], ctx[b], 0, 1), MUOpX86._RM_I("test", 0xF6, 0, ctx[m], Immediate(1, 1)),
    MUOpX86("je", 0x0F84, ins=(Label(f".IF_{ctx.uops.index(x)}:"),), ins_con=((),)),
    MUOpX86.V_VM("vmovups", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 0, 1), MUOpX86("", -1, Label(f".IF_{ctx.uops.index(x)}:"))]),
  (UPat.var("a").load(dtype=dtypes.float32.vec(2), name="x"), lambda ctx,a,x: MUOpX86.V_VM("vmovq", 0x7E, ctx[x], Memory(ctx[x].size, ctx[a]), 2, 1)), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.float32.vec(4), name="x"), lambda ctx,a,x: MUOpX86.V_VM("vmovups", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 0, 1)), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.float64.vec(2), name="x"), lambda ctx,a,x: MUOpX86.V_VM("vmovupd", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 1, 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.float32.vec(2)), allow_any_len=True), lambda ctx,a,b: MUOpX86.VM_V("vmovq", 0xD6, Memory(ctx[b].size, ctx[a]), ctx[b], 1, 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.float32.vec(4)), allow_any_len=True), lambda ctx,a,b: MUOpX86.VM_V("vmovups", 0x11, Memory(ctx[b].size, ctx[a]), ctx[b], 0, 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.float64.vec(2)), allow_any_len=True), lambda ctx,a,b: MUOpX86.VM_V("vmovupd", 0x11, Memory(ctx[b].size, ctx[a]), ctx[b], 1, 1)), # noqa: E501
  # float32 shuffles, if all elements share same src it's a single instruction otherwise they are inserted individually
  (UPat(Ops.VECTORIZE, dtypes.float32, (UPat.var(name="y"),), allow_any_len=True, name="x"), lambda ctx,y,x:
   MUOpX86.V_V_VM_I("vshufps", 0xC6, ctx[x], ctx[y], ctx[y], Immediate(shuf_imm(x), 1), 0, 1) if all(s.src == y.src for s in x.src) else \
   [MUOpX86.V_V_VM_I("vinsertps", 0x21, ctx[x], ctx[x], ctx[s], Immediate(gep_imm(s.arg[0] if s.op is Ops.NOOP and isinstance(s.arg, tuple) else 0,i), 1), 1, 3) for i,s in enumerate(x.src)]), # noqa: E501
])

x86_lowerer = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x86_vec_lowerer.rewrite(x, ctx) if x.dtype.count > 1 or x.op is Ops.STORE and x.src[1].dtype.count > 1 else None), # noqa: E501
  # defines, define global is modeled as a move from real to vitual
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: MUOpX86("mov", 0x8B, ctx[x], (ctx[x],), GPR, ((GPR[[7,6,2,1,8,9][x.arg]],),), reg=ctx[x], rm=ctx[x], w=1) if x.arg < 6 else \
                                                    MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(8, Register("rbp", 5, 8), disp=Immediate((x.arg-5)*8+8, 4)), 1)), # noqa: E501
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x: [MUOpX86.R_RM("mov", 0x8B, ctx[x], Register("rbp", 5, 8), 1), MUOpX86.RM_I("sub", 0x81, 5, ctx[x], Immediate(ctx.stack_size, 4), 1)]), # noqa: E501
  # TODO: idiv has unique register constraints
  ((UPat.var("a", dtypes.int32) // UPat.var("b")).named("x"), lambda ctx,a,b,x: [MUOpX86("mov", 0x8B, ctx[x], (ctx[a],), (Register("rax", 0, 4),), (GPR,), ctx[x], ctx[a]),
                                                                                 MUOpX86._RM("push", 0xFF, 6, Register("rdx", 2, 8)),
                                                                                 MUOpX86("cdq", 0x99),
                                                                                 MUOpX86._RM("idiv", 0xF7, 7, ctx[b], in_cons=tuple(r for r in GPR if r.name not in ("rax", "rdx"))),
                                                                                 MUOpX86._RM("pop", 0x8F, 0, Register("rdx", 2, 8))]),
  ((UPat.var("a", dtypes.uint64) // UPat.var("b")).named("x"), lambda ctx,a,b,x: [MUOpX86("mov", 0x8B, ctx[x], (ctx[a],), (Register("rax", 0, 8),), (GPR,), ctx[x], ctx[a], w=1),
                                                                                  MUOpX86._RM("push", 0xFF, 6, Register("rdx", 2, 8)),
                                                                                  MUOpX86.R_RM("xor", 0x33, Register("rdx", 2, 8), Register("rdx", 2, 8), 1),
                                                                                  MUOpX86._RM("div", 0xF7, 6, ctx[b], 1, in_cons=tuple(r for r in GPR if r.name not in ("rax", "rdx"))),
                                                                                  MUOpX86._RM("pop", 0x8F, 0, Register("rdx", 2, 8))]),
  # TODO: use compact immediate versions, add r32 imm32 can be add r32 imm8 if within range
  # int binary with immediate
  ((UPat.var("a", dtypes.ints16) * UPat.cvar("c")).named("x"), lambda ctx,a,c,x: MUOpX86.R_RM_I("imul", 0x69, ctx[x], ctx[a], Immediate(c.arg, 2), 0, 0x66)), # noqa: E501
  ((UPat.var("a", dtypes.ints32) * UPat.cvar("c")).named("x"), lambda ctx,a,c,x: MUOpX86.R_RM_I("imul", 0x69, ctx[x], ctx[a], Immediate(c.arg, 4))),
  ((UPat.var("a", dtypes.ints64) * UPat.cvar("c")).named("x"), lambda ctx,a,c,x: MUOpX86.R_RM_I("imul", 0x69, ctx[x], ctx[a], Immediate(c.arg, 4), 1)), # noqa: E501
  (UPat((Ops.SHL, Ops.SHR), dtypes.ints, (UPat.var("a"), UPat.cvar("c")), name="x"), lambda ctx,a,c,x: [MUOpX86.R_RM(*x86_mov(x.dtype), ctx[x], ctx[a], *x86_pre(x)), # noqa: E501
                                                                                                        MUOpX86.RM_I(*MUOpX86.imm_ops(x.op, x.dtype), ctx[x], Immediate(c.arg, 1), *x86_pre(x))]), # noqa: E501
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(UPat.var("a", dtypes.ints), UPat.cvar("c")), name="x"), lambda ctx,a,c,x: [MUOpX86._RM_I(*MUOpX86.imm_ops(x.op, a.dtype), ctx[a], Immediate(c.arg, min(c.dtype.itemsize, 4)), *x86_pre(a)), # noqa: E501
                                                                                                                MUOpX86.RM(*x86_setcc(x.src[0].dtype, x.op), ctx[x])]), # noqa: E501
  (UPat(GroupOp.Binary, src=(UPat.var("a", dtypes.ints+(dtypes.bool,)), UPat.cvar("c")), name="x"), lambda ctx,a,c,x: [MUOpX86.R_RM(*x86_mov(x.dtype), ctx[x], ctx[a], *x86_pre(x)), # noqa: E501
                                                                                                                       MUOpX86.RM_I(*MUOpX86.imm_ops(x.op, x.dtype), ctx[x], Immediate(c.arg, min(c.dtype.itemsize, 4)), *x86_pre(x))]), # noqa: E501
  # int binary with register/memory
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(UPat.var("a", dtypes.ints), UPat.var("b")), name="x"), lambda ctx,a,b,x: [MUOpX86._R_RM(*MUOpX86.i_ops(x.op, a.dtype), ctx[a], ctx[b], *x86_pre(a)), # noqa: E501
                                                                                                               MUOpX86.RM(*x86_setcc(x.src[0].dtype, x.op), ctx[x])]), # noqa: E501
  (UPat(GroupOp.Binary, src=(UPat.var("a", dtypes.ints+(dtypes.bool,)), UPat.var("b")), name="x"), lambda ctx,a,b,x: [MUOpX86.R_RM(*x86_mov(x.dtype), ctx[x], ctx[a], *x86_pre(x)), # noqa: E501
                                                                                                                      MUOpX86.R_RM(*MUOpX86.i_ops(x.op, x.dtype), ctx[x], ctx[b], *x86_pre(x))]), # noqa: E501
  # int ternary TODO: shouldn't need to set the flag everytime
  (UPat.var("m").where(UPat.var("a", dtypes.ints16+dtypes.ints32+dtypes.ints64), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: [MUOpX86.R_RM("mov", 0x8B, ctx[x], ctx[a], *x86_pre(x)), # noqa: E501
                                                                                                                                 MUOpX86._RM_I("test", 0xF6, 0, ctx[m], Immediate(1, 1)), # noqa: E501
                                                                                                                                 MUOpX86.R_RM("cmove", 0x0F44, ctx[x], ctx[b], *x86_pre(x))]), # noqa: E501
  # immediate loads
  (UPat.cvar("c", dtypes.ints8+(dtypes.bool,)).load(name="x"), lambda ctx,c,x: MUOpX86.RM_I("mov", 0xC6, 0, ctx[x], Immediate(c.arg, 1))),
  (UPat.cvar("c", dtypes.ints16).load(name="x"), lambda ctx,c,x: MUOpX86.RM_I("mov", 0xC7, 0, ctx[x], Immediate(c.arg, 2), 0, 0x66)),
  (UPat.cvar("c", dtypes.ints32).load(name="x"), lambda ctx,c,x: MUOpX86.RM_I("mov", 0xC7, 0, ctx[x], Immediate(c.arg, 4))),
  (UPat.cvar("c", dtypes.ints64).load(name="x"), lambda ctx,c,x: MUOpX86.R_I("movabs", 0xB8, ctx[x], Immediate(c.arg, 8), 1)),
  (UPat.cvar("c", dtypes.float32).load(dtype=dtypes.int32, name="x"), lambda ctx,c,x: MUOpX86.RM_I("mov", 0xC7, 0, ctx[x], Immediate(int.from_bytes(struct.pack("<f", c.arg), "little"), 4))), # noqa: E501
  (UPat.cvar("c", dtypes.float64).load(dtype=dtypes.int64, name="x"), lambda ctx,c,x: MUOpX86.R_I("movabs", 0xB8, ctx[x], Immediate(int.from_bytes(struct.pack("<d", c.arg), "little"), 8), 1)), # noqa: E501
  # int moves
  #(UPat.var("a").load(UPat.cvar("c"), dtype=dtypes.int32, name="x"), lambda ctx,a,c,x: MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(ctx[x].size, ctx[a], disp=ctx[c]))),
  (UPat.var("a").load(UPat.cvar("c"), UPat.var("m", dtypes.bool), dtype=dtypes.ints32, name="x"), lambda ctx,a,c,m,x: [MUOpX86.RM_I("mov", 0xC7, 0, ctx[x], Immediate(c.arg, 4)), # noqa: E501
                                                                                                                       MUOpX86._RM_I("test", 0xF6, 0, ctx[m], Immediate(1, 1)), # noqa: E501
                                                                                                                       MUOpX86("je", 0x0F84, ins=(Label(f".IF_{ctx.uops.index(x)}:"),), ins_con=((),)), # noqa: E501
                                                                                                                       MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(ctx[x].size, ctx[a])), # noqa: E501
                                                                                                                       MUOpX86("", -1, Label(f".IF_{ctx.uops.index(x)}:"))]), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.ints8+(dtypes.bool,), allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.R_RM("mov", 0x8A, ctx[x], Memory(ctx[x].size, ctx[a]))), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.ints16, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(ctx[x].size, ctx[a]), 0, 0x66)), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.ints32, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(ctx[x].size, ctx[a]))), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.ints64, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.R_RM("mov", 0x8B, ctx[x], Memory(ctx[x].size, ctx[a]), 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.ints8+(dtypes.bool,)), allow_any_len=True), lambda ctx,a,b: MUOpX86.RM_R("mov", 0x88, Memory(ctx[a].size, ctx[a]), ctx[b])), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.ints16), allow_any_len=True), lambda ctx,a,b: MUOpX86.RM_R("mov", 0x89, Memory(ctx[a].size, ctx[a]), ctx[b], 0, 0x66)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.ints32), allow_any_len=True), lambda ctx,a,b: MUOpX86.RM_R("mov", 0x89, Memory(ctx[a].size, ctx[a]), ctx[b])), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.ints64), allow_any_len=True), lambda ctx,a,b: MUOpX86.RM_R("mov", 0x89, Memory(ctx[a].size, ctx[a]), ctx[b], 1)), # noqa: E501
  # int extract
  (UPat.var("y", dtypes.ints8).gep(name="x"), lambda ctx,y,x: MUOpX86.RM_V_I("vpextrb", 0x14, ctx[x], ctx[y], Immediate(x.arg[0], 1), 1, 3)),
  (UPat.var("y", dtypes.ints16).gep(name="x"), lambda ctx,y,x: MUOpX86.RM_V_I("vpextrw", 0x15, ctx[x], ctx[y], Immediate(x.arg[0], 1), 1, 3)),
  (UPat.var("y", dtypes.ints32).gep(name="x"), lambda ctx,y,x: MUOpX86.RM_V_I("vpextrd", 0x16, ctx[x], ctx[y], Immediate(x.arg[0], 1), 1, 3)),
  (UPat.var("y", dtypes.ints64).gep(name="x"), lambda ctx,y,x: MUOpX86.RM_V_I("vpextrq", 0x16, ctx[x], ctx[y], Immediate(x.arg[0], 1), 1, 3, 1)),
  # int casts from unsigned
  (UPat.var("y", (dtypes.uint8, dtypes.bool)).cast(dtypes.ints16, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movzx", 0x0FB6, ctx[x], ctx[y], 0, 0x66)),
  (UPat.var("y", (dtypes.uint8, dtypes.bool)).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movzx", 0x0FB6, ctx[x], ctx[y])),
  (UPat.var("y", (dtypes.uint8, dtypes.bool)).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movzx", 0x0FB6, ctx[x], ctx[y], 1)),
  (UPat.var("y", dtypes.uint16).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movzx", 0x0FB7, ctx[x], ctx[y])),
  (UPat.var("y", dtypes.uint16).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movzx", 0x0FB7, ctx[x], ctx[y], 1)),
  # int casts from signed
  (UPat.var("y", dtypes.int8).cast(dtypes.ints16, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsx", 0x0FBE, ctx[x], ctx[y], 0, 0x66)),
  (UPat.var("y", dtypes.int8).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsx", 0x0FBE, ctx[x], ctx[y])),
  (UPat.var("y", dtypes.int8).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsx", 0x0FBE, ctx[x], ctx[y], 1)),
  (UPat.var("y", dtypes.int16).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsx", 0x0FBF, ctx[x], ctx[y])),
  (UPat.var("y", dtypes.int16).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsx", 0x0FBF, ctx[x], ctx[y], 1)),
  (UPat.var("y", dtypes.int32).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_RM("movsxd", 0x63, ctx[x], ctx[y], 1)),
  # int/float bitcasts
  (UPat.var("y", dtypes.ints32).bitcast((dtypes.float32, dtypes.mask32)).named("x"), lambda ctx,y,x: MUOpX86.V_RM("vmovd", 0x6E, ctx[x], ctx[y], 1, 1)), # noqa: E501
  (UPat.var("y", dtypes.ints64).bitcast((dtypes.float64, dtypes.mask64)).named("x"), lambda ctx,y,x: MUOpX86.V_RM("vmovq", 0x6E, ctx[x], ctx[y], 1, 1, 1)), # noqa: E501
  (UPat.var("y", (dtypes.float32, dtypes.mask32)).bitcast(dtypes.ints32).named("x"), lambda ctx,y,x: MUOpX86.RM_V("vmovd", 0x7E, ctx[x], ctx[y], 1, 1)), # noqa: E501
  (UPat.var("y", (dtypes.float64, dtypes.mask64)).bitcast(dtypes.ints64).named("x"), lambda ctx,y,x: MUOpX86.RM_V("vmovq", 0x7E, ctx[x], ctx[y], 1, 1, 1)), # noqa: E501
  # int/float casts
  (UPat.var("y", dtypes.int32).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_V_RM("vcvtsi2ss", 0x2A, ctx[x], ctx[x], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.int32).cast(dtypes.float64, name="x"), lambda ctx,y,x: MUOpX86.V_V_RM("vcvtsi2sd", 0x2A, ctx[x], ctx[x], ctx[y], 3, 1)),
  (UPat.var("y", dtypes.int64).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_V_RM("vcvtsi2ss", 0x2A, ctx[x], ctx[x], ctx[y], 2, 1, 1)),
  (UPat.var("y", dtypes.int64).cast(dtypes.float64, name="x"), lambda ctx,y,x: MUOpX86.V_V_RM("vcvtsi2sd", 0x2A, ctx[x], ctx[x], ctx[y], 3, 1, 1)),
  (UPat.var("y", dtypes.float32).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_VM("vcvttss2si", 0x2C, ctx[x], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.float32).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_VM("vcvttss2si", 0x2C, ctx[x], ctx[y], 2, 1, 1)),
  (UPat.var("y", dtypes.float64).cast(dtypes.ints32, name="x"), lambda ctx,y,x: MUOpX86.R_VM("vcvttsd2si", 0x2C, ctx[x], ctx[y], 3, 1)),
  (UPat.var("y", dtypes.float64).cast(dtypes.ints64, name="x"), lambda ctx,y,x: MUOpX86.R_VM("vcvttsd2si", 0x2C, ctx[x], ctx[y], 3, 1, 1)),
  # float casts NOTE: float16 casts are packed
  (UPat.var("y", dtypes.float16).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_VM("vcvtph2ps", 0x13, ctx[x], ctx[y], 1, 2)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float16, name="x"), lambda ctx,y,x: MUOpX86.VM_V_I("vcvtps2ph", 0x1D, ctx[x], ctx[y], Immediate(4, 1), 1, 3)), # noqa: E501
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda ctx,y,x: MUOpX86.V_V_VM("vcvtss2sd", 0x5A, ctx[x], ctx[x], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda ctx,y,x: MUOpX86.V_V_VM("vcvtsd2ss", 0x5A, ctx[x], ctx[x], ctx[y], 3, 1)),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt().named("x"), lambda ctx,y,x: MUOpX86.V_V_VM("vsqrtss", 0x51, ctx[x], ctx[y], ctx[y], 2, 1)),
  (UPat.var("y", dtypes.float64).sqrt().named("x"), lambda ctx,y,x: MUOpX86.V_V_VM("vsqrtsd", 0x51, ctx[x], ctx[y], ctx[y], 3, 1)),
  (UPat.var("y", dtypes.float32).reciprocal().named("x"), lambda ctx,y,x: MUOpX86.V_V_VM("vrcpss", 0x53, ctx[x], ctx[y], ctx[y], 2, 1)),
  (UPat(Ops.TRUNC, dtypes.float32, (UPat.var("y"),), name="x"), lambda ctx,y,x: MUOpX86.V_V_VM_I("vroundss", 0x0A, ctx[x], ctx[y], ctx[y], Immediate(3, 1), 1, 3)), # noqa: E501
  (UPat(Ops.TRUNC, dtypes.float64, (UPat.var("y"),), name="x"), lambda ctx,y,x: MUOpX86.V_V_VM_I("vroundsd", 0x0B, ctx[x], ctx[y], ctx[y], Immediate(3, 1), 1, 3)), # noqa: E501
  # float cmp
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpltss", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(1, 1), 2, 1)), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpneqss", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(4, 1), 2, 1)), # noqa: E501
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpltsd", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(1, 1), 3, 1)), # noqa: E501
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM_I("vcmpneqsd", 0xC2, ctx[x], ctx[a], ctx[b], Immediate(4, 1), 3, 1)), # noqa: E501
  # mask binary NOTE: only bitwise and packed
  ((UPat.var("a", dtypes.mask32) & UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vandps", 0x54, ctx[x], ctx[a], ctx[b], 0, 1)),
  ((UPat.var("a", dtypes.mask64) & UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vandpd", 0x54, ctx[x], ctx[a], ctx[b], 1, 1)),
  ((UPat.var("a", dtypes.mask32) | UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vorps", 0x56, ctx[x], ctx[a], ctx[b], 0, 1)),
  ((UPat.var("a", dtypes.mask64) | UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vorpd", 0x56, ctx[x], ctx[a], ctx[b], 1, 1)),
  ((UPat.var("a", dtypes.mask32) ^ UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vxorps", 0x57, ctx[x], ctx[a], ctx[b], 0, 1)),
  ((UPat.var("a", dtypes.mask64) ^ UPat.var("b")).named("x"), lambda ctx,a,b,x: MUOpX86.V_V_VM("vxorpd", 0x57, ctx[x], ctx[a], ctx[b], 1, 1)),
  # float binary
  (UPat(GroupOp.Binary, dtypes.float32, (UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM(*MUOpX86.f_ops(x.op, x.dtype.scalar().itemsize), ctx[x], ctx[a], ctx[b], 2, 1)), # noqa: E501
  (UPat(GroupOp.Binary, dtypes.float64, (UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,a,b,x: MUOpX86.V_V_VM(*MUOpX86.f_ops(x.op, x.dtype.scalar().itemsize), ctx[x], ctx[a], ctx[b], 3, 1)), # noqa: E501
  # float ternary NOTE: cmove is packed
  (UPat.var("m").where(UPat.var("a", dtypes.float32), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: MUOpX86.V_V_VM_V("vblendvps", 0x4A, ctx[x], ctx[b], ctx[a], ctx[m], 1, 3)), # noqa: E501
  (UPat.var("m").where(UPat.var("a", dtypes.float64), UPat.var("b")).named("x"), lambda ctx,m,a,b,x: MUOpX86.V_V_VM_V("vblendvpd", 0x4B, ctx[x], ctx[b], ctx[a], ctx[m], 1, 3)), # noqa: E501
  (UPat(Ops.MULACC, dtypes.float32, name="x"), lambda ctx,x: [MUOpX86.V_V_V("vmovss", 0x10, ctx[x], ctx[x.src[0]], ctx[x.src[0]], 2, 1),
                                                              MUOpX86.V_V_VM("vfmadd213ss", 0xA9, ctx[x], ctx[x.src[1]], ctx[x.src[2]], 1, 2)]),
  (UPat(Ops.MULACC, dtypes.float64, name="x"), lambda ctx,x: [MUOpX86.V_V_V("vmovsd", 0x10, ctx[x], ctx[x.src[0]], ctx[x.src[0]], 3, 1),
                                                              MUOpX86.V_V_VM("vfmadd213sd", 0xA9, ctx[x], ctx[x.src[1]], ctx[x.src[2]], 1, 2, 1)]),
  # float moves, TODO: this could be vmaskmovps
  (UPat.var("a").load(UPat.var("b"), UPat.var("m", dtypes.bool), dtype=dtypes.float32, name="x"), lambda ctx,a,b,m,x: [MUOpX86.V_V_V("vmovss", 0x10, ctx[x], ctx[b], ctx[b], 2, 1), # noqa: E501
                                                                                                                       MUOpX86._RM_I("test", 0xF6, 0, ctx[m], Immediate(1, 1)), # noqa: E501
                                                                                                                       MUOpX86("je", 0x0F84, ins=(Label(f".IF_{ctx.uops.index(x)}:"),), ins_con=((),)), # noqa: E501
                                                                                                                       MUOpX86.V_M("vmovss", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 2, 1), # noqa: E501
                                                                                                                       MUOpX86("", -1, Label(f".IF_{ctx.uops.index(x)}:"))]), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.float64, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.V_M("vmovsd", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 3, 1)), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.float32, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.V_M("vmovss", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 2, 1)), # noqa: E501
  (UPat.var("a").load(dtype=dtypes.float64, allow_any_len=True, name="x"), lambda ctx,a,x: MUOpX86.V_M("vmovsd", 0x10, ctx[x], Memory(ctx[x].size, ctx[a]), 3, 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.float32), allow_any_len=True), lambda ctx,a,b: MUOpX86.M_V("vmovss", 0x11, Memory(ctx[b].size, ctx[a]), ctx[b], 2, 1)), # noqa: E501
  (UPat.var("a").store(UPat.var("b", dtypes.float64), allow_any_len=True), lambda ctx,a,b: MUOpX86.M_V("vmovsd", 0x11, Memory(ctx[b].size, ctx[a]), ctx[b], 3, 1)), # noqa: E501
  # float extract TODO: add float64
  (UPat.var("y", dtypes.float32).gep(name="x"), lambda ctx,y,x: MUOpX86.V_V_VM_I("vinsertps", 0x21, ctx[x], ctx[x], ctx[y], Immediate(gep_imm(x.arg[0],0), 1), 1, 3)), # noqa: E501
  # range / endrange
  (UPat(Ops.RANGE, dtypes.int32, name="x"), lambda ctx,x: [MUOpX86.RM_I("mov", 0xC7, 0, ctx[x], Immediate(0, 4)),
                                                           MUOpX86("", -1, Label(f".LOOP_{x.arg}:"))]),
  (UPat(Ops.ENDRANGE, dtypes.void, (UPat(Ops.RANGE, dtypes.int32, (UPat.cvar("c"),), name="a"),)), lambda ctx,c,a: [MUOpX86.RM_I("add", 0x81, 0, ctx[a], Immediate(1, 4)), # noqa: E501
                                                                                                                    MUOpX86._RM_I("cmp", 0x81, 7, ctx[a], Immediate(c.arg, 4)), # noqa: E501
                                                                                                                    MUOpX86("jl", 0x0F8C, ins=(Label(f".LOOP_{a.arg}:"),), ins_con=((),))]), # noqa: E501
])

class X86Renderer(Renderer):
  device = "X86"
  has_local = False
  global_max = None
  extra_matcher = x86_matcher
  lowerer = x86_lowerer
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.SHL, Ops.SHR)}
  function_name: str = ""

  def __getitem__(self, x:UOp) -> str: # hacky helper
    # TODO: register size should probably come from MUOp not dtype?
    assert x.op is not Ops.CONST, "const is an immediate"
    if x in self.virtuals: return self.virtuals[x]
    # is this a hack? they're different types of noops, noop geps don't change the register size, others do
    if x.op is Ops.NOOP:
      if x.arg is not None: self.virtuals[x] = self.virtuals[x.src[0]]
      else: self.virtuals[x] = Register(self.virtuals[x.src[0]].name, 0, x.dtype.itemsize)
    else: self.virtuals[x] = Register(f"v{len(self.virtuals)}", 0, x.dtype.itemsize if not isinstance(x.dtype, PtrDType) else 8)
    return self.virtuals[x]

  def lower(self, uops:list[UOp]) -> list[MUOp]:
    self.uops = uops
    self.virtuals: dict[UOp, Register] = {}
    muops: list[MUOp] = []
    # not a fan of this being here
    self.stack_size = 0
    for u in uops:
      if u.op is Ops.SINK:
        if u.arg is not None: self.name = u.arg.function_name
        continue
      if u.op is Ops.CONST: continue
      if u.op is Ops.NOOP: continue
      if u.op is Ops.DEFINE_REG: self.stack_size += u.dtype.itemsize * cast(PtrDType, u.dtype).size
      if (mu:=cast(MUOp|list[MUOp], self.lowerer.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to lower {u.op} to MUOp with {u.dtype} srcs {[x.dtype for x in u.src]}")
      muops.extend([mu] if isinstance(mu, MUOp) else mu)
    if DEBUG >= 8: print("\n".join([str(mu) for mu in muops] + ["\n"]))
    return muops

  def regalloc(self, muops: list[MUOp]) -> list[MUOp]:
    # live ranges
    def is_range(x:Operand) -> bool: return isinstance(x, Label) and x.name.startswith(".LOOP")
    def virtuals(n:tuple[Operand, ...]) -> list[Register]:
      l = []
      for x in n:
        if isinstance(x, Register): l.append(x)
        elif isinstance(x, Memory):
          l.append(x.base)
          if x.index is not None: l.append(x.index)
      return l
    # first pass builds ranges
    live_range: dict[Register, list[int]] = {}
    label_range: dict[Label, list[int]] = {}
    for i,mu in enumerate(muops):
      if is_range(mu.out): label_range[mu.out] = [i]
      if mu.ins and is_range(mu.ins[0]): label_range[mu.ins[0]].append(i)
      for v in virtuals((mu.out,) + mu.ins):
        if v not in live_range: live_range[v] = [i]
        else: live_range[v].append(i)
    # second pass updates end of range, a var defined before a range and used inside it is needed for the whole range
    ranges: list[Label] = []
    for i,mu in enumerate(reversed(muops)):
      for v in virtuals(mu.ins):
        end = next((label_range[rng][1] for rng in ranges if live_range[v][0] < label_range[rng][0]), 0)
        if end > live_range[v][-1]: live_range[v].append(end)
      if mu.ins and is_range(mu.ins[0]): ranges.append(mu.ins[0])
      if is_range(mu.out): ranges.pop()

    # allocate registers
    reg_pool: list[Register] = [reg for reg in GPR + VEC if reg.name not in ("rbp", "rsp")]
    live: dict[Register, Register] = {}
    mem: dict[Register, Memory] = {}
    final_muops: list[MUOp] = []

    def alloc(cons:tuple[Register, ...]) -> Register:
      # allocate free register, otherwise spill one
      if (idx:=next((i for i,r in enumerate(reg_pool) if r in cons), None)) is not None: return reg_pool.pop(idx)
      # choose the virtual with the latest next use
      spilled = max([k for k,v in live.items() if v in cons], key=lambda k: next(j for j in live_range[k] if j >= i))
      if spilled not in mem:
        offset = self.stack_size + (spilled.size - self.stack_size % spilled.size) % spilled.size
        self.stack_size = offset + spilled.size
        mem[spilled] = Memory(spilled.size, Register("rbp", 5, 8), disp=Immediate(-self.stack_size, 4))
        # TODO: hoist store
        if final_muops[-1].opcode == 0x0F84: final_muops.insert(-1, MUOpX86.store(mem[spilled], live[spilled]))
        else: final_muops.append(MUOpX86.store(mem[spilled], live[spilled]))
      return live.pop(spilled)

    def rewrite(x:Operand, cons:tuple[Register, ...], mu:MUOpX86|None=None) -> Operand:
      if isinstance(x, Register):
        if x in GPR: return x #HACK
        if x in live and live[x] not in cons:
          reg = alloc(cons)
          reg = Register(reg.name, reg.index, x.size)
          final_muops.append(MUOpX86.assign(reg, live[x]))
          reg_pool.insert(0, live.pop(x))
          live[x] = reg
        elif x not in live:
          # TODO: breaks some tests
          # if last use of x and it can be memory don't load, if x is in another field it needs to be loaded anyway
          #if mu is not None and live_range[x][-1] == i and x in mem and x is mu.rm and x not in (mu.reg, mu.vvvv, mu.imm): return mem[x]
          reg = alloc(cons)
          live[x] = Register(reg.name, reg.index, x.size)
          if x in mem: final_muops.append(MUOpX86.load(live[x], mem[x]))
        elif x in live: live[x] = Register(live[x].name, live[x].index, x.size)
        return live[x]
      if isinstance(x, Memory):
        for v in (v for v in [x.base, x.index] if v is not None and v not in live):
          if v in GPR: return x #HACK
          assert v in mem, v
          reg = alloc(GPR)
          live[v] = Register(reg.name, reg.index, v.size)
          # HACK: can't load inside branch, this happens in conditional load
          if final_muops[-1].opcode == 0x0F84: final_muops.insert(-1, MUOpX86.load(live[v], mem[v]))
          else: final_muops.append(MUOpX86.load(live[v], mem[v]))
        return Memory(x.size, live[x.base], live.get(x.index, None), x.scale, x.disp)
      return x

    live_at_range: list[dict[Register, Register]] = []
    for i,mu in enumerate(muops):
      assert len(set(reg_pool)) == len(reg_pool), [str(r) for r in reg_pool]
      assert len(set(live.values())) == len(live.values())
      # TODO: if out is in ins can't coalesce but should
      # save registers at loop entry
      if is_range(mu.out): live_at_range.append(live.copy())
      # reload registers before next loop iteration
      if mu.ins and is_range(mu.ins[0]):
        rset = set(live_at_range.pop().items())
        patch: list[tuple[Register, Register]] = []
        while rset:
          out_degrees = {live.get(v, mem.get(v)) for v,_ in rset}
          in_degrees = {(v,r) for v,r in rset if r not in out_degrees or live.get(v) == r}
          # handle cycles by removing one and loading it at the end
          if not in_degrees:
            # TODO: the store is just for the loop accumulator, everything else doesn't need it, should remove it
            chosen = rset.pop()
            final_muops.append(MUOpX86.store(mem[chosen[0]], live[chosen[0]]))
            patch.append(chosen)
          rset -= in_degrees
          for v,r in in_degrees: rewrite(v, [r])
        for v,r in patch:
          assert v not in live
          rewrite(v, [r])
      # free dead registers
      for v in [v for v in live if live_range[v][-1] < i]: reg_pool.insert(0, live.pop(v))
      # rewrite sources
      ins_rewrite = tuple(rewrite(v, con, mu) for v,con in zip(mu.ins, mu.ins_con))
      # free registers before rewriting destination to coalesce
      for v in mu.ins:
        if isinstance(v, Register) and live_range[v][-1] == i and v in live and isinstance(mu.out, Register): reg_pool.insert(0, live.pop(v))
      # rewrite MUOp with real operands
      final_muops.append(mu.replace(rewrite(mu.out, mu.out_con, mu), ins_rewrite))
    return final_muops

  def setup(self, name:str, kernel:list[MUOp], stack_size:int) -> list[MUOp]:
    prologue = [MUOpX86._RM("push", 0xFF, 6, Register("rbp", 5, 8)),
                MUOpX86.R_RM("mov", 0x8B, Register("rbp", 5, 8), Register("rsp", 4, 8), 1),
                MUOpX86.RM_I("sub", 0x81, 5, Register("rsp", 4, 8), Immediate(stack_size, 4), 1)]
    epilogue = [MUOpX86.RM_I("add", 0x81, 0, Register("rsp", 4, 8), Immediate(stack_size, 4), 1),
                MUOpX86._RM("pop", 0x8F, 0, Register("rbp", 5, 8))]
    kernel = prologue + kernel + epilogue if stack_size > 0 else kernel
    return kernel + [MUOpX86("ret", 0xC3)]
  def to_muops(self, uops: list[UOp]) -> list[MUOp]: return self.setup("", self.regalloc(self.lower(uops)), self.stack_size)
