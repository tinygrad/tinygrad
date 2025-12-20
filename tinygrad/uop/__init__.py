# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
from enum import auto, IntEnum, Enum

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  # ** 1 -- defines/special **

  # define GLOBAL/VAR are ptrs to outside the Kernel
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); BIND = auto()

  # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  SPECIAL = auto()

  # define LOCAL/REG allocate things
  DEFINE_LOCAL = auto(); DEFINE_REG = auto()

  # ** 2 -- non op uops **

  # uops that aren't rendered
  NOOP = auto(); REWRITE_ERROR = auto()

  # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
  # GROUP is a NOOP that just merges things together
  SINK = auto(); AFTER = auto(); GROUP = auto()

  # vector creation / item selection
  GEP = auto(); VECTORIZE = auto()

  # ** 3 -- load/store **

  # INDEX is a BinaryOp similar to ADD, but it operates on pointers
  INDEX = auto()

  # load/store before math
  LOAD = auto(); STORE = auto()

  # ** 4 -- math **

  # tensor core math op, not elementwise
  WMMA = auto()

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  # TernaryOps
  WHERE = auto(); MULACC = auto()

  # ** 5 -- control flow / consts / custom **

  # control flow ops
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()

  # consts. VCONST is a vectorized const
  VCONST = auto(); CONST = auto()

  # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
  CUSTOM = auto(); CUSTOMI = auto()

  # ** 6 -- ops that don't exist in programs **

  # tensor graph ops
  UNIQUE = auto(); DEVICE = auto(); KERNEL = auto(); ASSIGN = auto()
  CUSTOM_KERNEL = auto()

  # local unique
  LUNIQUE = auto()

  # ops that adjust the behavior of the scheduler
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # buffer ops
  BUFFERIZE = auto(); COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto(); ENCDEC = auto()

  # the core 6 movement ops! these only exist in the tensor graph
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()
  MULTI = auto()  # MULTI is really a movement op

  # reduce
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto()

  # expander ops
  UNROLL = auto(); CONTRACT = auto(); CAT = auto(); PTRCAT = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})

  Defines = {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  Buffer = {Ops.LOAD, Ops.STORE, Ops.CONST, Ops.DEFINE_VAR}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # These can change the dtype to bool
  Comparison = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIPROCAL, Ops.LOG2, Ops.EXP2, Ops.IDIV, Ops.POW}

  All = set(Ops)

# **** backend specific ops ****

# NOTE: X86Ops with i suffix are variants that take an immediate, m suffix are variants that can write to memory instead of read from
class X86Ops(FastEnum):
  # register, not an instruction
  DEFINE_REG = auto()
  # const
  IMM = auto()
  # index
  LEA = auto()
  # register / memory / immediate moves
  MOV = auto(); MOVm = auto(); MOVi = auto() # noqa: E702
  VMOVSS = auto(); VMOVSD = auto(); VMOVUPS = auto() # noqa: E702
  VMOVSSm = auto(); VMOVSDm = auto(); VMOVUPSm = auto() # noqa: E702
  # casts
  MOVZX = auto(); MOVSX = auto(); MOVSXD = auto() # noqa: E702
  VPMOVZXBW = auto(); VPMOVZXBD = auto(); VPMOVZXBQ = auto() # noqa: E702
  VPMOVZXWD = auto(); VPMOVZXWQ = auto(); VPMOVZXDQ = auto() # noqa: E702
  VPMOVSXBW = auto(); VPMOVSXBD = auto(); VPMOVSXBQ = auto() # noqa: E702
  VPMOVSXWD = auto(); VPMOVSXWQ = auto(); VPMOVSXDQ = auto() # noqa: E702
  VCVTDQ2PS = auto(); VCVTDQ2PD = auto(); VCVTTPS2DQ = auto(); VCVTTPD2DQ = auto() # noqa: E702
  VCVTPH2PS = auto(); VCVTPS2PH = auto(); VCVTPS2PD = auto(); VCVTPD2PS = auto() # noqa: E702
  VCVTSS2SD = auto(); VCVTSD2SS = auto(); VCVTSI2SS = auto(); VCVTSI2SD = auto() # noqa: E702
  VCVTTSS2SI = auto(); VCVTTSD2SI = auto() # noqa: E702
  # bitcasts
  VMOVD = auto(); VMOVQ = auto(); VMOVDm = auto(); VMOVQm = auto() # noqa: E702
  # comparisons
  VCMPSS = auto(); VCMPSD = auto(); VCMPPS = auto(); VCMPPD = auto() # noqa: E702
  VPCMPGTB = auto(); VPCMPGTW = auto(); VPCMPGTD = auto(); VPCMPGTQ = auto() # noqa: E702
  VPCMPEQB = auto(); VPCMPEQW = auto(); VPCMPEQD = auto(); VPCMPEQQ = auto() # noqa: E702
  SETNE = auto(); SETE = auto(); SETL = auto(); SETB = auto() # noqa: E702
  # where
  CMOVNE = auto(); CMOVE = auto(); CMOVL = auto(); CMOVB = auto() # noqa: E702
  VPBLENDVB = auto(); VBLENDVPS = auto(); VBLENDVPD = auto() # noqa: E702
  # jumps
  JNE = auto(); JE = auto(); JL = auto(); JB = auto() # noqa: E702
  # vectorize / gep
  VSHUFPS = auto(); VINSERTPS = auto() # noqa: E702
  VPEXTRB = auto(); VPEXTRW = auto(); VPEXTRD = auto(); VPEXTRQ = auto() # noqa: E702
  VPINSRB = auto(); VPINSRW = auto(); VPINSRD = auto(); VPINSRQ = auto() # noqa: E702
  VPBROADCASTB = auto(); VPBROADCASTW = auto(); VPBROADCASTD = auto(); VPBROADCASTQ = auto() # noqa: E702
  VBROADCASTSS = auto() # TODO: VBROADCASTSD is ymm only, add once they are supported
  # int division
  IDIV = auto()
  CBW = auto(); CWD = auto(); CDQ = auto(); CQO = auto() # noqa: E702
  # int binary
  ADD = auto(); ADDi = auto(); SUB = auto(); SUBi = auto(); IMUL = auto(); IMULi = auto() # noqa: E702
  AND = auto(); ANDi = auto(); XOR = auto(); XORi = auto(); OR = auto(); ORi = auto() # noqa: E702
  SHL = auto(); SHLi = auto(); SHR = auto(); SHRi = auto(); SAR = auto(); SARi = auto(); CMP = auto(); CMPi = auto() # noqa: E702
  # float unary (sometimes not unary)
  VROUNDSS = auto(); VROUNDSD = auto(); VROUNDPS = auto(); VROUNDPD = auto() # noqa: E702
  VSQRTSS = auto(); VSQRTSD = auto(); VSQRTPS = auto(); VSQRTPD = auto() # noqa: E702
  # float scalar / vector binary
  VADDSS = auto(); VADDSD = auto(); VADDPS = auto(); VADDPD = auto() # noqa: E702
  VSUBSS = auto(); VSUBSD = auto(); VSUBPS = auto(); VSUBPD = auto() # noqa: E702
  VMULSS = auto(); VMULSD = auto(); VMULPS = auto(); VMULPD = auto() # noqa: E702
  VDIVSS = auto(); VDIVSD = auto(); VDIVPS = auto(); VDIVPD = auto() # noqa: E702
  # int vector binary
  VPADDB = auto(); VPADDW = auto(); VPADDD = auto(); VPADDQ = auto() # noqa: E702
  VPSUBB = auto(); VPSUBW = auto(); VPSUBD = auto(); VPSUBQ = auto() # noqa: E702
  VPMULLW = auto(); VPMULLD = auto() # noqa: E702
  # packed bitwise TODO: might also want vandp cause of different execution ports
  VPAND = auto(); VPOR = auto(); VPXOR = auto() # noqa: E702
  # packed variable shifts
  VPSLLVD = auto(); VPSLLVQ = auto(); VPSRLVD = auto(); VPSRLVQ = auto(); VPSRAVD = auto() # noqa: E702
  # fused multiply add TODO: add other variants to fuse more loads
  VFMADD213SS = auto(); VFMADD213SD = auto(); VFMADD213PS = auto(); VFMADD213PD = auto() # noqa: E702
  # return
  RET = auto()

# TODO: add associative groupop to fuse more loads
class X86GroupOp:
  # X86Ops whose first src is also the destination
  TwoAddress1st = {X86Ops.ADD, X86Ops.ADDi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi, X86Ops.OR, X86Ops.ORi, X86Ops.IMUL,
                   X86Ops.SUB, X86Ops.SUBi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi,
                   X86Ops.VFMADD213SS, X86Ops.VFMADD213SD, X86Ops.VFMADD213PS, X86Ops.VFMADD213PD}

  # X86Ops whose second src is also the destination
  TwoAddress2nd = {X86Ops.CMOVB, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVNE}

  # X86Ops whose first src can read from memory
  ReadMem1st = {X86Ops.MOV, X86Ops.VMOVSS, X86Ops.VMOVSD, X86Ops.VMOVUPS, X86Ops.MOVZX, X86Ops.MOVSX, X86Ops.MOVSXD, X86Ops.VMOVD, X86Ops.VMOVQ,
                X86Ops.VPMOVZXBW, X86Ops.VPMOVZXBD, X86Ops.VPMOVZXBQ, X86Ops.VPMOVZXWD, X86Ops.VPMOVZXWQ, X86Ops.VPMOVZXDQ,
                X86Ops.VPMOVSXBW, X86Ops.VPMOVSXBD, X86Ops.VPMOVSXBQ, X86Ops.VPMOVSXWD, X86Ops.VPMOVSXWQ, X86Ops.VPMOVSXDQ,
                X86Ops.VCVTDQ2PS, X86Ops.VCVTDQ2PD, X86Ops.VCVTTPS2DQ, X86Ops.VCVTTPD2DQ, X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI,
                X86Ops.VCVTPH2PS, X86Ops.VCVTPS2PD, X86Ops.VCVTPD2PS, X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB,
                X86Ops.VROUNDPS, X86Ops.VROUNDPD, X86Ops.VSQRTPS, X86Ops.VSQRTPD, X86Ops.CMPi, X86Ops.IMULi, X86Ops.IDIV, X86Ops.LEA,
                X86Ops.VPBROADCASTB, X86Ops.VPBROADCASTW, X86Ops.VPBROADCASTD, X86Ops.VPBROADCASTQ, X86Ops.VBROADCASTSS}

  # X86Ops whose second src can read from memory NOTE: some of these are TwoAddress1st so the second src is actually the first
  ReadMem2nd = {X86Ops.ADD, X86Ops.SUB, X86Ops.AND, X86Ops.OR, X86Ops.XOR, X86Ops.SHL, X86Ops.SHR, X86Ops.SAR, X86Ops.IMUL, X86Ops.CMP,
                X86Ops.VADDSS, X86Ops.VADDSD, X86Ops.VADDPS, X86Ops.VADDPD, X86Ops.VSUBSS, X86Ops.VSUBSD, X86Ops.VSUBPS, X86Ops.VSUBPD,
                X86Ops.VMULSS, X86Ops.VMULSD, X86Ops.VMULPS, X86Ops.VMULPD, X86Ops.VDIVSS, X86Ops.VDIVSD, X86Ops.VDIVPS, X86Ops.VDIVPD,
                X86Ops.VPADDB, X86Ops.VPADDW, X86Ops.VPADDD, X86Ops.VPADDQ, X86Ops.VPSUBB, X86Ops.VPSUBW, X86Ops.VPSUBD, X86Ops.VPSUBQ,
                X86Ops.VPCMPEQB, X86Ops.VPCMPEQW, X86Ops.VPCMPEQD, X86Ops.VPCMPEQQ, X86Ops.VPBLENDVB, X86Ops.VBLENDVPS, X86Ops.VBLENDVPD,
                X86Ops.VPCMPGTB, X86Ops.VPCMPGTW, X86Ops.VPCMPGTD, X86Ops.VPCMPGTQ, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VCMPPS, X86Ops.VCMPPD,
                X86Ops.VPMULLW, X86Ops.VPMULLD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VSQRTSS, X86Ops.VSQRTSD, X86Ops.VSHUFPS, X86Ops.VINSERTPS,
                X86Ops.VPINSRB, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.VPINSRQ, X86Ops.VPAND, X86Ops.VPOR, X86Ops.VPXOR, X86Ops.VPSLLVD,
                X86Ops.VPSLLVQ, X86Ops.VPSRLVD, X86Ops.VPSRLVQ, X86Ops.VPSRAVD, X86Ops.VCVTSI2SS, X86Ops.VCVTSI2SD, X86Ops.VCVTSS2SD, X86Ops.VCVTSD2SS}

  # X86Ops whose third src can read from memory NOTE: these are TwoAddress1st so the third src is actually the second
  ReadMem3rd = {X86Ops.VFMADD213SS, X86Ops.VFMADD213SD, X86Ops.VFMADD213PS, X86Ops.VFMADD213PD}

  # X86Ops that can write to memory
  WriteMem = {X86Ops.MOVm, X86Ops.MOVi, X86Ops.VMOVSSm, X86Ops.VMOVSDm, X86Ops.VMOVUPSm, X86Ops.VMOVDm, X86Ops.VMOVQm,
              X86Ops.ADDi, X86Ops.SUBi, X86Ops.ANDi, X86Ops.ORi, X86Ops.XORi, X86Ops.SHLi, X86Ops.SHRi, X86Ops.SARi, X86Ops.SETNE,
              X86Ops.SETE, X86Ops.SETL, X86Ops.SETB, X86Ops.VCVTPS2PH, X86Ops.VPEXTRB, X86Ops.VPEXTRW, X86Ops.VPEXTRD, X86Ops.VPEXTRQ}

  # X86Ops that read flags
  ReadFlags = {X86Ops.CMOVB, X86Ops.CMOVL, X86Ops.CMOVE, X86Ops.CMOVNE, X86Ops.SETB, X86Ops.SETL, X86Ops.SETE, X86Ops.SETNE, X86Ops.JB, X86Ops.JL,
               X86Ops.JE, X86Ops.JNE}

  # X86Ops that write flags or can modify flags to undefined values
  WriteFlags = {X86Ops.CMP, X86Ops.CMPi, X86Ops.ADD, X86Ops.ADDi, X86Ops.SUB, X86Ops.SUBi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi,
                X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi, X86Ops.IMUL, X86Ops.IMULi, X86Ops.IDIV, X86Ops.OR, X86Ops.ORi}

  All = set(X86Ops)
