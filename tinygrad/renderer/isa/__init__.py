# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
from tinygrad.uop.ops import Ops, auto

# ***** X86 *****

# NOTE: mypy doesn't allow extending enums even with our wrapper, it also doesn't allow overriding i.e. Ops.ADD to X86Ops.ADD
# we ignore it in both cases
class X86Ops(Ops): # type: ignore[misc]
  # NOTE: X86Ops with i suffix are variants that take an immediate, m suffix are variants that can write to memory instead of read from
  # register, not an instruction. FRAME_INDEX is used when the function arg is on the stack and is rewritten to IMM when stack size is known
  DEFINE_REG = auto(); FRAME_INDEX = auto() # type: ignore[misc]
  # const
  IMM = auto()
  # index
  LEA = auto()
  # register / memory / immediate moves
  MOV = auto(); MOVm = auto(); MOVi = auto(); MOVABS = auto()
  VMOVSS = auto(); VMOVSD = auto(); VMOVUPS = auto()
  VMOVSSm = auto(); VMOVSDm = auto(); VMOVUPSm = auto()
  # casts
  MOVZX = auto(); MOVSX = auto(); MOVSXD = auto()
  VPMOVZXBW = auto(); VPMOVZXBD = auto(); VPMOVZXBQ = auto()
  VPMOVZXWD = auto(); VPMOVZXWQ = auto(); VPMOVZXDQ = auto()
  VPMOVSXBW = auto(); VPMOVSXBD = auto(); VPMOVSXBQ = auto()
  VPMOVSXWD = auto(); VPMOVSXWQ = auto(); VPMOVSXDQ = auto()
  VCVTDQ2PS = auto(); VCVTDQ2PD = auto(); VCVTTPS2DQ = auto(); VCVTTPD2DQ = auto()
  VCVTPH2PS = auto(); VCVTPS2PH = auto(); VCVTPS2PD = auto(); VCVTPD2PS = auto()
  VCVTSS2SD = auto(); VCVTSD2SS = auto(); VCVTSI2SS = auto(); VCVTSI2SD = auto()
  VCVTTSS2SI = auto(); VCVTTSD2SI = auto()
  # bitcasts
  VMOVD = auto(); VMOVQ = auto(); VMOVDm = auto(); VMOVQm = auto()
  # comparisons
  VUCOMISS = auto(); VUCOMISD = auto()
  VCMPSS = auto(); VCMPSD = auto(); VCMPPS = auto(); VCMPPD = auto()
  VPCMPGTB = auto(); VPCMPGTW = auto(); VPCMPGTD = auto(); VPCMPGTQ = auto()
  VPCMPEQB = auto(); VPCMPEQW = auto(); VPCMPEQD = auto(); VPCMPEQQ = auto()
  SETNE = auto(); SETE = auto(); SETL = auto(); SETB = auto()
  # where
  CMOVNE = auto(); CMOVE = auto(); CMOVL = auto(); CMOVB = auto()
  VPBLENDVB = auto(); VBLENDVPS = auto(); VBLENDVPD = auto()
  # jumps
  JNE = auto(); JE = auto(); JL = auto(); JB = auto()
  # vectorize / gep
  VSHUFPS = auto(); VINSERTPS = auto()
  VPEXTRB = auto(); VPEXTRW = auto(); VPEXTRD = auto(); VPEXTRQ = auto()
  VPINSRB = auto(); VPINSRW = auto(); VPINSRD = auto(); VPINSRQ = auto()
  VPBROADCASTB = auto(); VPBROADCASTW = auto(); VPBROADCASTD = auto(); VPBROADCASTQ = auto()
  VBROADCASTSS = auto() # TODO: VBROADCASTSD is ymm only, add once they are supported
  # int division
  IDIV = auto(); DIV = auto() # type: ignore[misc]
  CBW = auto(); CWD = auto(); CDQ = auto(); CQO = auto()
  # int binary
  ADD = auto(); ADDi = auto(); SUB = auto(); SUBi = auto(); IMUL = auto(); IMULi = auto() # type: ignore[misc]
  AND = auto(); ANDi = auto(); XOR = auto(); XORi = auto(); OR = auto(); ORi = auto() # type: ignore[misc]
  SHL = auto(); SHLi = auto(); SHR = auto(); SHRi = auto(); SAR = auto(); SARi = auto(); CMP = auto(); CMPi = auto() # type: ignore[misc]
  # float unary (sometimes not unary)
  VROUNDSS = auto(); VROUNDSD = auto(); VROUNDPS = auto(); VROUNDPD = auto()
  VSQRTSS = auto(); VSQRTSD = auto(); VSQRTPS = auto(); VSQRTPD = auto()
  # float scalar / vector binary
  VADDSS = auto(); VADDSD = auto(); VADDPS = auto(); VADDPD = auto()
  VSUBSS = auto(); VSUBSD = auto(); VSUBPS = auto(); VSUBPD = auto()
  VMULSS = auto(); VMULSD = auto(); VMULPS = auto(); VMULPD = auto()
  VDIVSS = auto(); VDIVSD = auto(); VDIVPS = auto(); VDIVPD = auto()
  VMAXSS = auto(); VMAXSD = auto(); VMAXPS = auto(); VMAXPD = auto()
  VMINSS = auto(); VMINSD = auto(); VMINPS = auto(); VMINPD = auto()
  # int vector binary
  VPADDB = auto(); VPADDW = auto(); VPADDD = auto(); VPADDQ = auto()
  VPSUBB = auto(); VPSUBW = auto(); VPSUBD = auto(); VPSUBQ = auto()
  VPMULLW = auto(); VPMULLD = auto()
  # packed bitwise TODO: might also want vandp cause of different execution ports
  VPAND = auto(); VPOR = auto(); VPXOR = auto()
  # packed variable shifts
  VPSLLVD = auto(); VPSLLVQ = auto(); VPSRLVD = auto(); VPSRLVQ = auto(); VPSRAVD = auto()
  # fused multiply add TODO: add other variants to fuse more loads
  VFMADD213SS = auto(); VFMADD213SD = auto(); VFMADD213PS = auto(); VFMADD213PD = auto()
  # return
  RET = auto()

# TODO: add commutative groupop to fuse more loads
class X86GroupOp:
  # X86Ops whose first src is also the destination
  TwoAddress1st = {X86Ops.ADD, X86Ops.ADDi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi, X86Ops.OR, X86Ops.ORi, X86Ops.IMUL,
                   X86Ops.SUB, X86Ops.SUBi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi,
                   X86Ops.VFMADD213SS, X86Ops.VFMADD213SD, X86Ops.VFMADD213PS, X86Ops.VFMADD213PD,
                   X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB}

  # X86Ops whose first src can read from memory
  ReadMem1st = {X86Ops.MOV, X86Ops.VMOVSS, X86Ops.VMOVSD, X86Ops.VMOVUPS, X86Ops.MOVZX, X86Ops.MOVSX, X86Ops.MOVSXD, X86Ops.VMOVD, X86Ops.VMOVQ,
                X86Ops.VPMOVZXBW, X86Ops.VPMOVZXBD, X86Ops.VPMOVZXBQ, X86Ops.VPMOVZXWD, X86Ops.VPMOVZXWQ, X86Ops.VPMOVZXDQ,
                X86Ops.VPMOVSXBW, X86Ops.VPMOVSXBD, X86Ops.VPMOVSXBQ, X86Ops.VPMOVSXWD, X86Ops.VPMOVSXWQ, X86Ops.VPMOVSXDQ,
                X86Ops.VCVTDQ2PS, X86Ops.VCVTDQ2PD, X86Ops.VCVTTPS2DQ, X86Ops.VCVTTPD2DQ, X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI,
                X86Ops.VCVTPH2PS, X86Ops.VCVTPS2PD, X86Ops.VCVTPD2PS, X86Ops.VROUNDPS, X86Ops.VROUNDPD, X86Ops.VSQRTPS, X86Ops.VSQRTPD,
                X86Ops.VPBROADCASTB, X86Ops.VPBROADCASTW, X86Ops.VPBROADCASTD, X86Ops.VPBROADCASTQ, X86Ops.VBROADCASTSS,
                X86Ops.CMPi, X86Ops.IMULi, X86Ops.IDIV, X86Ops.DIV, X86Ops.LEA}

  # X86Ops whose second src can read from memory NOTE: some of these are TwoAddress1st so the second src is actually the first
  ReadMem2nd = {X86Ops.ADD, X86Ops.SUB, X86Ops.AND, X86Ops.OR, X86Ops.XOR, X86Ops.SHL, X86Ops.SHR, X86Ops.SAR, X86Ops.IMUL, X86Ops.CMP,
                X86Ops.VADDSS, X86Ops.VADDSD, X86Ops.VADDPS, X86Ops.VADDPD, X86Ops.VSUBSS, X86Ops.VSUBSD, X86Ops.VSUBPS, X86Ops.VSUBPD,
                X86Ops.VMULSS, X86Ops.VMULSD, X86Ops.VMULPS, X86Ops.VMULPD, X86Ops.VDIVSS, X86Ops.VDIVSD, X86Ops.VDIVPS, X86Ops.VDIVPD,
                X86Ops.VPADDB, X86Ops.VPADDW, X86Ops.VPADDD, X86Ops.VPADDQ, X86Ops.VPSUBB, X86Ops.VPSUBW, X86Ops.VPSUBD, X86Ops.VPSUBQ,
                X86Ops.VPCMPEQB, X86Ops.VPCMPEQW, X86Ops.VPCMPEQD, X86Ops.VPCMPEQQ, X86Ops.VPBLENDVB, X86Ops.VBLENDVPS, X86Ops.VBLENDVPD,
                X86Ops.VPCMPGTB, X86Ops.VPCMPGTW, X86Ops.VPCMPGTD, X86Ops.VPCMPGTQ, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VCMPPS, X86Ops.VCMPPD,
                X86Ops.VPMULLW, X86Ops.VPMULLD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VSQRTSS, X86Ops.VSQRTSD, X86Ops.VSHUFPS, X86Ops.VINSERTPS,
                X86Ops.VPINSRB, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.VPINSRQ, X86Ops.VPAND, X86Ops.VPOR, X86Ops.VPXOR, X86Ops.VPSLLVD,
                X86Ops.VPSLLVQ, X86Ops.VPSRLVD, X86Ops.VPSRLVQ, X86Ops.VPSRAVD, X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB,
                X86Ops.VMAXSS, X86Ops.VMAXSD, X86Ops.VMAXPS, X86Ops.VMAXPD, X86Ops.VMINSS, X86Ops.VMINSD, X86Ops.VMINPS, X86Ops.VMINPD,
                X86Ops.VCVTSI2SS, X86Ops.VCVTSI2SD, X86Ops.VCVTSS2SD, X86Ops.VCVTSD2SS, X86Ops.VUCOMISS, X86Ops.VUCOMISD}

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
  WriteFlags = {X86Ops.CMP, X86Ops.CMPi, X86Ops.ADD, X86Ops.ADDi, X86Ops.SUB, X86Ops.SUBi, X86Ops.IMUL, X86Ops.IMULi, X86Ops.IDIV, X86Ops.DIV,
                X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi,
                X86Ops.OR, X86Ops.ORi, X86Ops.VUCOMISS, X86Ops.VUCOMISD}

  All = set(X86Ops)
