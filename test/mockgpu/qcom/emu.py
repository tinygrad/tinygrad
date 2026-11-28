import functools, hashlib, itertools, math, pathlib
from dataclasses import dataclass
from typing import Callable
import numpy as np
from tinygrad.helpers import Context, getbits, diskcache_get, diskcache_put
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import DType, dtypes, AddrSpace
from tinygrad.device import Device
from tinygrad.renderer import TinyELF
from tinygrad.codegen import to_program

NREGS, A0, P0, INVALID_REG, MAX_LOCAL = 64 * 4, 61 * 4, 62 * 4, 63 * 4, 1024
NROWS, NCONSTS, MAX_IMGS, IMG_META, MAX_CALLS, N_PARAMS, PC_NONE = NREGS + 3, 4096, 1024, 5, 16, 14, 0x7FFFFFFF
HALF_TYPES = (0, 2, 4, 6, 7) # f16, u16, s16, u8, s8
FLUT = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0]
REG_INIT = 0

def sext(x:int, n:int) -> int: return x - (1 << n) if x & (1 << (n - 1)) else x

class Field:
  def __init__(self, lo:int, hi:int|None=None): self.lo, self.hi = lo, lo if hi is None else hi
  def __get__(self, i, owner=None) -> int: return getbits(i.word, self.lo, self.hi)

@dataclass(frozen=True)
class Src: kind:str; val:int; half:bool = False; absneg:int = 0; r:bool = False # noqa: E702

def multisrc(x:int, half:bool, r:int) -> Src:
  absneg = getbits(x, 14, 15)
  if getbits(x, 11, 13) == 0b000: return Src("r", getbits(x, 0, 7), half, absneg, bool(r))
  if getbits(x, 10, 13) == 0b0011 and not half: return Src("rel", sext(getbits(x, 0, 9), 10), half, absneg)
  if getbits(x, 11, 12) == 0b10: return Src("c", getbits(x, 0, 10), half, absneg, bool(r))
  if getbits(x, 11, 13) == 0b100: return Src("i", sext(getbits(x, 0, 10), 11), half, absneg)
  if getbits(x, 11, 13) == 0b101:
    v = FLUT[getbits(x, 0, 9)]
    return Src("i", int(np.float16(v).view(np.uint16)) if getbits(x, 10, 10) else int(np.float32(v).view(np.uint32)), half, absneg)
  raise NotImplementedError(f"multisrc encoding {x:#x}")

def cat3src(x:int, half:bool, immed:int, r:int, neg:int) -> Src:
  if getbits(x, 11, 12) == 0b00: return Src("r", getbits(x, 0, 7), half, neg, bool(r))
  if getbits(x, 12, 12): return Src("i", getbits(x, 0, 11), half, neg) if immed else Src("c", getbits(x, 0, 10), half, neg, bool(r))
  raise NotImplementedError(f"cat3 source encoding {x:#x}")

class Inst:
  cat, opc, repeat = Field(61, 63), Field(55, 58), 0
  srcs:list[Src]
  def __init__(self, pc:int, word:int):
    self.pc, self.word = pc, word
    self.op, self.iterations = self.cat << 7 | self.opc, self.repeat + 1
  def error(self): return NotImplementedError(f"pc {self.pc}: {mesa.opc_t.get(self.op, self.op)} is not emulated {self.word:#x}")

class Cat0(Inst):
  immed, brtype, inv2, comp2 = Field(0, 31), Field(37, 39), Field(45), Field(46, 47)
  opc_hi, inv1, comp1 = Field(49), Field(52), Field(53, 54)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.op = mesa.OPC_BR + self.brtype if (self.opc_hi, self.opc) == (0, 1) else self.opc_hi << 4 | self.opc
    self.target = pc + sext(self.immed, 32)
    if self.op not in CTRL_IDX: raise self.error()

class Cat1(Inst):
  dst, repeat, src_r, dst_type, dst_rel = Field(32, 39), Field(40, 41), Field(43), Field(46, 48), Field(49)
  src_type, src_mode, round, multi = Field(50, 52), Field(53, 54), Field(55, 56), Field(57, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    half = self.src_type in HALF_TYPES
    if (self.multi, self.dst_rel) == (0b10, 0):
      self.op, self.iterations = mesa.OPC_SWZ + getbits(word, 40, 41), 1
      if self.op != mesa.OPC_SWZ: raise self.error()
      self.srcs, self.dsts = [Src("r", getbits(word, 0, 7), half), Src("r", getbits(word, 8, 15), half)], [self.dst, getbits(word, 16, 23)]
      return
    self.op = mesa.OPC_MOV
    if self.dst_rel or self.multi or self.src_mode == 0b11 or self.round > 1: raise self.error()
    if self.src_mode == 0b00:
      if getbits(word, 11, 11):
        if not getbits(word, 10, 10) or half: raise self.error()
        self.srcs = [Src("rel", sext(getbits(word, 0, 9), 10))]
      else: self.srcs = [Src("r", getbits(word, 0, 7), half, r=bool(self.src_r))]
    elif self.src_mode == 0b01: self.srcs = [Src("c", getbits(word, 0, 10), half, r=bool(self.src_r))]
    else: self.srcs = [Src("i", getbits(word, 0, 31), half)]

class Cat2(Inst):
  dst, repeat, sat, src1_r, dst_conv, ei = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(47)
  cond, src2_r, full, opc = Field(48, 50), Field(51), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in ALU_OPS or (self.op in CMPS and self.cond >= 6): raise self.error()
    if self.ei or (self.sat and self.op not in CMPS and self.op not in FLOAT_OPS): raise self.error()
    self.dst_half = (not self.full) ^ bool(self.dst_conv)
    self.srcs = [multisrc(getbits(word, 0, 15), not self.full, self.src1_r)]
    if self.op not in CAT2_1SRC: self.srcs.append(multisrc(getbits(word, 16, 31), not self.full, self.src2_r))

class Cat3(Inst):
  src1, alt, src1_neg, src2_r = Field(0, 12), Field(13), Field(14), Field(15)
  src3, src3_r, src2_neg, src3_neg = Field(16, 28), Field(29), Field(30), Field(31)
  dst, repeat, sat_full, src1_r, dst_conv, src2 = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(47, 54)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.op += 8 * self.alt
    if self.op not in ALU_OPS: raise self.error()
    half = not self.sat_full if self.alt else self.op in CAT3_HALF
    self.sat, self.dst_half = 0 if self.alt else self.sat_full, half ^ bool(self.dst_conv)
    if self.op == mesa.OPC_MAD_U16: self.dst_half = not self.dst_conv
    if self.sat and self.op not in FLOAT_OPS: raise self.error()
    self.srcs = [cat3src(self.src1, half, self.alt, self.src1_r, self.src1_neg), Src("r", self.src2, half, self.src2_neg, bool(self.src2_r)),
                 cat3src(self.src3, half, self.alt, self.src3_r, self.src3_neg)]

class Cat4(Inst):
  dst, repeat, sat, src_r, dst_conv, full, opc = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in SFU_IDX: raise self.error()
    self.dst_half, self.srcs = (not self.full) ^ bool(self.dst_conv), [multisrc(getbits(word, 0, 15), not self.full, self.src_r)]

class Cat5(Inst):
  full, src1, samp, tex, dst = Field(0), Field(1, 8), Field(21, 24), Field(25, 31), Field(32, 39)
  src3, desc_mode, wrmask, type, flags, opc = Field(21, 28), Field(29, 31), Field(40, 43), Field(44, 46), Field(48, 52), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.s2en = self.flags == 0b01000 and self.desc_mode == 0
    if self.op != mesa.OPC_ISAM or (self.flags and not self.s2en) or not self.full: raise self.error()

class Load(Inst):
  off, addr, size, dst, type, opc = Field(1, 13), Field(14, 21), Field(24, 31), Field(32, 39), Field(49, 51), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.offset = sext(self.off, 13)

class Store(Inst):
  val, off_hi, size, off_lo, dst_off = Field(1, 8), Field(9, 13), Field(24, 31), Field(32, 39), Field(40)
  addr, type, opc = Field(41, 48), Field(49, 51), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.offset = sext(self.off_hi << 8 | self.off_lo, 13) if self.dst_off else 0

def reg_offset(i:Load|Store, word:int, src2:int) -> tuple[int, int, int]:
  if getbits(word, 11, 11): raise i.error()
  i.offset = 0
  return src2, getbits(word, 12, 13), getbits(word, 9, 10)

class Ldg(Load):
  size = Field(24, 26)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if not getbits(word, 0, 0) or getbits(word, 52, 53): raise self.error()
    self.reg_off = reg_offset(self, word, getbits(word, 1, 8)) if getbits(word, 22, 22) else None

class Stg(Store):
  size = Field(24, 26)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if not getbits(word, 23, 23) or getbits(word, 53, 53): raise self.error()
    self.reg_off = reg_offset(self, word, self.off_lo) if getbits(word, 52, 52) else None

class Ibo(Inst):
  mode, bindless, d_minus_one, typed, type_size_minus_one, opc = Field(6, 7), Field(8), Field(9, 10), Field(11), Field(12, 13), Field(14, 19)
  has_offset, coord, val, ssbo, type = Field(23), Field(24, 31), Field(32, 39), Field(41, 48), Field(49, 51)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.ncomp = self.type_size_minus_one + 1
    if self.op != mesa.OPC_STIB or self.mode or self.bindless or self.has_offset or not self.typed or self.d_minus_one != 1:
      raise self.error()

class Cat7(Inst):
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in (mesa.OPC_BAR, mesa.OPC_FENCE): raise self.error()

CMPS = {mesa.OPC_CMPS_F, mesa.OPC_CMPS_U, mesa.OPC_CMPS_S, mesa.OPC_CMPV_F, mesa.OPC_CMPV_U, mesa.OPC_CMPV_S}
CMPV = {mesa.OPC_CMPV_F, mesa.OPC_CMPV_U, mesa.OPC_CMPV_S}
CAT2_1SRC = {mesa.OPC_SIGN_F, mesa.OPC_ABSNEG_F, mesa.OPC_FLOOR_F, mesa.OPC_TRUNC_F, mesa.OPC_ABSNEG_S, mesa.OPC_NOT_B, mesa.OPC_CLZ_B}
BITWISE = {mesa.OPC_AND_B, mesa.OPC_OR_B, mesa.OPC_XOR_B, mesa.OPC_NOT_B}
CAT3_HALF = {mesa.OPC_MAD_F16, mesa.OPC_SEL_B16, mesa.OPC_SEL_S16}
CATS = {0: Cat0, 1: Cat1, 2: Cat2, 3: Cat3, 4: Cat4, 5: Cat5, 7: Cat7}
CAT6 = {mesa.OPC_LDG: Ldg, mesa.OPC_STG: Stg, mesa.OPC_LDL: Load, mesa.OPC_LDP: Load, mesa.OPC_STL: Store, mesa.OPC_STP: Store}

def decode_inst(pc:int, word:int) -> Inst:
  cat = getbits(word, 61, 63)
  if cat == 6 and getbits(word, 52, 53) == 0b10 and getbits(word, 20, 22) == 0b110: return Ibo(pc, word)
  cls = CAT6.get(6 << 7 | getbits(word, 54, 58)) if cat == 6 else CATS.get(cat)
  if cls is None: raise NotImplementedError(f"pc {pc}: cat{cat} is not emulated {word:#x}")
  return cls(pc, word)

# VM bytecode: REC_LEN uint32 per instruction; opcode IDs are indices into the category tables below
REC_LEN = 11
K_UNDEF, K_FMAD32, K_FMAD16, K_FMISC, K_IALU32, K_IMISC, K_SFU, K_MOV, K_MEM, K_TEX, K_CTRL = range(11)
MOV_SAME, MOV_SWZ, MOV_COV, TEX_ISAM, TEX_STIB = 0, 1, 2, 0, 1
SRC_KIND = {"r": 1, "c": 2, "rel": 3, "i": 4}
ERRS = ["", "undef", "relative const out of range", "out of bounds local access", "out of bounds private access", "texture is not bound",
        "IBO is not bound", "call stack overflow", "ret without call", "every thread is waiting at a bar"]
FMAD_OPS = {mesa.OPC_MAD_F32, mesa.OPC_MAD_F16, mesa.OPC_ADD_F, mesa.OPC_MUL_F}
SIGNED_IOPS = {mesa.OPC_ADD_S, mesa.OPC_SUB_S, mesa.OPC_MIN_S, mesa.OPC_MAX_S, mesa.OPC_ABSNEG_S,
               mesa.OPC_ASHR_B, mesa.OPC_SEL_S32, mesa.OPC_SEL_S16, mesa.OPC_SAD_S32, mesa.OPC_CMPS_S, mesa.OPC_CMPV_S}
def _idx(groups:tuple[tuple[int, ...], ...]) -> dict[int, int]: return {op: i for i, g in enumerate(groups) for op in g}
FMISC_IDX = _idx(((mesa.OPC_MIN_F,), (mesa.OPC_MAX_F,), (mesa.OPC_SIGN_F,), (mesa.OPC_ABSNEG_F,), (mesa.OPC_FLOOR_F,),
                  (mesa.OPC_TRUNC_F,), (mesa.OPC_SEL_F32,), (mesa.OPC_CMPS_F, mesa.OPC_CMPV_F)))
# K_IALU32 runs the 32-bit ops before MIN_U without absneg, K_IMISC runs all of them
INT_IDX = _idx((
  (mesa.OPC_ADD_U,), (mesa.OPC_ADD_S,), (mesa.OPC_SUB_U,), (mesa.OPC_SUB_S,),
  (mesa.OPC_AND_B,), (mesa.OPC_OR_B,), (mesa.OPC_XOR_B,), (mesa.OPC_NOT_B,),
  (mesa.OPC_SHL_B,), (mesa.OPC_SHR_B,), (mesa.OPC_ASHR_B,), (mesa.OPC_GETBIT_B,),
  (mesa.OPC_MUL_S24,), (mesa.OPC_MUL_U24,), (mesa.OPC_MULL_U,),
  (mesa.OPC_MADSH_M16, mesa.OPC_MADSH_U16), (mesa.OPC_MAD_U16,), (mesa.OPC_MAD_S24,),
  (mesa.OPC_SEL_B32, mesa.OPC_SEL_B16), (mesa.OPC_SEL_S32, mesa.OPC_SEL_S16),
  (mesa.OPC_SHRM,), (mesa.OPC_SHRG,), (mesa.OPC_SHLG,), (mesa.OPC_SHLM,), (mesa.OPC_ANDG,),
  (mesa.OPC_MIN_U,), (mesa.OPC_MIN_S,), (mesa.OPC_MAX_U,), (mesa.OPC_MAX_S,), (mesa.OPC_ABSNEG_S,),
  (mesa.OPC_CLZ_B,), (mesa.OPC_SAD_S32,),
  (mesa.OPC_CMPS_U, mesa.OPC_CMPV_U), (mesa.OPC_CMPS_S, mesa.OPC_CMPV_S)))
SFU_IDX = _idx(((mesa.OPC_RCP,), (mesa.OPC_RSQ,), (mesa.OPC_LOG2,), (mesa.OPC_EXP2,), (mesa.OPC_SIN,), (mesa.OPC_SQRT,)))
MEM_IDX = _idx(((mesa.OPC_LDG,), (mesa.OPC_STG,), (mesa.OPC_LDL,), (mesa.OPC_STL,), (mesa.OPC_LDP,), (mesa.OPC_STP,)))
CTRL_IDX = _idx(((mesa.OPC_NOP, mesa.OPC_PREDE, mesa.OPC_FENCE), (mesa.OPC_END,), (mesa.OPC_JUMP,), (mesa.OPC_BR,), (mesa.OPC_BRAO,),
                 (mesa.OPC_BRAA,), (mesa.OPC_PREDT,), (mesa.OPC_PREDF,), (mesa.OPC_CALL,), (mesa.OPC_RET,), (mesa.OPC_BAR,)))
FLOAT_OPS, ALU_OPS = FMAD_OPS | FMISC_IDX.keys(), FMAD_OPS | FMISC_IDX.keys() | INT_IDX.keys()
N_IALU32 = INT_IDX[mesa.OPC_MIN_U]

def _rec(*xs:int) -> tuple[int, ...]: return xs + (0,) * (REC_LEN - len(xs))
def pack_src(s:Src) -> tuple[int, int]:
  return SRC_KIND[s.kind] | (int(s.half) << 3) | (s.absneg << 4) | (int(s.r) << 6), s.val & 0xFFFFFFFF
def _pack3(srcs:list[Src]) -> tuple[int, ...]:
  ms, vs = zip(*(pack_src(s) for s in srcs)) if srcs else ((), ())
  return (*ms, *(0,) * (3 - len(ms)), *vs, *(0,) * (3 - len(vs)))

def encode_inst(i:Inst, pred_targets:dict[int, int]) -> tuple[int, ...]:
  if isinstance(i, Cat0):
    if i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF):
      return _rec(K_CTRL, CTRL_IDX[i.op], pred_targets[i.pc]) if i.pc in pred_targets else _rec(K_UNDEF)
    return _rec(K_CTRL, CTRL_IDX[i.op], i.target, i.comp1, i.inv1 | (i.inv2 << 1) | (i.comp2 << 4))
  if isinstance(i, Cat7): return _rec(K_CTRL, CTRL_IDX[i.op])
  if isinstance(i, Cat1):
    if i.op == mesa.OPC_SWZ:
      (m0, v0), (m1, v1) = pack_src(i.srcs[0]), pack_src(i.srcs[1])
      return (K_MOV, MOV_SWZ, i.dst, 1, int(i.dst_type in HALF_TYPES), m0, m1, 0, v0, v1, i.dsts[1])
    flags = int(i.dst_type in HALF_TYPES) | (int(i.src_type in HALF_TYPES) << 1) | (int(i.round == 1) << 2) | (i.src_type << 4) | (i.dst_type << 8)
    sub = MOV_SAME if i.src_type == i.dst_type and i.src_type not in (mesa.TYPE_U8, mesa.TYPE_U8_32) else MOV_COV
    return _rec(K_MOV, sub, i.dst, i.iterations, flags, *_pack3(i.srcs))
  if isinstance(i, (Cat2, Cat3)):
    src_half = i.srcs[0].half
    if i.op in FMAD_OPS:
      srcs = [i.srcs[0], Src("i", 0x3C00 if src_half else 0x3F800000, src_half), i.srcs[1]] if i.op == mesa.OPC_ADD_F else \
             [i.srcs[0], i.srcs[1], Src("i", 0x8000 if src_half else 0x80000000, src_half)] if i.op == mesa.OPC_MUL_F else i.srcs
      return _rec(K_FMAD16 if src_half else K_FMAD32, 0, i.dst, i.iterations, int(i.dst_half) | (int(i.sat) << 1), *_pack3(srcs))
    is_bw = i.op in BITWISE
    cond = i.cond if isinstance(i, Cat2) and i.op in CMPS else 0
    flags = int(i.dst_half) | (int(i.sat) << 1) | (int(i.op in CMPV) << 2) | (cond << 4) | \
            (int(is_bw) << 8) | (int(src_half) << 9) | (int(i.op in SIGNED_IOPS) << 10)
    if i.op in FMISC_IDX: return _rec(K_FMISC, FMISC_IDX[i.op], i.dst, i.iterations, flags, *_pack3(i.srcs))
    iop = INT_IDX[i.op]
    fast_i = not src_half and not i.dst_half and not i.sat and iop < N_IALU32 and (is_bw or all(s.absneg == 0 for s in i.srcs))
    return _rec(K_IALU32 if fast_i else K_IMISC, iop, i.dst, i.iterations, flags, *_pack3(i.srcs))
  if isinstance(i, Cat4):
    return _rec(K_SFU, SFU_IDX[i.op], i.dst, i.iterations, int(i.dst_half) | (int(i.sat) << 1) | (int(i.srcs[0].half) << 2), *_pack3(i.srcs))
  if isinstance(i, (Load, Store)):
    ro = getattr(i, "reg_off", None)
    r_src2, r_sh, r_off = ro if ro is not None else (0, 0, 0)
    flags = int(i.type in HALF_TYPES) | (int(ro is not None) << 2) | (i.type << 4) | (r_sh << 8) | (r_off << 12)
    esz = 1 if i.type in (mesa.TYPE_U8, mesa.TYPE_U8_32) else 2 if i.type in HALF_TYPES else 4
    return (K_MEM, MEM_IDX[i.op], i.dst if isinstance(i, Load) else i.val, i.size, flags, esz, 0, 0, i.addr, i.offset & 0xFFFFFFFF, r_src2)
  if isinstance(i, Cat5):
    return _rec(K_TEX, TEX_ISAM, i.dst, i.wrmask, int(i.s2en) | (i.type << 4), 0, 0, 0, i.src1, i.src3 if i.s2en else i.tex)
  assert isinstance(i, Ibo)
  return _rec(K_TEX, TEX_STIB, i.ssbo, i.ncomp, i.type << 4, 0, 0, 0, i.coord, i.val)

@functools.cache
def decode(image:bytes) -> np.ndarray:
  prog:list[Inst|None] = []
  for pc, word in enumerate(np.frombuffer(image, np.uint64).tolist()):
    try: prog.append(decode_inst(pc, word))
    except NotImplementedError: prog.append(None)
  preds = [i for i in prog if isinstance(i, Cat0) and i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF, mesa.OPC_PREDE)]
  pred_targets:dict[int, int] = {}
  for idx, i in enumerate(preds):
    if i.op == mesa.OPC_PREDE: continue
    nxt, after = preds[idx + 1] if idx + 1 < len(preds) else None, preds[idx + 2] if idx + 2 < len(preds) else None
    if nxt is not None and nxt.op == mesa.OPC_PREDE: pred_targets[i.pc] = nxt.pc
    elif nxt is not None and nxt.op != i.op and after is not None and after.op == mesa.OPC_PREDE: pred_targets[i.pc] = nxt.pc + 1
  return np.array([encode_inst(i, pred_targets) if i is not None else _rec(K_UNDEF) for i in prog], np.uint32).reshape(-1)

@dataclass(frozen=True)
class Image: addr:int; width:int; height:int; pitch:int; dtype:np.dtype # noqa: E702

@dataclass(frozen=True)
class Dispatch:
  image:bytes; consts:np.ndarray; local_size:tuple[int, ...]; groups:tuple[int, ...]; localid_reg:int; wgid_reg:int # noqa: E702
  lmem_size:int; pvt_size:int; textures:list[Image]; ibos:list[Image]; demote:bool; entry:int # noqa: E702

I32, U32, U16, U8, U64, I64 = dtypes.int32, dtypes.uint32, dtypes.uint16, dtypes.uint8, dtypes.uint64, dtypes.int64
F32, F64, F16, I8, I16 = dtypes.float32, dtypes.float64, dtypes.half, dtypes.int8, dtypes.int16
def _c(v:int|float|bool, dt:DType=U32) -> UOp: return UOp.const(v, dt)
def _ci(v:int) -> UOp: return UOp.const(v, I32)
TRUE, FALSE = _c(True, dtypes.bool), _c(False, dtypes.bool)
def _not(b:UOp) -> UOp: return b.eq(FALSE)
def _bits(x:UOp, lo:int, n:int) -> UOp: return (x >> _c(lo)) & _c((1 << n) - 1)
def _bit(x:UOp, lo:int) -> UOp: return _bits(x, lo, 1).ne(_c(0))
def _fu(half:bool) -> tuple[DType, DType]: return (F16, U16) if half else (F32, U32)
def _sel(idx:UOp, vals:list[UOp]) -> UOp:
  return functools.reduce(lambda acc, iv: idx.eq(_c(iv[0])).where(iv[1], acc), reversed(list(enumerate(vals[:-1]))), vals[-1])
def _ftz_u(u:UOp, half:bool) -> UOp:
  sign, tiny, dt = (0x8000, 0x0400, U16) if half else (0x80000000, 0x00800000, U32)
  return ((u & _c(sign - 1, dt)) < _c(tiny, dt)).where(u & _c(sign, dt), u)
def _ftz(f:UOp, half:bool) -> UOp: return _ftz_u(f.bitcast(_fu(half)[1]), half).bitcast(_fu(half)[0])
def _cnan_u(u:UOp, half:bool) -> UOp:
  sign, inf, qnan, dt = (0x8000, 0x7C00, 0x7E00, U16) if half else (0x80000000, 0x7F800000, 0x7FC00000, U32)
  return ((u & _c(sign - 1, dt)) > _c(inf, dt)).where(_c(qnan, dt), u)
def _isnan(f:UOp, half:bool) -> UOp:
  sign, inf, dt = (0x8000, 0x7C00, U16) if half else (0x80000000, 0x7F800000, U32)
  return (f.bitcast(dt) & _c(sign - 1, dt)) > _c(inf, dt)
def _cov_to_float(v_f64:UOp, dst_half:bool, even:UOp) -> UOp:
  fdt, udt = _fu(dst_half)
  r = v_f64.cast(fdt)
  ru = r.bitcast(udt)
  rtz = ((even.eq(_c(0))) & (r.cast(F64).abs() > v_f64.abs())).where(ru - _c(1, udt), ru)
  return _ftz_u(rtz, True).bitcast(fdt) if dst_half else rtz.bitcast(fdt)
def _clz_u(u:UOp, bits:int) -> UOp:
  x, n = u, _c(0, dt:=U16 if bits == 16 else U32)
  for s in [s for s in (16, 8, 4, 2, 1) if s < bits]:
    z = (x & _c(((1 << s) - 1) << (bits - s), dt)).eq(_c(0, dt))
    n, x = z.where(n + _c(s, dt), n), z.where(x << _c(s, dt), x)
  return u.eq(_c(0, dt)).where(_c((1 << bits) - 1, dt), n)
def _fminmax(a:UOp, b:UOp, half:bool, is_max:bool) -> UOp:
  fdt, udt = _fu(half)
  z = _c(0.0, fdt)
  zres = ((a.bitcast(udt) & b.bitcast(udt)) if is_max else (a.bitcast(udt) | b.bitcast(udt))).bitcast(fdt)
  return _isnan(a, half).where(b, _isnan(b, half).where(a, (a.eq(z) & b.eq(z)).where(zres, ((a > b) if is_max else (a < b)).where(a, b))))
def _sat_f(fval:UOp, half:bool) -> UOp:
  z, one = _c(0.0, fdt:=_fu(half)[0]), _c(1.0, fdt)
  return _isnan(fval, half).where(fval, (fval < z).where(z, (fval > one).where(one, fval)))

def _build_vm() -> tuple[TinyELF, list[int]]:
  ax = itertools.count().__next__
  prog, consts = UOp.param(0, U32, 1 << 22, name="prog"), UOp.param(1, U32, NCONSTS, name="consts")
  vmem = UOp.param(2, U8, 1 << 46, name="vmem")
  r_buf, h_buf = UOp.param(3, U32, NROWS * MAX_LOCAL, name="r"), UOp.param(4, U16, NROWS * MAX_LOCAL, name="h")
  lmem, pvt = UOp.param(5, U8, 1 << 20, name="lmem"), UOp.param(6, U8, 1 << 26, name="pvt")
  tex_m, ibo_m = UOp.param(7, U64, MAX_IMGS * IMG_META, name="tex_m"), UOp.param(8, U64, MAX_IMGS * IMG_META, name="ibo_m")
  err, params = UOp.param(9, I32, 2, name="err"), UOp.param(10, I32, N_PARAMS, name="p_i")
  vmem16, vmem32 = UOp.param(11, U16, 1 << 45, name="vmem16"), UOp.param(12, U32, 1 << 44, name="vmem32")
  st_pc, st_done = UOp.alloc((MAX_LOCAL,), I32, 0, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), U8, 1, AddrSpace.REG)
  st_blk, st_depth = UOp.alloc((MAX_LOCAL,), U8, 2, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), I32, 3, AddrSpace.REG)
  st_calls = UOp.alloc((MAX_CALLS * MAX_LOCAL,), I32, 4, AddrSpace.REG)
  st_uni, st_cur = UOp.alloc((1,), I32, 5, AddrSpace.REG), UOp.alloc((1,), I32, 6, AddrSpace.REG)
  st_go = UOp.alloc((1,), U8, 7, AddrSpace.REG)
  tmp, tmp_f64 = UOp.alloc((MAX_LOCAL,), U32, 8, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), F64, 9, AddrSpace.REG)

  n_local, n_groups, lx, ly, gx, gy, lid_reg, wgid_reg, lmem_sz, pvt_sz, n_tex, n_ibo, demote, entry = \
    [params.index(j).load() for j in range(N_PARAMS)]
  init_err = UOp.group(err.index(0).store(_ci(0)), err.index(1).store(_ci(0)))
  def when(c:UOp) -> UOp: return UOp.range(c.cast(I32), ax())
  def lanes() -> UOp: return UOp.range(n_local, ax())

  wg = UOp.range(n_groups, ax())
  wg_dep = UOp.group(init_err, wg)
  l0 = lanes()
  g_xyz = [(wg % gx).cast(U32), ((wg // gx) % gy).cast(U32), (wg // (gx * gy)).cast(U32)]
  l_xyz = [(l0 % lx).cast(U32), ((l0 // lx) % ly).cast(U32), (l0 // (lx * ly)).cast(U32)]
  has_lid, has_wg = lid_reg.ne(_ci(INVALID_REG)), wgid_reg.ne(_ci(INVALID_REG))
  init_ids = UOp.group(
    *[r_buf.after(wg_dep).index(((lid_reg + _ci(j)) * n_local + l0).valid(has_lid)).store(l_xyz[j]) for j in range(3)],
    *[r_buf.after(wg_dep).index(((wgid_reg + _ci(j)) * n_local + l0).valid(has_wg)).store(g_xyz[j]) for j in range(3)],
    st_pc.after(wg_dep).index(l0).store(entry), st_done.after(wg_dep).index(l0).store(_c(0, U8)),
    st_blk.after(wg_dep).index(l0).store(_c(0, U8)), st_depth.after(wg_dep).index(l0).store(_ci(0))).end(l0)
  init_wg = UOp.group(init_ids, st_uni.after(wg_dep).index(0).store(entry), st_go.after(wg_dep).index(0).store(_c(1, U8)))

  loop = UOp.loop(ax()).replace(src=(UOp(Ops.NOOP).after(init_wg),))
  uni = st_uni.after(loop).index(0).load()
  is_uni = uni >= _ci(0)

  g_div, l_min = when(_not(is_uni)), lanes()
  d_l, b_l = st_done.after(loop, g_div).index(l_min).load(), st_blk.after(loop, g_div).index(l_min).load()
  p_l = st_pc.after(loop, g_div).index(l_min).load()
  any_undone = d_l.eq(_c(0, U8)).cast(I32).reduce(l_min, arg=Ops.MAX)
  min_pc = -(d_l.eq(_c(0, U8)) & b_l.eq(_c(0, U8))).where(-p_l, _ci(-PC_NONE)).reduce(l_min, arg=Ops.MAX)
  dead, e_dead = min_pc.eq(_ci(PC_NONE)) & any_undone.ne(_ci(0)), _ci(ERRS.index("every thread is waiting at a bar"))
  pick_div = UOp.group(st_cur.after(loop, g_div).index(0).store(min_pc),
                       st_go.after(loop, g_div).index(0).store((min_pc < _ci(PC_NONE)).cast(U8)),
                       err.after(loop, g_div).index(0).store(dead.where(e_dead, err.after(loop, g_div).index(0).load()))).end(g_div)
  g_uni = when(is_uni)
  picked = UOp.group(pick_div, st_cur.after(loop, g_uni).index(0).store(uni).end(g_uni))

  go_now = st_go.after(picked).index(0).load().ne(_c(0, U8)) & err.after(picked).index(0).load().eq(_ci(0))
  cur = st_cur.after(picked).index(0).load()
  rec = [prog.after(picked).index((cur * _ci(REC_LEN) + _ci(j)).valid(go_now)).load() for j in range(REC_LEN)]
  kind, op, dst, iters, flags, s_meta, s_val = rec[0], rec[1], rec[2].cast(I32), rec[3].cast(I32), rec[4], rec[5:8], rec[8:11]
  dst_half, sat, cond = flags & _c(1), _bit(flags, 1), _bits(flags, 4, 3)
  def is_kind(k:int) -> UOp: return go_now & kind.eq(_c(k))
  def op_is(table:dict[int, int], *opcs:int) -> UOp: return functools.reduce(lambda a, b: a | b, [op.eq(_c(table[o])) for o in opcs])

  def lane_active(lane:UOp, dep:UOp) -> UOp:
    return is_uni.where(TRUE, st_pc.after(dep).index(lane).load().eq(cur) & st_done.after(dep).index(lane).load().eq(_c(0, U8)))
  def set_err(dep:UOp, cond:UOp, msg:str) -> UOp:
    e0 = err.after(dep).index(0).load()
    do_set = cond & e0.eq(_ci(0))
    return UOp.group(err.after(dep).index(0).store(do_set.where(_ci(ERRS.index(msg)), e0)),
                     err.after(dep).index(1).store(do_set.where(cur, err.after(dep).index(1).load())))
  def ld_reg(row:UOp, lane:UOp, *dep:UOp, buf:UOp=r_buf) -> UOp: return buf.after(*dep).index(row * n_local + lane).load()
  def write_reg(dep:UOp, lane:UOp, active:UOp, reg:UOp, dst_half:UOp, val_u32:UOp) -> UOp:
    use_h, idx = dst_half.ne(_c(0)) & (reg < _ci(A0)), reg.clamp(0, NREGS - 1) * n_local + lane
    return UOp.group(h_buf.after(dep).index(idx.valid(active & use_h)).store((val_u32 & _c(0xFFFF)).cast(U16)),
                     r_buf.after(dep).index(idx.valid(active & _not(use_h))).store(val_u32))
  def commit_tmp(dep:UOp, reg:UOp, dst_half:UOp) -> UOp:
    fast = is_uni & (dst_half.eq(_c(0)) | (reg >= _ci(A0)))
    g_f, l_f = when(fast), lanes()
    g_g, l_g = when(_not(fast)), lanes()
    return UOp.group(
      r_buf.after(dep, g_f).index(reg.clamp(0, NREGS - 1) * n_local + l_f).store(tmp.after(dep, g_f).index(l_f).load()).end(l_f).end(g_f),
      write_reg(UOp.group(dep, g_g), l_g, lane_active(l_g, g_g), reg, dst_half, tmp.after(dep, g_g).index(l_g).load()).end(l_g).end(g_g))

  # resolved outside the lane loop, each lane then reads (buf[row + lane] & and_mask) ^ xor_mask
  def prep_src(dep:UOp, slot:int, meta:UOp, val:UOp, k:UOp, is_float:bool, half:bool, bw_neg:UOp|None=None,
               demote_f16:UOp|bool|None=None) -> tuple[UOp, tuple[UOp, UOp, UOp]]:
    udt = U16 if half else U32
    full_m, sign_m, abs_m = _c(0xFFFF if half else 0xFFFFFFFF, udt), _c(0x8000 if half else 0x80000000, udt), _c(0x7FFF if half else 0x7FFFFFFF, udt)
    src_kind = meta & _c(7)
    is_r, is_c, need_rel = [src_kind.eq(_c(SRC_KIND[s])) for s in ("r", "c", "rel")]
    r_inc = _bits(meta, 6, 1).cast(I32) * k
    reg, c_idx = (val.cast(I32) + r_inc).clamp(0, NREGS - 1), (val.cast(I32) + r_inc).clamp(0, 2 * NCONSTS - 1)
    c_full = consts.after(dep).index(c_idx.clamp(0, NCONSTS - 1)).load()
    if half:
      c_pair, df16 = consts.after(dep).index((c_idx >> _ci(1)).clamp(0, NCONSTS - 1)).load(), is_float if demote_f16 is None else demote_f16
      to_f16, raw16 = c_full.bitcast(F32).cast(F16).bitcast(U16).cast(U32), c_full & _c(0xFFFF)
      c_f16 = df16.where(to_f16, raw16) if isinstance(df16, UOp) else to_f16 if df16 else raw16
      cval = demote.ne(_ci(0)).where(c_f16, (c_pair >> ((c_idx & _ci(1)).cast(U32) * _c(16))) & _c(0xFFFF))
    else: cval = c_full
    need_r2h = _c(half, dtypes.bool) & is_r & (reg >= _ci(A0))
    g_pre, l_pre = when(need_rel | need_r2h), lanes()
    rel_idx = ld_reg(_ci(A0), l_pre, dep, g_pre).bitcast(I32) + val.bitcast(I32)
    rel_oob = lane_active(l_pre, UOp.group(dep, g_pre)) & need_rel & ((rel_idx < _ci(0)) | (rel_idx >= _ci(NCONSTS)))
    vrel = consts.after(dep, g_pre).index(rel_idx.clamp(0, NCONSTS - 1)).load()
    idx, pre_val = _ci(NREGS + slot) * n_local + l_pre, need_rel.where(vrel, ld_reg(reg, l_pre, dep, g_pre))
    st = h_buf.after(dep, g_pre).index(idx).store((pre_val & _c(0xFFFF)).cast(U16)) if half else r_buf.after(dep, g_pre).index(idx).store(vrel)
    pre = UOp.group(st, set_err(UOp.group(dep, g_pre), rel_oob, "relative const out of range")).end(l_pre).end(g_pre)
    is_reg, row = is_r | need_rel, (need_rel | need_r2h).where(_ci(NREGS + slot), is_r.where(reg, _ci(0)))
    abs_b, neg_b = _bit(meta, 5), _bit(meta, 4)
    and_m, xor_m = (abs_b.where(abs_m, full_m), neg_b.where(sign_m, _c(0, udt))) if is_float else \
                   (full_m, (neg_b & bw_neg.ne(_c(0))).where(full_m, _c(0, udt))) if bw_neg is not None else (full_m, _c(0, udt))
    sv = (is_c.where(cval, val).cast(udt) & and_m) ^ xor_m
    return pre, (row * n_local, is_reg.where(and_m, _c(0, udt)), is_reg.where(xor_m, sv))

  def prep_3(dep:UOp, k:UOp, is_float:bool, half:bool, bw_neg:UOp|None=None) -> tuple[UOp, list[tuple[UOp, UOp, UOp]]]:
    ps = [prep_src(dep, j, s_meta[j], s_val[j], k, is_float, half, bw_neg) for j in range(3)]
    return UOp.group(*[p[0] for p in ps]), [p[1] for p in ps]
  def ld_src(buf:UOp, pre:UOp, lane:UOp, spec:tuple[UOp, UOp, UOp]) -> UOp:
    return (buf.after(pre).index(spec[0] + lane).load() & spec[1]) ^ spec[2]
  def alu_open(gate:UOp, is_float:bool, half:bool, bw_neg:UOp|None=None) -> tuple[UOp, UOp, UOp, UOp, list[UOp]]:
    g, k = when(gate), UOp.range(iters, ax())
    pre, specs = prep_3(UOp.group(g, k), k, is_float, half, bw_neg)
    lane = lanes()
    srcs = [ld_src(h_buf if half else r_buf, pre, lane, s) for s in specs]
    return g, k, pre, lane, [_ftz(s.bitcast(_fu(half)[0]), half) for s in srcs] if is_float else srcs
  def ld_src0(out:UOp, dep:UOp, k:UOp, src_half:UOp, is_float:bool, conv_f:Callable, conv_h:Callable, demote_h:UOp|None=None) -> UOp:
    g_f, g_h = when(_not(src_half)), when(src_half)
    pre_f, spec_f = prep_src(UOp.group(dep, g_f), 0, s_meta[0], s_val[0], k, is_float, False)
    pre_h, spec_h = prep_src(UOp.group(dep, g_h), 0, s_meta[0], s_val[0], k, is_float, True, demote_f16=demote_h)
    l_f, l_h = lanes(), lanes()
    return UOp.group(out.after(pre_f).index(l_f).store(conv_f(ld_src(r_buf, pre_f, l_f, spec_f))).end(l_f).end(g_f),
                     out.after(pre_h).index(l_h).store(conv_h(ld_src(h_buf, pre_h, l_h, spec_h))).end(l_h).end(g_h))
  def cmp_out(b:UOp, udt:DType) -> UOp:
    v = sat.where((c:=b.cast(udt)) ^ _c(1, udt), c)
    return _bit(flags, 2).where(_c(0, udt) - v, v)
  def _pack_f(fv:UOp, half:bool, dst_half:UOp) -> UOp:
    return dst_half.ne(_c(0)).where(fv.bitcast(U16).cast(U32), fv.cast(F32).bitcast(U32)) if half else \
           dst_half.ne(_c(0)).where(_cov_to_float(fv.cast(F64), True, _c(0)).bitcast(U16).cast(U32), fv.bitcast(U32))
  def _ialu_32(au:UOp, bu:UOp, cu:UOp, bits:int, udt:DType, sdt:DType) -> list[UOp]:
    ai, bi, all1 = au.bitcast(sdt), bu.bitcast(sdt), _c(0xFFFF if bits == 16 else 0xFFFFFFFF, udt)
    sh, sha = bu & _c(bits - 1, udt), au & _c(bits - 1, udt)
    a32, b32, c32 = au.cast(U32), bu.cast(U32), cu.cast(U32)
    ms24 = (((a32 & _c(0xFFFFFF)) ^ _c(0x800000)) - _c(0x800000)) * (((b32 & _c(0xFFFFFF)) ^ _c(0x800000)) - _c(0x800000))
    return [
      (au + bu).cast(U32), (au + bu).cast(U32), (au - bu).cast(U32), (au - bu).cast(U32),
      (au & bu).cast(U32), (au | bu).cast(U32), (au ^ bu).cast(U32), (au ^ all1).cast(U32),
      (au << sh).cast(U32), (au >> sh).cast(U32), (ai >> sh.bitcast(sdt)).bitcast(udt).cast(U32), ((au >> sh) & _c(1, udt)).cast(U32),
      ms24, (a32 & _c(0xFFFFFF)) * (b32 & _c(0xFFFFFF)), (a32 & _c(0xFFFF)) * (b32 & _c(0xFFFF)),
      (((a32 & _c(0xFFFF)) * (b32 >> _c(16))) << _c(16)) + c32, ((a32 & _c(0xFFFF)) * (b32 & _c(0xFFFF))) + c32, ms24 + c32,
      bu.ne(_c(0, udt)).where(au, cu).cast(U32), (bi >= _c(0, sdt)).where(au, cu).cast(U32),
      ((bu >> sha) & cu).cast(U32), ((bu >> sha) | cu).cast(U32), ((bu << sha) | cu).cast(U32),
      ((bu << sha) & cu).cast(U32), ((bu & au) | cu).cast(U32)]

  def emit_undef() -> UOp: return set_err(g:=when(is_kind(K_UNDEF)), TRUE, "undef").end(g)

  def emit_fmad(k_tag:int, half:bool) -> UOp:
    fdt, udt = _fu(half)
    g, k, pre, lane, (a, b, c) = alu_open(is_kind(k_tag), True, half)
    comp = tmp.after(pre).index(lane).store(_cnan_u(_ftz_u((_ftz(a * b, half) + c).bitcast(udt), half), half).cast(U32)).end(lane)
    g_p, l_p = when(sat | dst_half.ne(_c(int(half)))), lanes()
    fv = tmp.after(comp, g_p).index(l_p).load().cast(udt).bitcast(fdt)
    post = tmp.after(comp, g_p).index(l_p).store(_pack_f(sat.where(_sat_f(fv, half), fv), half, dst_half)).end(l_p).end(g_p)
    return commit_tmp(UOp.group(comp, post), dst + k, dst_half).end(k).end(g)

  def emit_fmisc(half:bool) -> UOp:
    fdt, udt = _fu(half)
    g, k, pre, lane, (a, b, c) = alu_open(is_kind(K_FMISC) & _bits(flags, 9, 1).eq(_c(int(half))), True, half)
    z, one, a64 = _c(0.0, fdt), _c(1.0, fdt), a.cast(F64)
    tr64 = a64.trunc()
    sign_f = _isnan(a, half).where(z, (a > z).where(one, (a < z).where(_c(-1.0, fdt), a)))
    fout = _sel(op, [_fminmax(a, b, half, False), _fminmax(a, b, half, True), sign_f, a,
                     (a64 < tr64).where(tr64 - _c(1.0, F64), tr64).cast(fdt), tr64.cast(fdt), ((b > z) | b.eq(z)).where(a, c)])
    fu = _ftz_u(fout.bitcast(udt), half)
    fc = op_is(FMISC_IDX, mesa.OPC_SEL_F32).where(fu, _cnan_u(fu, half)).bitcast(fdt)
    out_f = _pack_f(sat.where(_sat_f(fc, half), fc), half, dst_half)
    cmp_u = cmp_out(_sel(cond, [a < b, (a < b) | a.eq(b), a > b, (a > b) | a.eq(b), a.eq(b), a.ne(b)]), udt)
    comp = tmp.after(pre).index(lane).store(op_is(FMISC_IDX, mesa.OPC_CMPS_F).where(cmp_u.cast(U32), out_f)).end(lane)
    return commit_tmp(comp, dst + k, dst_half).end(k).end(g)

  def emit_ialu32() -> UOp:
    g, k, pre, lane, (a, b, c) = alu_open(is_kind(K_IALU32), False, False, _bits(flags, 8, 1))
    return commit_tmp(tmp.after(pre).index(lane).store(_sel(op, _ialu_32(a, b, c, 32, U32, I32))).end(lane), dst + k, _c(0)).end(k).end(g)

  def emit_imisc(half:bool) -> UOp:
    bits, udt, sdt = (16, U16, I16) if half else (32, U32, I32)
    all1 = _c(0xFFFF if half else 0xFFFFFFFF, udt)
    g, k, pre, lane, srcs = alu_open(is_kind(K_IMISC) & _bits(flags, 9, 1).eq(_c(int(half))), False, half)
    is_bw, is_s = _bit(flags, 8), _bit(flags, 10)
    def imod(u:UOp, m:UOp) -> UOp:
      u = (is_s & _bit(m, 5) & (u.bitcast(sdt) < _c(0, sdt))).where(_c(0, udt) - u, u)
      return _bit(m, 4).where(is_bw.where(u ^ all1, _c(0, udt) - u), u)
    au, bu, cu = [imod(u, m) for u, m in zip(srcs, s_meta)]
    ai, bi = au.bitcast(sdt), bu.bitcast(sdt)
    cu_b = _sel(cond, [au < bu, au <= bu, au > bu, au >= bu, au.eq(bu), au.ne(bu)])
    cs_b = _sel(cond, [ai < bi, ai <= bi, ai > bi, ai >= bi, ai.eq(bi), ai.ne(bi)])
    is_cmps_s = op_is(INT_IDX, mesa.OPC_CMPS_S)
    cmp_v = cmp_out(is_cmps_s.where(cs_b, cu_b), udt)
    out32 = _sel(op, _ialu_32(au, bu, cu, bits, udt, sdt) + [
      (au < bu).where(au, bu).cast(U32), (ai < bi).where(au, bu).cast(U32),
      (au > bu).where(au, bu).cast(U32), (ai > bi).where(au, bu).cast(U32), au.cast(U32),
      _clz_u(au, bits).cast(U32), (au + bu + cu).cast(U32), cmp_v.cast(U32), cmp_v.cast(U32)])
    if half: out32 = (dst_half.eq(_c(0)) & is_s & _not(is_cmps_s)).where(out32.cast(U16).bitcast(I16).cast(I32).bitcast(U32), out32)
    return commit_tmp(tmp.after(pre).index(lane).store(out32).end(lane), dst + k, dst_half).end(k).end(g)

  def emit_sfu() -> UOp:
    g, k = when(is_kind(K_SFU)), UOp.range(iters, ax())
    src_half = _bit(flags, 2)
    ld = ld_src0(tmp_f64, UOp.group(g, k), k, src_half, True, lambda v: _ftz(v.bitcast(F32), False).cast(F64),
                 lambda v: _ftz(v.bitcast(F16), True).cast(F64))
    fns = [lambda x: _c(1.0, F64) / x, lambda x: _c(1.0, F64) / x.sqrt(),
           lambda x: x.log2(), lambda x: x.exp2(), lambda x: x.sin(), lambda x: x.sqrt()]
    evals = []
    for sop, fn in enumerate(fns):
      gs, ls = when(op.eq(_c(sop))), lanes()
      evals.append(tmp_f64.after(ld, gs).index(ls).store(fn(tmp_f64.after(ld, gs).index(ls).load())).end(ls).end(gs))
    ev, lane = UOp.group(*evals), lanes()
    y32 = tmp_f64.after(ev).index(lane).load().cast(F32)
    y16 = _cnan_u(_cov_to_float(y32.cast(F64), True, _c(0)).bitcast(U16), True).bitcast(F16)
    y32c = _cnan_u(_ftz_u(y32.bitcast(U32), False), False).bitcast(F32)
    out = src_half.where(_pack_f(sat.where(_sat_f(y16, True), y16), True, dst_half), _pack_f(sat.where(_sat_f(y32c, False), y32c), False, dst_half))
    return commit_tmp(tmp.after(ev).index(lane).store(out).end(lane), dst + k, dst_half).end(k).end(g)

  def emit_mov_same(half:bool) -> UOp:
    buf = h_buf if half else r_buf
    g = when(is_kind(K_MOV) & op.eq(_c(MOV_SAME)) & dst_half.eq(_c(int(half))))
    k, lane = UOp.range(iters, ax()), lanes()
    pre, spec = prep_src(UOp.group(g, k), 0, s_meta[0], s_val[0], k, False, half, demote_f16=_bits(flags, 4, 3).eq(_c(mesa.TYPE_F16)))
    return commit_tmp(tmp.after(pre).index(lane).store(ld_src(buf, pre, lane, spec).cast(U32)).end(lane), dst + k, _c(int(half))).end(k).end(g)

  def emit_swz(half:bool) -> UOp:
    buf = h_buf if half else r_buf
    g = when(is_kind(K_MOV) & op.eq(_c(MOV_SWZ)) & _bits(s_meta[0], 3, 1).eq(_c(int(half))))
    pre0, spec0 = prep_src(g, 0, s_meta[0], s_val[0], _ci(0), False, half)
    pre1, spec1 = prep_src(g, 1, s_meta[1], s_val[1], _ci(0), False, half)
    pre, lane = UOp.group(pre0, pre1), lanes()
    v0, v1 = ld_src(buf, pre, lane, spec0).cast(U32), ld_src(buf, pre, lane, spec1).cast(U32)
    dep, act = UOp.group(v0, v1), lane_active(lane, pre)
    return UOp.group(write_reg(dep, lane, act, dst, dst_half, v0), write_reg(dep, lane, act, s_val[2].cast(I32), dst_half, v1)).end(lane).end(g)

  def emit_cov() -> UOp:
    g, k = when(is_kind(K_MOV) & op.eq(_c(MOV_COV))), UOp.range(iters, ax())
    src_t, dst_t, even = _bits(flags, 4, 3), _bits(flags, 8, 3), _bits(flags, 2, 1)
    ld = ld_src0(tmp, UOp.group(g, k), k, _bit(flags, 1), False, lambda v: v, lambda v: v.cast(U32), demote_h=src_t.eq(_c(mesa.TYPE_F16)))
    raw = tmp.after(ld).index(lane:=lanes()).load()
    # cov from u8 (6) and s8 (7) both sign-extend
    s_f64 = _sel(src_t, [raw.cast(U16).bitcast(F16).cast(F64), raw.bitcast(F32).cast(F64), (raw & _c(0xFFFF)).cast(F64), raw.cast(F64),
                         raw.cast(U16).bitcast(I16).cast(F64), raw.bitcast(I32).cast(F64), raw.cast(U8).bitcast(I8).cast(F64)])
    s_i64 = _sel(src_t, [_c(0, I64), _c(0, I64), (raw & _c(0xFFFF)).cast(I64), raw.cast(I64),
                         raw.cast(U16).bitcast(I16).cast(I64), raw.bitcast(I32).cast(I64), raw.cast(U8).bitcast(I8).cast(I64)])
    f_clean = s_f64.ne(s_f64).where(_c(0.0, F64), s_f64)
    dmin = _sel(dst_t, [_c(v, F64) for v in (0.0, 0.0, 0.0, 0.0, -32768.0, -2147483648.0, 0.0, -128.0)])
    dmax = _sel(dst_t, [_c(v, F64) for v in (0.0, 0.0, 65535.0, 4294967295.0, 32767.0, 2147483647.0, 255.0, 127.0)])
    f_cl = (f_clean < dmin).where(dmin, (f_clean > dmax).where(dmax, f_clean.trunc()))
    f2i = dst_t.eq(_c(mesa.TYPE_U32)).where(f_cl.cast(U32), f_cl.cast(I64).cast(U32))
    to_f = dst_t.eq(_c(mesa.TYPE_F16)).where(_cov_to_float(s_f64, True, even).bitcast(U16).cast(U32), _cov_to_float(s_f64, False, even).bitcast(U32))
    out = src_t.eq(dst_t).where(raw, (dst_t < _c(2)).where(to_f, (src_t < _c(2)).where(f2i, s_i64.cast(U32))))
    out = dst_t.eq(_c(mesa.TYPE_U8)).where(out & _c(0xFF), dst_t.eq(_c(mesa.TYPE_U8_32)).where(out.cast(U8).bitcast(I8).cast(U16).cast(U32), out))
    return commit_tmp(tmp.after(ld).index(lane).store(out).end(lane), dst + k, dst_half).end(k).end(g)

  def emit_mem() -> UOp:
    g, lane = when(is_kind(K_MEM)), lanes()
    act, m_type, size, esz = lane_active(lane, g), _bits(flags, 4, 3), iters, s_meta[0].cast(I32)
    areg, moff = s_val[0].cast(I32).clamp(0, NREGS - 1), s_val[1].bitcast(I32)
    addr_lo, addr_hi = ld_reg(areg, lane, g), ld_reg((areg + _ci(1)).clamp(0, NREGS - 1), lane, g).cast(U64) << _c(32, U64)
    r_src2, r_sh, r_off = s_val[2].cast(I32).clamp(0, NREGS - 1), _bits(flags, 8, 2).cast(U64), _bits(flags, 12, 2).cast(U64)
    gaddr = (addr_lo.cast(U64) | addr_hi) + moff.cast(I64).bitcast(U64)
    gaddr = _bit(flags, 2).where(gaddr + ((ld_reg(r_src2, lane, g).cast(U64) << r_sh) + r_off) * esz.cast(U64), gaddr)
    loc_off = addr_lo.bitcast(I32) + moff
    is_g, is_l = op_is(MEM_IDX, mesa.OPC_LDG, mesa.OPC_STG), op_is(MEM_IDX, mesa.OPC_LDL, mesa.OPC_STL)
    is_p, is_ld = op_is(MEM_IDX, mesa.OPC_LDP, mesa.OPC_STP), op_is(MEM_IDX, mesa.OPC_LDG, mesa.OPC_LDL, mesa.OPC_LDP)
    l_ok, p_ok = (loc_off >= _ci(0)) & (loc_off + esz * size <= lmem_sz), (loc_off >= _ci(0)) & (loc_off + esz * size <= pvt_sz)
    mem_ok, comp = act & is_g.where(TRUE, is_l.where(l_ok, p_ok)), UOp.range(size, ax())
    c_off = comp * esz
    def mptr(b:int, ok:UOp) -> list[UOp]:
      return [vmem.after(g).index((gaddr + (c_off + _ci(b)).cast(U64)).cast(I64).valid(ok & is_g)),
              lmem.after(g).index((loc_off + c_off + _ci(b)).valid(ok & is_l)),
              pvt.after(g).index((lane * pvt_sz + loc_off + c_off + _ci(b)).valid(ok & is_p))]
    def ld_b(b:int) -> UOp:
      ok = mem_ok & is_ld & (_ci(b) < esz)
      gv, lv, pv = [p.load() for p in mptr(b, ok)]
      return ok.where(is_g.where(gv, is_l.where(lv, pv)).cast(U32) << _c(b * 8), _c(0))
    ld_val = ld_b(0) | ld_b(1) | ld_b(2) | ld_b(3)
    ld_val = m_type.eq(_c(mesa.TYPE_U8_32)).where(ld_val.cast(U8).bitcast(I8).cast(U16).cast(U32), ld_val)
    st_reg = (dst + comp).clamp(0, NREGS - 1)
    st_val = dst_half.ne(_c(0)).where(ld_reg(st_reg, lane, g, buf=h_buf).cast(U32), ld_reg(st_reg, lane, g))
    def st_b(b:int) -> UOp:
      bv = ((st_val >> _c(b * 8)) & _c(0xFF)).cast(U8)
      return UOp.group(*[p.store(bv) for p in mptr(b, mem_ok & _not(is_ld) & (_ci(b) < esz))])
    return UOp.group(write_reg(UOp.group(g, gaddr, loc_off, ld_val), lane, act & is_ld, dst + comp, dst_half, ld_val),
                     *[st_b(b) for b in range(4)], set_err(g, act & is_l & _not(l_ok), "out of bounds local access"),
                     set_err(g, act & op_is(MEM_IDX, mesa.OPC_STP) & _not(p_ok), "out of bounds private access")).end(comp).end(lane).end(g)

  def img_meta(m:UOp, dep:UOp, i:UOp, n:UOp, ok:UOp|None=None) -> list[UOp]:
    base = i.clamp(0, (n - _ci(1)).clamp(0, MAX_IMGS - 1)) * _ci(IMG_META)
    return [m.after(dep).index(base + _ci(j) if ok is None else (base + _ci(j)).valid(ok)).load() for j in range(IMG_META)]
  def ld_xy(coord:UOp, lane:UOp, g:UOp) -> list[UOp]: return [ld_reg(c.clamp(0, NREGS - 1), lane, g).cast(U64) for c in (coord, coord + _ci(1))]
  def img_ptrs(g:UOp, addr:UOp, elem:UOp, ok:UOp, is16:UOp) -> tuple[UOp, UOp]:
    return (vmem16.after(g).index(((addr >> _c(1, U64)) + elem).cast(I64).valid(ok & is16)),
            vmem32.after(g).index(((addr >> _c(2, U64)) + elem).cast(I64).valid(ok & _not(is16))))

  def emit_isam() -> UOp:
    g = when(is_kind(K_TEX) & op.eq(_c(TEX_ISAM)))
    coord, tex_i, wrmask, tex_in_reg = s_val[0].cast(I32), s_val[1].cast(I32), iters, (flags & _c(1)).ne(_c(0))
    meta_static, lane = img_meta(tex_m, g, tex_i, n_tex), lanes()
    act = lane_active(lane, g)
    tex_reg = h_buf.after(g).index((tex_i.clamp(0, NREGS - 1) * n_local + lane).valid(tex_in_reg)).load().cast(I32)
    t_idx = tex_in_reg.where(tex_reg, tex_i)
    t_oob = act & ((t_idx < _ci(0)) | (t_idx >= n_tex))
    t_addr, t_w, t_h, t_pitch, t_f16 = [tex_in_reg.where(d, s) for d, s in zip(img_meta(tex_m, g, t_idx, n_tex, tex_in_reg), meta_static)]
    tx, ty = ld_xy(coord, lane, g)
    t_ok, t_elem = act & _not(t_oob) & (tx < t_w) & (ty < t_h), ty * t_pitch + tx * _c(4, U64)
    is_dst_f16, is_i16 = _bits(flags, 4, 3).eq(_c(mesa.TYPE_F16)), t_f16.ne(_c(0, U64))
    t_vals = []
    for ch in range(4):
      has_ch = _bit(wrmask.cast(U32), ch)
      ok_ch, elem = t_ok & has_ch, t_elem + _c(ch, U64)
      p16, p32 = img_ptrs(g, t_addr, elem, ok_ch, is_i16)
      fv = ok_ch.where(is_i16.where(p16.load().bitcast(F16).cast(F32), p32.load().bitcast(F32)), _c(0.0, F32))
      t_vals.append((has_ch, is_dst_f16.where(fv.cast(F16).bitcast(U16).cast(U32), fv.bitcast(U32))))
    dep = UOp.group(*[v for _, v in t_vals])
    return UOp.group(set_err(g, t_oob, "texture is not bound"), *[
      write_reg(dep, lane, act & hc & _not(t_oob), dst + sum(((wrmask >> _ci(p)) & _ci(1) for p in range(ch)), _ci(0)), is_dst_f16.cast(U32), oc)
      for ch, (hc, oc) in enumerate(t_vals)]).end(lane).end(g)

  def emit_stib() -> UOp:
    g, coord, ibo_i, ncomp = when(is_kind(K_TEX) & op.eq(_c(TEX_STIB))), s_val[0].cast(I32), dst, iters
    i_addr, i_w, i_h, i_pitch, i_f16 = img_meta(ibo_m, g, ibo_i, n_ibo)
    lane, comp = lanes(), UOp.range(ncomp, ax())
    act = lane_active(lane, g)
    ib_oob = act & ((ibo_i < _ci(0)) | (ibo_i >= n_ibo))
    ix, iy = ld_xy(coord, lane, g)
    ib_ok, is_img_f16 = act & _not(ib_oob) & (ix < i_w) & (iy < i_h), i_f16.ne(_c(0, U64))
    vreg = (s_val[1].cast(I32) + comp).clamp(0, NREGS - 1)
    vf = _bits(flags, 4, 3).eq(_c(mesa.TYPE_F16)).where(ld_reg(vreg, lane, g, buf=h_buf).bitcast(F16).cast(F32), ld_reg(vreg, lane, g).bitcast(F32))
    p16, p32 = img_ptrs(g, i_addr, iy * i_pitch + ix * _c(4, U64) + comp.cast(U64), ib_ok, is_img_f16)
    stores = UOp.group(p16.store(vf.cast(F16).bitcast(U16)), p32.store(vf.bitcast(U32)))
    return UOp.group(stores, set_err(g, ib_oob, "IBO is not bound")).end(comp).end(lane).end(g)

  exec_done = UOp.group(
    emit_undef(), UOp.group(emit_fmad(K_FMAD32, False), emit_fmad(K_FMAD16, True)), UOp.group(emit_fmisc(False), emit_fmisc(True)), emit_ialu32(),
    UOp.group(emit_imisc(False), emit_imisc(True)), emit_sfu(),
    UOp.group(emit_mov_same(False), emit_mov_same(True), emit_swz(False), emit_swz(True), emit_cov()), emit_mem(), emit_isam(), emit_stib())

  # pc update
  is_non_ctrl, is_ctrl = go_now & kind.ne(_c(K_CTRL)) & kind.ne(_c(K_UNDEF)), is_kind(K_CTRL)
  g_adv_u = when(is_non_ctrl & is_uni)
  g_adv_d, l_ad = when(is_non_ctrl & _not(is_uni)), lanes()
  is_nop, is_end, is_call, is_ret, is_bar = [op_is(CTRL_IDX, o) for o in (mesa.OPC_NOP, mesa.OPC_END, mesa.OPC_CALL, mesa.OPC_RET, mesa.OPC_BAR)]
  is_br = op_is(CTRL_IDX, mesa.OPC_JUMP, mesa.OPC_BR, mesa.OPC_BRAO, mesa.OPC_BRAA, mesa.OPC_PREDT, mesa.OPC_PREDF)
  g_nop = when(is_ctrl & is_uni & is_nop)
  g_cf = when(is_ctrl & _not(is_uni & is_nop))
  l_bar, l_cf = lanes(), lanes()
  bar_other = (_not(is_uni) & st_done.after(exec_done, g_cf).index(l_bar).load().eq(_c(0, U8)) &
               st_pc.after(exec_done, g_cf).index(l_bar).load().ne(cur)).cast(I32).reduce(l_bar, arg=Ops.MAX)
  dep_cf, target, comp1 = UOp.group(exec_done, g_cf, bar_other), dst, iters & _ci(3)
  act_cf, depth = lane_active(l_cf, dep_cf), st_depth.after(dep_cf).index(l_cf).load()
  br1 = ld_reg(_ci(P0) + comp1, l_cf, dep_cf).ne(_c(0)) ^ (flags & _c(1)).ne(_c(0))
  br2 = ld_reg(_ci(P0) + _bits(flags, 4, 2).cast(I32), l_cf, dep_cf).ne(_c(0)) ^ _bit(flags, 1)
  p0_nz = ld_reg(_ci(P0), l_cf, dep_cf).ne(_c(0))
  br_go = _sel(op, [FALSE, FALSE, TRUE, br1, br1 | br2, br1 & br2, _not(p0_nz), p0_nz])
  nxt_pc = cur + _ci(1)
  ret_pc = st_calls.after(dep_cf).index(((depth - _ci(1)).clamp(0, MAX_CALLS - 1)) * n_local + l_cf).load()
  bar_pc = bar_other.ne(_ci(0)).where(cur, nxt_pc)
  new_pc = is_br.where(br_go.where(target, nxt_pc), is_call.where(target, is_ret.where(ret_pc, is_bar.where(bar_pc, nxt_pc))))
  call_ovf, ret_udf = act_cf & is_call & (depth >= _ci(MAX_CALLS)), act_cf & is_ret & (depth <= _ci(0))
  s_cf_lanes = UOp.group(
    st_pc.after(dep_cf).index(l_cf.valid(act_cf)).store(new_pc),
    st_done.after(dep_cf).index(l_cf.valid(act_cf)).store(is_end.where(_c(1, U8), st_done.after(dep_cf).index(l_cf).load())),
    st_blk.after(dep_cf).index(l_cf.valid(act_cf)).store(is_bar.where(bar_other.ne(_ci(0)).cast(U8), st_blk.after(dep_cf).index(l_cf).load())),
    st_depth.after(dep_cf).index(l_cf.valid(act_cf)).store(is_call.where(depth + _ci(1), is_ret.where(depth - _ci(1), depth))),
    st_calls.after(dep_cf).index((depth.clamp(0, MAX_CALLS - 1) * n_local + l_cf).valid(act_cf & is_call & _not(call_ovf))).store(nxt_pc),
    set_err(dep_cf, call_ovf, "call stack overflow"), set_err(dep_cf, ret_udf, "ret without call")).end(l_cf)
  l_chk, pc0 = lanes(), st_pc.after(s_cf_lanes).index(0).load()
  any_diff = (st_done.after(s_cf_lanes).index(l_chk).load().ne(_c(0, U8)) | st_blk.after(s_cf_lanes).index(l_chk).load().ne(_c(0, U8)) |
              st_pc.after(s_cf_lanes).index(l_chk).load().ne(pc0)).cast(I32).reduce(l_chk, arg=Ops.MAX)
  any_left = st_done.after(s_cf_lanes).index(l_chk).load().eq(_c(0, U8)).cast(I32).reduce(l_chk, arg=Ops.MAX)
  step = UOp.group(
    st_uni.after(exec_done, g_adv_u).index(0).store(nxt_pc).end(g_adv_u),
    st_pc.after(exec_done, g_adv_d).index(l_ad.valid(lane_active(l_ad, g_adv_d))).store(nxt_pc).end(l_ad).end(g_adv_d),
    st_uni.after(exec_done, g_nop).index(0).store(nxt_pc).end(g_nop),
    UOp.group(st_uni.after(s_cf_lanes).index(0).store(any_diff.eq(_ci(0)).where(pc0, _ci(-1))),
              st_go.after(s_cf_lanes).index(0).store(any_left.cast(U8))).end(g_cf))
  wg_done = step.backedge(loop, st_go.after(step).index(0).load().ne(_c(0, U8)) & err.after(step).index(0).load().eq(_ci(0))).end(wg)

  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", TRANSCENDENTAL=1, CAPTURE_PROCESS_REPLAY=0):
    prg = to_program(UOp.sink(wg_done, arg=KernelInfo(name="qcom_a630_vm")), Device["CPU"].renderer)
  # not TinyELF.from_program: its profile key hashes the whole VM graph, which recurses past hypothesis' limit and takes longer than the build
  sig = tuple((p.arg.name, p.arg.addrspace, p.dtype, p._shape) for p in prg.kernel_params)
  return TinyELF(prg.src[3].arg, prg.src[0].arg.function_name, prg.arg.target, sig), [p.arg.slot for p in prg.kernel_params]

@functools.cache
def vm_kernel() -> Callable[..., float|None]:
  if (vm:=diskcache_get("qcom_a630_vm", key:=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())) is None:
    diskcache_put("qcom_a630_vm", key, vm:=_build_vm())
  obj, slots = vm
  rt = Device["CPU"].runtime(obj, links={})
  return lambda *bufs: rt(*[bufs[s] for s in slots])

def _pack_images(imgs:list[Image]) -> np.ndarray:
  if not imgs: return np.zeros(IMG_META, np.uint64)
  return np.array([[i.addr, i.width, i.height, i.pitch // i.dtype.itemsize, int(i.dtype == np.float16)] for i in imgs], np.uint64).reshape(-1)

def run(d:Dispatch):
  n_local, n_groups = math.prod(d.local_size), math.prod(d.groups)
  assert n_local <= MAX_LOCAL, f"local size {n_local} exceeds {MAX_LOCAL}"
  prog = decode(d.image)
  r_buf, h_buf = np.full(NROWS * n_local, REG_INIT, np.uint32), np.full(NROWS * n_local, REG_INIT & 0xFFFF, np.uint16)
  lmem, pvt = np.zeros(max(d.lmem_size, 1), np.uint8), np.zeros(max(d.pvt_size * n_local, 1), np.uint8)
  tex_m, ibo_m, err = _pack_images(d.textures), _pack_images(d.ibos), np.zeros(2, np.int32)
  params = np.array([n_local, n_groups, d.local_size[0], d.local_size[1], d.groups[0], d.groups[1],
                     d.localid_reg, d.wgid_reg, d.lmem_size, d.pvt_size, len(d.textures), len(d.ibos), int(d.demote), d.entry], np.int32)
  vm_kernel()(prog.ctypes.data, d.consts.ctypes.data, 0, r_buf.ctypes.data, h_buf.ctypes.data,
              lmem.ctypes.data, pvt.ctypes.data, tex_m.ctypes.data, ibo_m.ctypes.data, err.ctypes.data, params.ctypes.data, 0, 0)
  code, epc = int(err[0]), int(err[1])
  if (msg:=ERRS[code]) == "undef": raise decode_inst(epc, int(np.frombuffer(d.image, np.uint64)[epc])).error()
  if code: raise RuntimeError(msg if msg == "every thread is waiting at a bar" else f"pc {epc}: {msg}")
