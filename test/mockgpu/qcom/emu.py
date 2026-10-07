import functools, math
from dataclasses import dataclass, replace
from typing import Callable
import numpy as np
from tinygrad.helpers import getbits, to_mv
from tinygrad.runtime.autogen import mesa

NREGS = 64 * 4 # regid = gpr*4 + component
A0, P0 = 61 * 4, 62 * 4
TYPES = [np.dtype(t) for t in ("f2", "f4", "u2", "u4", "i2", "i4", "u1", "i1")] # last is u8_32
HALF_TYPES = (0, 2, 4, 6, 7)
FLUT = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0]

def view(kind:str, half:bool) -> np.dtype: return np.dtype(f"{kind}{2 if half else 4}")
def sext(x:int, n:int) -> int: return x - (1 << n) if x & (1 << (n - 1)) else x

class Field:
  def __init__(self, lo:int, hi:int|None=None): self.lo, self.hi = lo, lo if hi is None else hi
  def __get__(self, i, owner=None) -> int: return getbits(i.word, self.lo, self.hi)

@dataclass(frozen=True)
class Src:
  kind:str; val:int; half:bool = False; absneg:int = 0; r:bool = False # noqa: E702
  def at(self, k:int) -> "Src": return replace(self, val=self.val + k) if self.r and k else self

def multisrc(x:int, half:bool, r:int) -> Src:
  absneg = getbits(x, 14, 15)
  if getbits(x, 11, 13) == 0b000: return Src("r", getbits(x, 0, 7), half, absneg, bool(r))
  if getbits(x, 10, 13) == 0b0011 and not half: return Src("c<", sext(getbits(x, 0, 9), 10), half, absneg)
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
    if self.op not in CAT0: raise self.error()

class Cat1(Inst):
  dst, repeat, src_r, dst_type, dst_rel = Field(32, 39), Field(40, 41), Field(43), Field(46, 48), Field(49)
  src_type, src_mode, round, multi = Field(50, 52), Field(53, 54), Field(55, 56), Field(57, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    half = self.src_type in HALF_TYPES
    if (self.multi, self.dst_rel) == (0b10, 0): # swz reads both sources before writing
      self.op, self.iterations = mesa.OPC_SWZ + getbits(word, 40, 41), 1
      if self.op != mesa.OPC_SWZ: raise self.error()
      self.srcs, self.dsts = [Src("r", getbits(word, 0, 7), half), Src("r", getbits(word, 8, 15), half)], [self.dst, getbits(word, 16, 23)]
      return
    self.op = mesa.OPC_MOV
    if self.dst_rel or self.multi or self.src_mode == 0b11 or self.round > 1: raise self.error()
    self.srcs = [Src("r", getbits(word, 0, 7), half, r=bool(self.src_r)) if self.src_mode == 0b00 else
                 Src("c", getbits(word, 0, 10), half, r=bool(self.src_r)) if self.src_mode == 0b01 else Src("i", getbits(word, 0, 31), half)]
    if self.src_mode == 0b00 and getbits(word, 11, 11):
      if not getbits(word, 10, 10) or half: raise self.error()
      self.srcs = [Src("c<", sext(getbits(word, 0, 9), 10))]

class Cat2(Inst):
  dst, repeat, sat, src1_r, dst_conv, ei = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(47)
  cond, src2_r, full, opc = Field(48, 50), Field(51), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in CAT2 and (self.op not in CMPS or self.cond >= len(COND)): raise self.error()
    if self.ei and (self.op != mesa.OPC_ADD_U or not self.full): raise self.error()
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
    if self.op not in CAT3: raise self.error()
    half = not self.sat_full if self.alt else self.op in CAT3_HALF
    self.sat, self.dst_half = 0 if self.alt else self.sat_full, half ^ bool(self.dst_conv)
    if self.op == mesa.OPC_MAD_U16: self.dst_half = not self.dst_conv
    self.srcs = [cat3src(self.src1, half, self.alt, self.src1_r, self.src1_neg), Src("r", self.src2, half, self.src2_neg, bool(self.src2_r)),
                 cat3src(self.src3, half, self.alt, self.src3_r, self.src3_neg)]

class Cat4(Inst):
  dst, repeat, sat, src_r, dst_conv, full, opc = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in CAT4: raise self.error()
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

CATS = {0: Cat0, 1: Cat1, 2: Cat2, 3: Cat3, 4: Cat4, 5: Cat5, 7: Cat7}
CAT6 = {mesa.OPC_LDG: Ldg, mesa.OPC_STG: Stg, mesa.OPC_LDL: Load, mesa.OPC_LDP: Load, mesa.OPC_STL: Store, mesa.OPC_STP: Store}

def decode_inst(pc:int, word:int) -> Inst:
  cat = getbits(word, 61, 63)
  if cat == 6 and getbits(word, 52, 53) == 0b10 and getbits(word, 20, 22) == 0b110: return Ibo(pc, word) # bits 54-58 aren't the opcode here
  cls = CAT6.get(6 << 7 | getbits(word, 54, 58)) if cat == 6 else CATS.get(cat)
  if cls is None: raise NotImplementedError(f"pc {pc}: cat{cat} is not emulated {word:#x}")
  return cls(pc, word)

@functools.cache
def decode(image:bytes) -> list[Inst|NotImplementedError]: # padding after end may not decode
  prog:list[Inst|NotImplementedError] = []
  for pc, word in enumerate(np.frombuffer(image, np.uint64).tolist()):
    try: prog.append(decode_inst(pc, word))
    except NotImplementedError as e: prog.append(e)
  preds = [i for i in prog if isinstance(i, Cat0) and i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF, mesa.OPC_PREDE)]
  pad = [*preds, None, None]
  for i, nxt, after in zip(preds, pad[1:], pad[2:]):
    if i.op == mesa.OPC_PREDE: continue
    if nxt is not None and nxt.op == mesa.OPC_PREDE: i.target = nxt.pc
    elif nxt is not None and nxt.op != i.op and after is not None and after.op == mesa.OPC_PREDE: i.target = nxt.pc + 1
    else: prog[i.pc] = i.error()
  return prog

@dataclass(frozen=True)
class Image: addr:int; width:int; height:int; pitch:int; dtype:np.dtype # noqa: E702

@dataclass(frozen=True)
class Dispatch:
  image:bytes; consts:np.ndarray; local_size:tuple[int, ...]; groups:tuple[int, ...]; localid_reg:int; wgid_reg:int # noqa: E702
  lmem_size:int; pvt_size:int; ranges:list[tuple[int, int]]; textures:list[Image]; ibos:list[Image]; demote:bool; samplers:list[bool] # noqa: E702
  entry:int; merged:bool # noqa: E702

class MergedHalf: # hrN.c is the low (c even) or high half of full component (N*4+c) // 2
  def __init__(self, r:np.ndarray): self.r16 = r.view(np.uint16)
  def __getitem__(self, k:int) -> np.ndarray: return self.r16[k // 2][k % 2::2]
  def __setitem__(self, k:int|tuple[int, np.ndarray], v:np.ndarray):
    if isinstance(k, tuple): self[k[0]][k[1]] = v
    else: self[k][:] = v

class Threads:
  def __init__(self, d:Dispatch, n:int):
    self.d, self.r = d, np.zeros((NREGS, n), np.uint32)
    self.h:np.ndarray|MergedHalf = MergedHalf(self.r) if d.merged else np.zeros((NREGS, n), np.uint16)
    self.everyone = self.mask = np.ones(n, bool)
    self.maps = sorted(d.ranges)
    self.starts, self.ends = np.array([s for s,_ in self.maps], np.uint64), np.array([s + sz for s,sz in self.maps], np.uint64)
    self.bufs:dict[int, np.ndarray] = {}
    self.mems:dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

  def buf(self, k:int) -> np.ndarray:
    if k not in self.bufs: self.bufs[k] = np.frombuffer(to_mv(*self.maps[k]), np.uint8)
    return self.bufs[k]

  def read(self, s:Src, dt:np.dtype) -> np.ndarray:
    if s.kind == "r":
      raw = (self.h if s.half and s.val < A0 else self.r)[s.val]
      if s.half and raw.dtype == np.uint32: raw = raw.astype(np.uint16)
    elif s.kind == "c<":
      idx = self.r[A0].view(np.int32).astype(np.int64) + s.val
      if (self.mask & ((idx < 0) | (idx >= len(self.d.consts)))).any(): raise RuntimeError(f"relative const c<a0.x + {s.val}> out of range")
      raw = self.d.consts[np.clip(idx, 0, len(self.d.consts) - 1)]
    else:
      if s.kind == "c": c = int(self.d.consts[s.val] if self.d.demote or not s.half else self.d.consts.view(np.uint16)[s.val])
      else: c = s.val & 0xFFFFFFFF
      if s.kind == "c" and s.half and dt == np.float16 and self.d.demote:
        c = int(np.uint32(c).view(np.float32).astype(np.float16).view(np.uint16))
      raw = np.full(self.mask.shape, c & 0xFFFF if s.half else c, view("u", s.half))
    v = raw.view(dt) if raw.dtype.itemsize == dt.itemsize else raw.astype(dt)
    if s.absneg & 2: v = np.abs(v)
    if s.absneg & 1: v = -v
    return v

  def write(self, regid:int, half:bool, v:np.ndarray, mask:np.ndarray|None=None):
    if regid >= A0: half = False
    if half and v.dtype == np.float32: v = cov_to_float(v, np.dtype(np.float16))
    v, mask = v.astype(view("f" if v.dtype.kind == "f" else "u", half), copy=False).view(view("u", half)), self.mask if mask is None else mask
    if mask is self.everyone: (self.h if half else self.r)[regid] = v
    else: (self.h if half else self.r)[regid, mask] = v[mask]

def bits(v): return np.unpackbits(v.astype(f"<u{v.dtype.itemsize}").view(np.uint8).reshape(len(v), -1), axis=1, bitorder="little")
def first_set(b, v): return np.where(b.any(axis=1), b.argmax(axis=1), -1).astype(v.dtype)
def clz(v): return first_set(bits(v)[:, ::-1], v)
def ctz(v): return first_set(bits(v), v)
def fmin(a, b): # -0 below +0, numpy's fmin only does that on some cpus
  return np.where((a == 0) & (b == 0), np.where(np.signbit(a) | np.signbit(b), -np.abs(a), np.abs(a)), np.fmin(a, b))
def fmax(a, b):
  return np.where((a == 0) & (b == 0), np.where(np.signbit(a) & np.signbit(b), -np.abs(a), np.abs(a)), np.fmax(a, b))
def sign(v): # a zero keeps its sign, NaN gives +0
  one = v.dtype.type(1)
  return np.where(np.isnan(v), v.dtype.type(0), np.where(v > 0, one, np.where(v < 0, -one, v)))
def lo(v, n): return v & v.dtype.type(((1 << n) - 1) & np.iinfo(v.dtype).max)
def s24(v): return (v.astype(np.int32) << 8) >> 8
def shamt(a, b): return b & b.dtype.type(8 * a.dtype.itemsize - 1)

COND = [np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal]
CMPS = {mesa.OPC_CMPS_F: "f", mesa.OPC_CMPS_U: "u", mesa.OPC_CMPS_S: "i", mesa.OPC_CMPV_F: "f", mesa.OPC_CMPV_U: "u", mesa.OPC_CMPV_S: "i"}
CMPV = {mesa.OPC_CMPV_F, mesa.OPC_CMPV_U, mesa.OPC_CMPV_S}
CAT2_1SRC = {mesa.OPC_SIGN_F, mesa.OPC_ABSNEG_F, mesa.OPC_FLOOR_F, mesa.OPC_TRUNC_F, mesa.OPC_ABSNEG_S, mesa.OPC_NOT_B, mesa.OPC_CLZ_B,
             mesa.OPC_SETRM}
BITWISE = {mesa.OPC_AND_B, mesa.OPC_OR_B, mesa.OPC_XOR_B, mesa.OPC_NOT_B}
CAT0 = {mesa.OPC_NOP, mesa.OPC_END, mesa.OPC_JUMP, mesa.OPC_CALL, mesa.OPC_RET, mesa.OPC_BR, mesa.OPC_BRAO, mesa.OPC_BRAA,
        mesa.OPC_PREDT, mesa.OPC_PREDF, mesa.OPC_PREDE}
CAT2:dict[int, tuple[str, Callable]] = {
  mesa.OPC_ADD_F: ("f", np.add), mesa.OPC_MIN_F: ("f", fmin), mesa.OPC_MAX_F: ("f", fmax), mesa.OPC_MUL_F: ("f", np.multiply),
  mesa.OPC_SIGN_F: ("f", sign), mesa.OPC_ABSNEG_F: ("f", np.positive), mesa.OPC_FLOOR_F: ("f", np.floor), mesa.OPC_TRUNC_F: ("f", np.trunc),
  mesa.OPC_ADD_U: ("u", np.add), mesa.OPC_ADD_S: ("i", np.add), mesa.OPC_SUB_U: ("u", np.subtract), mesa.OPC_SUB_S: ("i", np.subtract),
  mesa.OPC_MIN_U: ("u", np.minimum), mesa.OPC_MIN_S: ("i", np.minimum),
  mesa.OPC_MAX_U: ("u", np.maximum), mesa.OPC_MAX_S: ("i", np.maximum), mesa.OPC_ABSNEG_S: ("i", np.positive),
  mesa.OPC_AND_B: ("u", np.bitwise_and), mesa.OPC_OR_B: ("u", np.bitwise_or), mesa.OPC_NOT_B: ("u", np.invert),
  mesa.OPC_XOR_B: ("u", np.bitwise_xor), mesa.OPC_MUL_S24: ("u", lambda a, b: s24(a) * s24(b)),
  mesa.OPC_MUL_U24: ("u", lambda a, b: lo(a.astype(np.uint32), 24) * lo(b.astype(np.uint32), 24)),
  mesa.OPC_MULL_U: ("u", lambda a, b: lo(a, 16) * lo(b, 16)), mesa.OPC_CLZ_B: ("u", clz), mesa.OPC_SETRM: ("u", ctz),
  mesa.OPC_SHL_B: ("u", lambda a, b: a << shamt(a, b)), mesa.OPC_SHR_B: ("u", lambda a, b: a >> shamt(a, b)),
  mesa.OPC_ASHR_B: ("i", lambda a, b: a >> shamt(a, b)), mesa.OPC_GETBIT_B: ("u", lambda a, b: (a >> shamt(a, b)) & a.dtype.type(1))}
CAT3_HALF = {mesa.OPC_MAD_F16, mesa.OPC_SEL_B16, mesa.OPC_SEL_S16}
CAT3:dict[int, tuple[str, Callable]] = {
  mesa.OPC_MADSH_M16: ("u", lambda a, b, c: (lo(a, 16) * (b >> 16) << 16) + c),
  mesa.OPC_MADSH_U16: ("u", lambda a, b, c: (lo(a, 16) * (b >> 16) << 16) + c), mesa.OPC_MAD_U16: ("u", lambda a, b, c: lo(a, 16) * lo(b, 16) + c),
  mesa.OPC_MAD_S24: ("u", lambda a, b, c: s24(a) * s24(b) + c),
  mesa.OPC_SEL_S32: ("i", lambda a, b, c: np.where(b >= 0, a, c)), mesa.OPC_SEL_S16: ("i", lambda a, b, c: np.where(b >= 0, a, c)),
  mesa.OPC_SEL_F32: ("f", lambda a, b, c: np.where(b >= 0, a, c)),
  mesa.OPC_MAD_F16: ("f", lambda a, b, c: ftz(a * b) + c), mesa.OPC_MAD_F32: ("f", lambda a, b, c: ftz(a * b) + c),
  mesa.OPC_SEL_B16: ("u", lambda a, b, c: np.where(b != 0, a, c)), mesa.OPC_SEL_B32: ("u", lambda a, b, c: np.where(b != 0, a, c)),
  mesa.OPC_SHRM: ("u", lambda a, b, c: (b >> shamt(b, a)) & c), mesa.OPC_SHRG: ("u", lambda a, b, c: (b >> shamt(b, a)) | c),
  mesa.OPC_SHLG: ("u", lambda a, b, c: (b << shamt(b, a)) | c),
  mesa.OPC_SHLM: ("u", lambda a, b, c: (b << shamt(b, a)) & c), mesa.OPC_ANDG: ("u", lambda a, b, c: (b & a) | c),
  mesa.OPC_SAD_S32: ("i", lambda a, b, c: a + b + c)}
CAT4:dict[int, Callable] = {mesa.OPC_RCP: np.reciprocal, mesa.OPC_RSQ: lambda x: 1 / np.sqrt(x), mesa.OPC_LOG2: np.log2,
  mesa.OPC_EXP2: np.exp2, mesa.OPC_SIN: np.sin, mesa.OPC_SQRT: np.sqrt, mesa.OPC_HRSQ: lambda x: 1 / np.sqrt(x), mesa.OPC_HLOG2: np.log2,
  mesa.OPC_HEXP2: np.exp2}

def cov_to_float(v, dt, even=False):
  with np.errstate(over="ignore"): r = v.astype(dt)
  if not even: r = np.where(np.abs(r.astype(np.float64)) > np.abs(v.astype(np.float64)), np.nextafter(r, dt.type(0)), r).astype(dt)
  return np.where(np.abs(r) < np.finfo(dt).tiny, np.copysign(dt.type(0), r), r) if dt == np.float16 else r

def exec_mov(t:Threads, i:Cat1, k:int):
  if i.op != mesa.OPC_MOV:
    vals = [t.read(s, view("u", s.half)).copy() for s in i.srcs] # read returns views of the register file
    for d, v in zip(i.dsts, vals): t.write(d, i.dst_type in HALF_TYPES, v)
    return
  src_dt, dst_dt = TYPES[i.src_type], TYPES[i.dst_type]
  v = t.read(i.srcs[0].at(k), src_dt)
  if i.src_type in (6, 7): v = v.view(np.int8) # cov from u8 sign-extends
  if v.dtype.kind == "f" and dst_dt.kind != "f":
    v = np.clip(np.trunc(np.nan_to_num(v.astype(np.float64))), np.iinfo(dst_dt).min, np.iinfo(dst_dt).max)
  elif dst_dt.kind == "f" and v.dtype != dst_dt: v = cov_to_float(v, dst_dt, even=i.round == 1)
  t.write(i.dst + k, i.dst_type in HALF_TYPES, v.astype(dst_dt))

def ftz(v): # float alu flushes denormal sources and results, cov doesn't
  return np.where(np.abs(v) < np.finfo(v.dtype).tiny, np.copysign(v.dtype.type(0), v), v) if v.dtype.kind == "f" else v

def canonical_nan(v):
  return np.where(np.isnan(v), v.dtype.type(np.nan), v) if v.dtype.kind == "f" else v

def exec_alu(t:Threads, i:Cat2|Cat3|Cat4, k:int):
  srcs = [s.at(k) for s in i.srcs]
  if isinstance(i, Cat2) and i.op in CMPS:
    out = COND[i.cond](*[ftz(t.read(s, view(CMPS[i.op], s.half))) for s in srcs]).astype(view("u", srcs[0].half))
    if i.sat: out = out ^ out.dtype.type(1)
    if i.op in CMPV: out = -out
  elif isinstance(i, Cat4): # computed in f64 since numpy's f32 results vary by cpu, half results are truncated
    x = ftz(t.read(srcs[0], view("f", srcs[0].half)))
    out = CAT4[i.op](x.astype(np.float64)).astype(np.float32)
    if x.dtype == np.float16: out = cov_to_float(out, np.dtype(np.float16))
  elif i.op in BITWISE: # (neg) is a bitwise not here
    out = CAT2[i.op][1](*[~t.read(replace(s, absneg=0), view("u", s.half)) if s.absneg & 1 else t.read(s, view("u", s.half)) for s in srcs])
  else:
    kind, fn = (CAT2 if isinstance(i, Cat2) else CAT3)[i.op]
    out = fn(*[ftz(t.read(s, view(kind, s.half))) for s in srcs])
    if isinstance(i, Cat2) and i.ei:
      out = ((t.read(srcs[0], np.dtype(np.uint32)).astype(np.uint64) + t.read(srcs[1], np.dtype(np.uint32))) >> np.uint64(1)).astype(np.uint32)
  out = ftz(out) if i.op == mesa.OPC_SEL_F32 else canonical_nan(ftz(out))
  sat = i.sat and not (isinstance(i, Cat2) and i.op in CMPS)
  if sat and out.dtype.kind != "f": raise i.error()
  t.write(i.dst + k, i.dst_half, np.clip(out, 0, 1) if sat else out)

def global_lanes(t:Threads, i:Ldg|Stg, nbytes:int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  addrs = (t.r[i.addr].astype(np.uint64) | (t.r[i.addr + 1].astype(np.uint64) << np.uint64(32))) + np.uint64(i.offset & (2**64 - 1))
  if i.reg_off is not None:
    src2, shift, off = i.reg_off
    addrs += ((t.r[src2].astype(np.uint64) << np.uint64(shift)) + np.uint64(off)) << np.uint64(0 if i.type >= 6 else 1 if i.type in HALF_TYPES else 2)
  which = np.searchsorted(t.starts, addrs, side="right").astype(np.int64) - 1
  if (bad := t.mask & ((which < 0) | (addrs + np.uint64(nbytes) > t.ends[np.maximum(which, 0)]))).any():
    raise RuntimeError(f"pc {i.pc}: out of bounds global access at {int(addrs[np.argmax(bad)]):#x}")
  ret = []
  for k in np.unique(which[t.mask]).tolist():
    lanes = t.mask & (which == k)
    ret.append((lanes, t.buf(k), (addrs[lanes] - np.uint64(t.maps[k][0])).astype(np.int64)))
  return ret

def exec_mem(t:Threads, i:Load|Store, k:int):
  dt, half = TYPES[i.type], i.type in HALF_TYPES
  if isinstance(i, (Ldg, Stg)): views = global_lanes(t, i, dt.itemsize * i.size)
  else:
    mem, base, size = t.mems[i.op]
    offs = t.r[i.addr][t.mask].astype(np.int64) + i.offset
    bad = (offs < 0) | (offs + dt.itemsize * i.size > size)
    if bad.any() and i.op != mesa.OPC_LDP: raise RuntimeError(f"pc {i.pc}: out of bounds local/private access")
    lanes, outside = t.mask.copy(), t.mask.copy()
    lanes[t.mask], outside[t.mask] = ~bad, bad
    if bad.any() and isinstance(i, Load): # ldp past the private size reads 0
      for c in range(i.size): t.write(i.dst + c, half, np.zeros(len(t.mask), dt), outside)
    views = [(lanes, mem, (base[t.mask] + offs)[~bad])]
  for lanes, mem, offs in views:
    for c in range(i.size):
      idx = offs[:, None] + np.arange(dt.itemsize) + c * dt.itemsize
      if isinstance(i, Load):
        v = np.zeros(len(lanes), dt)
        v[lanes] = mem[idx].copy().view(dt).reshape(-1)
        t.write(i.dst + c, half, v, lanes)
      else:
        row = (t.h if half else t.r)[i.val + c][lanes]
        mem[idx] = (row.astype(np.uint8) if dt.itemsize == 1 else row).view(np.uint8).reshape(-1, dt.itemsize)

def texels(img:Image, x:np.ndarray, y:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  x, y = x.view(np.int32).astype(np.int64), y.view(np.int32).astype(np.int64)
  ok = (x >= 0) & (x < img.width) & (y >= 0) & (y < img.height)
  return np.frombuffer(to_mv(img.addr, img.pitch * img.height), np.uint8), ok, np.where(ok, y * img.pitch + x * 4 * img.dtype.itemsize, 0)

def exec_isam(t:Threads, i:Cat5, k:int): # out of bounds reads the zero border color
  tex = i.tex
  if i.s2en:
    if len(idx := np.unique(t.h[i.src3][t.mask])) != 1: raise i.error()
    tex = int(idx[0])
  if tex >= len(t.d.textures): raise RuntimeError(f"pc {i.pc}: texture {tex} is not bound")
  if not i.s2en and (i.samp >= len(t.d.samplers) or not t.d.samplers[i.samp]): raise RuntimeError(f"pc {i.pc}: unsupported sampler {i.samp}")
  img, dt = t.d.textures[tex], TYPES[i.type]
  mem, ok, off = texels(img, t.r[i.src1], t.r[i.src1 + 1])
  for n, c in enumerate(c for c in range(4) if i.wrmask >> c & 1):
    v = mem[off[:, None] + c * img.dtype.itemsize + np.arange(img.dtype.itemsize)].copy().view(img.dtype).reshape(-1)
    t.write(i.dst + n, dt == np.float16, np.where(ok, v, 0).astype(dt))

def exec_ibo(t:Threads, i:Ibo, k:int): # out of bounds stores are dropped
  if i.ssbo >= len(t.d.ibos): raise RuntimeError(f"pc {i.pc}: IBO {i.ssbo} is not bound")
  img, dt = t.d.ibos[i.ssbo], TYPES[i.type]
  mem, ok, off = texels(img, t.r[i.coord], t.r[i.coord + 1])
  lanes, esz = t.mask & ok, img.dtype.itemsize
  for c in range(i.ncomp):
    v = (t.h if dt == np.float16 else t.r)[i.val + c].view(dt).astype(img.dtype)
    mem[off[lanes][:, None] + c * esz + np.arange(esz)] = v[lanes].view(np.uint8).reshape(-1, esz)

EXEC:dict[type, Callable] = {Cat1: exec_mov, Cat2: exec_alu, Cat3: exec_alu, Cat4: exec_alu, Cat5: exec_isam, Ibo: exec_ibo, Ldg: exec_mem,
                             Stg: exec_mem, Load: exec_mem, Store: exec_mem}

def run(d:Dispatch):
  n_local, n_groups = math.prod(d.local_size), math.prod(d.groups)
  tid = np.arange(n_local * n_groups)
  lid, gid = tid % n_local, tid // n_local
  prog = decode(d.image)
  t = Threads(d, len(tid))
  ops = {i.op for i in prog if isinstance(i, (Load, Store))}
  if ops & {mesa.OPC_LDL, mesa.OPC_STL}:
    t.mems[mesa.OPC_LDL] = t.mems[mesa.OPC_STL] = (np.zeros(n_groups * d.lmem_size, np.uint8), gid * d.lmem_size, d.lmem_size)
  if ops & {mesa.OPC_LDP, mesa.OPC_STP}:
    t.mems[mesa.OPC_LDP] = t.mems[mesa.OPC_STP] = (np.zeros(len(tid) * d.pvt_size, np.uint8), tid * d.pvt_size, d.pvt_size)
  for reg, v, dims in [(d.localid_reg, lid, d.local_size), (d.wgid_reg, gid, d.groups)]:
    if reg != 0xfc: t.r[reg], t.r[reg + 1], t.r[reg + 2] = v % dims[0], (v // dims[0]) % dims[1], v // (dims[0] * dims[1])
  pc, done, blocked = np.zeros(len(tid), np.int64), np.zeros(len(tid), bool), np.zeros(len(tid), bool)
  calls, depth = np.zeros((16, len(tid)), np.int64), np.zeros(len(tid), np.int64)
  together:int|None = d.entry # pc while nothing has diverged
  with np.errstate(all="ignore"):
    while not done.all():
      if together is not None: cur, t.mask = together, t.everyone
      else:
        if not (live := ~done & ~blocked).any(): raise RuntimeError("every thread is waiting at a bar")
        cur = int(pc[live].min()) # lowest pc first, so paths reconverge and nobody passes a bar early
        t.mask = (pc == cur) & ~done
      if isinstance(i := prog[cur], NotImplementedError): raise i
      if (fn := EXEC.get(type(i))) is not None:
        for k in range(i.iterations): fn(t, i, k)
        if together is not None: together = cur + 1
        else: pc[t.mask] = cur + 1
        continue
      if together is not None: pc[:], together = cur, None
      pc[t.mask] = cur + 1
      if isinstance(i, Cat7) and i.op == mesa.OPC_BAR:
        waiting = np.zeros(n_groups, bool)
        np.logical_or.at(waiting, gid, ~done & (pc != cur) & ~t.mask)
        hold = t.mask & waiting[gid]
        pc[hold], blocked[hold], blocked[t.mask & ~hold] = cur, True, False
      elif isinstance(i, Cat0):
        if i.op == mesa.OPC_END: done |= t.mask
        elif i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF): pc[t.mask & ((t.r[P0] != 0) == (i.op == mesa.OPC_PREDF))] = i.target
        elif i.op == mesa.OPC_JUMP: pc[t.mask] = i.target
        elif i.op == mesa.OPC_CALL:
          if (depth[t.mask] >= len(calls)).any(): raise RuntimeError(f"pc {cur}: call stack overflow")
          calls[depth[t.mask], np.nonzero(t.mask)[0]], pc[t.mask] = cur + 1, i.target
          depth[t.mask] += 1
        elif i.op == mesa.OPC_RET:
          if (depth[t.mask] == 0).any(): raise RuntimeError(f"pc {cur}: ret without call")
          depth[t.mask] -= 1
          pc[t.mask] = calls[depth[t.mask], np.nonzero(t.mask)[0]]
        elif i.op in (mesa.OPC_BR, mesa.OPC_BRAO, mesa.OPC_BRAA):
          cond = (t.r[P0 + i.comp1] != 0) ^ bool(i.inv1)
          if i.op == mesa.OPC_BRAO: cond |= (t.r[P0 + i.comp2] != 0) ^ bool(i.inv2)
          if i.op == mesa.OPC_BRAA: cond &= (t.r[P0 + i.comp2] != 0) ^ bool(i.inv2)
          pc[t.mask & cond] = i.target
      if not done.any() and not blocked.any() and (pc == pc[0]).all(): together = int(pc[0])
