import ctypes, functools, math
from dataclasses import dataclass, replace
from typing import Callable, cast
import numpy as np
from tinygrad.helpers import Context, getbits, to_mv
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen import to_program

NREGS = 64 * 4 # regid = gpr*4 + component
A0, P0 = 61 * 4, 62 * 4
TYPES = [np.dtype(t) for t in ("f2", "f4", "u2", "u4", "i2", "i4", "u1", "i1")] # last is u8_32
HALF_TYPES = (0, 2, 4, 6, 7)
FLUT = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0]

@functools.cache
def view(kind:str, half:bool) -> np.dtype: return np.dtype(f"{kind}{2 if half else 4}")
def sext(x:int, n:int) -> int: return x - (1 << n) if x & (1 << (n - 1)) else x

class Field:
  def __init__(self, lo:int, hi:int|None=None): self.lo, self.hi = lo, lo if hi is None else hi
  def __set_name__(self, owner, name): self.name = name
  def __get__(self, i, owner=None) -> int:
    i.__dict__[self.name] = v = getbits(i.word, self.lo, self.hi)
    return v

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
  srcs:list[Src]
  def __init__(self, pc:int, word:int):
    self.pc, self.word = pc, word
    self.op, self.iterations = self.cat << 7 | self.opc, self.repeat + 1
  def error(self): return NotImplementedError(f"pc {self.pc}: {mesa.opc_t.get(self.op, self.op)} is not emulated {self.word:#x}")
  @functools.cached_property
  def repeat_srcs(self) -> list[list[Src]]: return [[s.at(k) for s in self.srcs] for k in range(self.iterations)]

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
  v = t.read(i.repeat_srcs[k][0], src_dt)
  if i.src_type in (6, 7): v = v.view(np.int8) # cov from u8 sign-extends
  if v.dtype.kind == "f" and dst_dt.kind != "f":
    v = np.clip(np.trunc(np.nan_to_num(v.astype(np.float64))), np.iinfo(dst_dt).min, np.iinfo(dst_dt).max)
  elif dst_dt.kind == "f" and v.dtype != dst_dt: v = cov_to_float(v, dst_dt, even=i.round == 1)
  t.write(i.dst + k, i.dst_type in HALF_TYPES, v.astype(dst_dt, copy=False))

@functools.cache
def float_tiny(dt:np.dtype): return np.finfo(dt).tiny

def ftz(v): # float alu flushes denormal sources and results, cov doesn't
  if v.dtype.kind != "f" or not (mask := np.abs(v) < float_tiny(v.dtype)).any(): return v
  return np.where(mask, np.copysign(v.dtype.type(0), v), v)

def canonical_nan(v):
  if v.dtype.kind != "f" or not (mask := np.isnan(v)).any(): return v
  return np.where(mask, v.dtype.type(np.nan), v)

def exec_alu(t:Threads, i:Cat2|Cat3|Cat4, k:int):
  srcs = i.repeat_srcs[k]
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

def exec_isams(t:Threads, insts:tuple[Cat5, ...]): # out of bounds reads the zero border color
  i, tex = insts[0], insts[0].tex
  if i.s2en:
    if len(idx := np.unique(t.h[i.src3][t.mask])) != 1: raise i.error()
    tex = int(idx[0])
  if tex >= len(t.d.textures): raise RuntimeError(f"pc {i.pc}: texture {tex} is not bound")
  if not i.s2en and (i.samp >= len(t.d.samplers) or not t.d.samplers[i.samp]): raise RuntimeError(f"pc {i.pc}: unsupported sampler {i.samp}")
  img, dt, channels = t.d.textures[tex], TYPES[i.type], wrmask_channels(i.wrmask)
  if len(channels) == 0: return
  if isinstance(t.h, np.ndarray) and img.dtype in TEX_DTYPES and dt in TEX_DTYPES and all(j.dst + len(channels) <= A0 for j in insts):
    recs = np.array([(j.src1, j.dst) for j in insts], np.int32)
    tmp = np.empty(len(channels) * t.r.shape[1], dt)
    isam_kernel(img.dtype, dt, i.wrmask)(t.r.ctypes.data, t.mask.ctypes.data, img.addr, recs.ctypes.data, tmp.ctypes.data, t.h.ctypes.data,
                                         n=t.r.shape[1], count=len(recs), width=img.width, height=img.height, pitch=img.pitch // img.dtype.itemsize)
    return
  for j in insts:
    mem, ok, off = texels(img, t.r[j.src1], t.r[j.src1 + 1])
    values = mem.view(img.dtype)[off[:, None] // img.dtype.itemsize + channels]
    values[~ok] = 0
    values = values.astype(dt, copy=False)
    for n in range(len(channels)): t.write(j.dst + n, dt == np.float16, values[:, n])

def exec_isam(t:Threads, i:Cat5, k:int): exec_isams(t, (i,))

def exec_ibo(t:Threads, i:Ibo, k:int): # out of bounds stores are dropped
  if i.ssbo >= len(t.d.ibos): raise RuntimeError(f"pc {i.pc}: IBO {i.ssbo} is not bound")
  img, dt = t.d.ibos[i.ssbo], TYPES[i.type]
  mem, ok, off = texels(img, t.r[i.coord], t.r[i.coord + 1])
  lanes, elems, regs = t.mask & ok, mem.view(img.dtype), t.h if dt == np.float16 else t.r
  for c in range(i.ncomp): elems[off[lanes] // img.dtype.itemsize + c] = regs[i.val + c].view(dt)[lanes].astype(img.dtype)

BRANCHES = {mesa.OPC_JUMP, mesa.OPC_PREDT, mesa.OPC_PREDF, mesa.OPC_BR, mesa.OPC_BRAO, mesa.OPC_BRAA}
JUMPS = BRANCHES | {mesa.OPC_CALL}

def taken(t:Threads, i:Cat0) -> np.ndarray:
  if i.op == mesa.OPC_JUMP: return t.everyone
  if i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF): return (t.r[P0] != 0) == (i.op == mesa.OPC_PREDF)
  cond = (t.r[P0 + i.comp1] != 0) ^ bool(i.inv1)
  if i.op == mesa.OPC_BRAO: cond |= (t.r[P0 + i.comp2] != 0) ^ bool(i.inv2)
  if i.op == mesa.OPC_BRAA: cond &= (t.r[P0 + i.comp2] != 0) ^ bool(i.inv2)
  return cond

def is_nop(i:Inst|NotImplementedError) -> bool: return isinstance(i, Cat0) and i.op == mesa.OPC_NOP

@functools.cache
def block_starts(image:bytes, entry:int) -> set[int]: return {entry} | {i.target for i in decode(image) if isinstance(i, Cat0) and i.op in JUMPS}

def straight_runs(prog:list, starts:set[int], key:Callable) -> list[tuple[int, int, list]]:
  runs, pc = [], 0
  while pc < len(prog):
    if (k := key(prog[pc])) is None:
      pc += 1
      continue
    start, insts = pc, [prog[pc]]
    pc += 1
    while pc < len(prog) and pc not in starts and (is_nop(prog[pc]) or key(prog[pc]) == k):
      if not is_nop(prog[pc]): insts.append(prog[pc])
      pc += 1
    runs.append((start, pc, insts))
  return runs

MAX_BLOCK_LANES = 2**23 # reg * n + lane stays in int32
MAX_RECORDS, RECORD_LEN = 2**16, 11
U32, LO16 = 2**32 - 1, 2**16 - 1
def f32_bits(x:float) -> int: return int(np.float32(x).view(np.uint32))
SIGN, INF, TINY, NAN = f32_bits(-0.0), f32_bits(np.inf), f32_bits(float(np.finfo(np.float32).tiny)), f32_bits(np.nan)

def lane_params() -> tuple[UOp, UOp, UOp]:
  r = UOp.param(0, dtypes.uint32, NREGS * MAX_BLOCK_LANES, name="r")
  mask = UOp.param(1, dtypes.uint8, MAX_BLOCK_LANES, name="mask")
  return r, mask, UOp.variable("n", 1, MAX_BLOCK_LANES)

def cpu_kernel(body:UOp, name:str) -> Callable[..., None]:
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    prg = to_program(UOp.sink(body, arg=KernelInfo(name=name)), Device['CPU'].renderer)
  runtime, globals_, var_names = get_runtime('CPU', prg), prg.arg.globals, [v.expr for v in prg.arg.vars]
  # called directly like the amd emulator, the runtime wrapper costs more than a small kernel
  fxn = ctypes.CFUNCTYPE(None, *[ctypes.c_uint64] * (len(globals_) + len(var_names)))(runtime.addr)
  return lambda *bufs, **vals: fxn(*[bufs[g] for g in globals_], *[vals[v] for v in var_names])

def blend(active:UOp, new:UOp, old:UOp) -> UOp: # bitwise, since a where becomes a masked store, which is microcoded on x86
  keep = active.cast(new.dtype) * new.dtype.max
  return (new & keep) | (old & (keep ^ new.dtype.max))

def ftz_f32(f:UOp) -> UOp:
  u = f.bitcast(dtypes.uint32)
  return ((u & (SIGN - 1)) < TINY).where(u & SIGN, u).bitcast(dtypes.float32)

def canonical_nan_f32(f:UOp) -> UOp:
  u = f.bitcast(dtypes.uint32)
  return ((u & (SIGN - 1)) > INF).where(u.const_like(NAN), u)

FMAD, IMAD, SHL, SHR, ASHR, AND, OR, XOR, SEL = range(9)
FLOAT_OPS, INT_OPS = (FMAD,), (IMAD, SHL, SHR, ASHR, AND, OR, XOR, SEL)
ONE_F, NEG_ZERO, ONE, ZERO = (Src("i", v) for v in (f32_bits(1.0), SIGN, 1, 0))
NEG_ONE = ALL_ONES = Src("i", U32)
LO, HI = LO16, U32 ^ LO16
# opcode -> (kernel op, kind, sources)
# add, mul and mov become a * b + c with a constant operand: x * 1, x + -0.0 and x * -1 + y keep the exact bits
# kind says how a source's abs/neg modifiers apply: f flips float sign bits, b is a bitwise not, u and i have none
BLOCK_OPS:dict[int, tuple[int, str, tuple]] = {
  mesa.OPC_MAD_F32: (FMAD, "f", (0, 1, 2)), mesa.OPC_ADD_F: (FMAD, "f", (0, ONE_F, 1)), mesa.OPC_MUL_F: (FMAD, "f", (0, 1, NEG_ZERO)),
  mesa.OPC_MOV: (IMAD, "u", (0, ONE, ZERO)), mesa.OPC_ADD_U: (IMAD, "u", (0, ONE, 1)), mesa.OPC_ADD_S: (IMAD, "i", (0, ONE, 1)),
  mesa.OPC_SUB_U: (IMAD, "u", (1, NEG_ONE, 0)), mesa.OPC_MULL_U: (IMAD, "u", ((0, LO), (1, LO), ZERO)),
  mesa.OPC_MADSH_M16: (IMAD, "u", ((0, LO), (1, HI), 2)), mesa.OPC_SHL_B: (SHL, "u", (0, 1, ZERO)), mesa.OPC_SHR_B: (SHR, "u", (0, 1, ZERO)),
  mesa.OPC_ASHR_B: (ASHR, "i", (0, 1, ZERO)), mesa.OPC_AND_B: (AND, "b", (0, 1, ZERO)), mesa.OPC_OR_B: (OR, "b", (0, 1, ZERO)),
  mesa.OPC_XOR_B: (XOR, "b", (0, 1, ZERO)), mesa.OPC_NOT_B: (XOR, "b", (0, ALL_ONES, ZERO)), mesa.OPC_SEL_B32: (SEL, "u", (0, 1, 2))}

def block_source(spec:int|Src|tuple, srcs:list[Src], kind:str) -> tuple[int, int, int]|None: # (row, and, xor) applied as (r[row] & and) ^ xor
  idx, mask = spec if isinstance(spec, tuple) else (spec, U32)
  s = srcs[idx] if isinstance(idx, int) else idx
  if s.half or s.kind not in ("r", "i") or (s.kind == "r" and s.val >= A0): return None
  if kind == "f": a, x = SIGN - 1 if s.absneg & 2 else U32, SIGN if s.absneg & 1 else 0
  elif kind == "b": a, x = U32, U32 if s.absneg & 1 else 0
  elif s.absneg & (1 if kind == "u" else 3): return None
  else: a, x = U32, 0
  if s.kind == "i": return 0, 0, (s.val & a & mask) ^ x
  return s.val, a & mask, x

@functools.cache
def block_records(i:Inst|NotImplementedError) -> tuple[tuple[int, ...], ...]|None:
  if not isinstance(i, (Cat1, Cat2, Cat3)) or i.op not in BLOCK_OPS or i.dst + i.iterations > A0: return None
  if isinstance(i, Cat1) and not (i.src_type == i.dst_type and i.src_type in (1, 3, 5)): return None
  if isinstance(i, (Cat2, Cat3)) and (i.sat or i.dst_conv or i.dst_half or getattr(i, "ei", 0)): return None
  op, kind, spec = BLOCK_OPS[i.op]
  recs = []
  for k, srcs in enumerate(i.repeat_srcs):
    sources = [block_source(x, srcs, kind) for x in spec]
    if None in sources: return None
    rows, ands, xors = zip(*cast(list[tuple[int, int, int]], sources))
    recs.append((op, i.dst + k, *rows, *ands, *xors))
  return tuple(recs)

@dataclass(frozen=True)
class AluBlock:
  end:int
  records:np.ndarray # RECORD_LEN uint32 per instruction: op, dst, 3 source rows, 3 and masks, 3 xors
  kernel_ops:tuple[int, ...]

@functools.cache
def alu_blocks(image:bytes, entry:int) -> dict[int, AluBlock]:
  blocks = {}
  for start, end, insts in straight_runs(decode(image), block_starts(image, entry), lambda i: True if block_records(i) is not None else None):
    recs = [rec for i in insts for rec in block_records(i) or ()]
    assert len(recs) < MAX_RECORDS, f"pc {start}: alu block too long"
    used = {rec[0] for rec in recs}
    kernel_ops = FLOAT_OPS if used <= set(FLOAT_OPS) else INT_OPS if used <= set(INT_OPS) else FLOAT_OPS + INT_OPS
    blocks[start] = AluBlock(end, np.array(recs, np.uint32), kernel_ops)
  return blocks

@functools.cache
def alu_kernel(kernel_ops:tuple[int, ...]) -> Callable[..., None]:
  r, mask, n = lane_params()
  recs = UOp.param(2, dtypes.uint32, RECORD_LEN * MAX_RECORDS, name="recs")
  tmp = UOp.param(3, dtypes.uint32, MAX_BLOCK_LANES, name="tmp")
  count = UOp.variable("count", 1, MAX_RECORDS)
  k, lane, lane2 = UOp.range(count, 0), UOp.range(n, 1), UOp.range(n, 2)
  rec = [recs.index(k * RECORD_LEN + j).load() for j in range(RECORD_LEN)]
  op, dst, rows, ands, xors = rec[0], rec[1], rec[2:5], rec[5:8], rec[8:11]
  a, b, c = [(r.index(row.cast(dtypes.int32) * n + lane).load() & m) ^ x for row, m, x in zip(rows, ands, xors)]
  fa, fb, fc = [ftz_f32(x.bitcast(dtypes.float32)) for x in (a, b, c)]
  amt = b & 31
  outs = {FMAD: canonical_nan_f32(ftz_f32(ftz_f32(fa * fb) + fc)),
          IMAD: a * b + c, SHL: a << amt, SHR: a >> amt, ASHR: (a.bitcast(dtypes.int32) >> amt.bitcast(dtypes.int32)).bitcast(dtypes.uint32),
          AND: a & b, OR: a | b, XOR: a ^ b, SEL: b.ne(0).where(a, c)}
  out = outs[kernel_ops[0]]
  for o in kernel_ops[1:]: out = op.eq(o).where(outs[o], out)
  # through a scratch row: a destination that is also a source (r = a * b + r) would fail clang's overlap check and run scalar
  computed = tmp.index(lane).store(out).end(lane)
  reg = r.index(dst.cast(dtypes.int32) * n + lane2)
  store = reg.store(blend(mask.index(lane2).load(), tmp.after(computed).index(lane2).load(), reg.load()))
  return cpu_kernel(store.end(lane2).end(k), "qcom_alu_" + "_".join(map(str, kernel_ops)))

def exec_block(t:Threads, b:AluBlock):
  tmp = np.empty(t.r.shape[1], np.uint32)
  alu_kernel(b.kernel_ops)(t.r.ctypes.data, t.mask.ctypes.data, b.records.ctypes.data, tmp.ctypes.data, n=len(tmp), count=len(b.records))

TEX_DTYPES = {np.dtype(np.float16): dtypes.half, np.dtype(np.float32): dtypes.float32}

@functools.cache
def wrmask_channels(wrmask:int) -> np.ndarray: return np.array([c for c in range(4) if wrmask >> c & 1], np.int32)

@dataclass(frozen=True)
class IsamRun:
  end:int
  insts:tuple[Cat5, ...]

@functools.cache
def isam_runs(image:bytes, entry:int) -> dict[int, IsamRun]:
  def key(i) -> tuple|None: return (i.tex, i.samp, i.type, i.wrmask) if isinstance(i, Cat5) and not i.s2en else None
  runs = {}
  for start, end, insts in straight_runs(decode(image), block_starts(image, entry), key):
    assert len(insts) < MAX_RECORDS, f"pc {start}: isam run too long"
    if len(insts) > 1: runs[start] = IsamRun(end, tuple(insts))
  return runs

@functools.cache
def isam_kernel(img_dt:np.dtype, dt:np.dtype, wrmask:int) -> Callable[..., None]:
  r, mask, n = lane_params()
  bits = dtypes.uint32 if dt == np.float32 else dtypes.uint16
  img = UOp.param(2, TEX_DTYPES[img_dt], 2**30, name="img")
  recs = UOp.param(3, dtypes.int32, 2 * MAX_RECORDS, name="recs")
  tmp = UOp.param(4, bits, 4 * MAX_BLOCK_LANES, name="tmp")
  out = r if dt == np.float32 else UOp.param(5, bits, NREGS * MAX_BLOCK_LANES, name="half")
  count = UOp.variable("count", 1, MAX_RECORDS)
  width, height, pitch = [UOp.variable(v, 0, 2**24) for v in ("width", "height", "pitch")]
  k, lane, lane2 = UOp.range(count, 0), UOp.range(n, 1), UOp.range(n, 2)
  src1, dst = recs.index(k * 2).load(), recs.index(k * 2 + 1).load()
  x, y = r.index(src1 * n + lane).load(), r.index((src1 + 1) * n + lane).load()
  inside = (x < width.cast(dtypes.uint32)) & (y < height.cast(dtypes.uint32))
  base = y.cast(dtypes.int32) * pitch + x.cast(dtypes.int32) * 4
  gathered = []
  for j, c in enumerate(wrmask_channels(wrmask).tolist()):
    texel = img.index(inside.where(base + c, 0)).load() # every lane loads, so clamp into the image
    gathered.append(tmp.index(j * n + lane).store(inside.where(texel, 0).cast(TEX_DTYPES[dt]).bitcast(bits)))
  done, active = UOp.group(*gathered).end(lane), mask.index(lane2).load()
  stores = []
  for j in range(len(gathered)):
    reg = out.index((dst + j) * n + lane2)
    stores.append(reg.store(blend(active, tmp.after(done).index(j * n + lane2).load(), reg.load())))
  return cpu_kernel(UOp.group(*stores).end(lane2).end(k), "qcom_isam")

EXEC:dict[type, Callable] = {Cat1: exec_mov, Cat2: exec_alu, Cat3: exec_alu, Cat4: exec_alu, Cat5: exec_isam, Ibo: exec_ibo, Ldg: exec_mem,
                             Stg: exec_mem, Load: exec_mem, Store: exec_mem}

def run(d:Dispatch):
  n_local, n_groups = math.prod(d.local_size), math.prod(d.groups)
  tid = np.arange(n_local * n_groups)
  lid, gid = tid % n_local, tid // n_local
  prog = decode(d.image)
  blocks:dict[int, AluBlock|IsamRun] = {**isam_runs(d.image, d.entry), **alu_blocks(d.image, d.entry)}
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
      nxt:int|None = None
      if (b := blocks.get(cur)) is not None:
        if isinstance(b, AluBlock): exec_block(t, b)
        else: exec_isams(t, b.insts)
        nxt = b.end
      elif (fn := EXEC.get(type(i))) is not None:
        for k in range(i.iterations): fn(t, i, k)
        nxt = cur + 1
      elif together is not None and is_nop(i): nxt = cur + 1
      elif together is not None and isinstance(i, Cat0) and i.op in BRANCHES and ((go := taken(t, i)).all() or not go.any()):
        nxt = i.target if go.all() else cur + 1
      if nxt is not None:
        if together is not None: together = nxt
        else: pc[t.mask] = nxt
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
        elif i.op in BRANCHES: pc[t.mask & taken(t, i)] = i.target
        elif i.op == mesa.OPC_CALL:
          if (depth[t.mask] >= len(calls)).any(): raise RuntimeError(f"pc {cur}: call stack overflow")
          calls[depth[t.mask], np.nonzero(t.mask)[0]], pc[t.mask] = cur + 1, i.target
          depth[t.mask] += 1
        elif i.op == mesa.OPC_RET:
          if (depth[t.mask] == 0).any(): raise RuntimeError(f"pc {cur}: ret without call")
          depth[t.mask] -= 1
          pc[t.mask] = calls[depth[t.mask], np.nonzero(t.mask)[0]]
      if not done.any() and not blocked.any() and (pc == pc[0]).all(): together = int(pc[0])
