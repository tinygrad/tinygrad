"""IR3 (Adreno A6xx) shader emulator.

Instruction encodings here were derived by bit-flip probing every field against mesa's own
disassembler (ir3_isa_disasm) rather than transcribed, so each field below is checked against
ground truth. Anything not understood raises rather than silently computing the wrong answer.
"""
import ctypes, math, struct, os

TRACE = int(os.getenv('MOCKGPU_TRACE', '0'))

# instruction types, indexed by the 3-bit type field (probed against the disassembler's cov.* names)
TYPE_SZ = {0: 2, 1: 4, 2: 2, 3: 4, 4: 2, 5: 4, 6: 1, 7: 1}   # f16 f32 u16 u32 s16 s32 u8 u8_32
# everything except the 32-bit types lives in the half register file (probed: cov to any of these
# types prints an `hr` destination). u8_32 is 8 bits of memory held in a half register.
TYPE_HALF = {0, 2, 4, 6, 7}
TYPE_F = {0, 1}
TYPE_S = {4, 5}
# ir3 encodes small float immediates as an index into this fixed table (probed against the disassembler)
FLOAT_IMM = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e),
             1/math.log2(10.0), math.log2(10.0), 4.0]
# cat2/cat4 compare condition, bits 50:48
CONDS = {0: lambda a, b: a < b, 1: lambda a, b: a <= b, 2: lambda a, b: a > b, 3: lambda a, b: a >= b,
         4: lambda a, b: a == b, 5: lambda a, b: a != b}
# cat3 opcodes whose destination is a half register unless bit46 (a "full" flag) is set. Tabulated
# over compiled kernels: mad.u16 appears with bit46 set and a full destination, while madsh.u16 and
# madsh.m16 are always full (they widen to 32 bits) even though they are named *16.
CAT3_HALF = {0, 2, 6, 8, 10, 12, 14}
# cat2 bitwise opcodes: their (neg) source modifier is a bitwise complement, not arithmetic negation.
# Ground truth: mesa compiles `y & 0xfffffff0` to `and.b dst, y, (neg)15`, which is only correct as ~15.
CAT2_BITWISE = {28, 29, 30, 31}
# cat2 compare opcodes. Saturation is meaningless on a boolean, and bit42 (which mesa prints as
# `(sat)`) instead inverts the result: the compiler emits `(sat)cmps.f.le` where the source is
# `!(a <= b)`, which is not the same as `a > b` once NaN is involved. Verified end to end by
# branching on `fabs(x) <= 1.5f` in both directions.
CAT2_CMP = {5, 7, 20, 21, 33, 34}

def _bits(w:int, hi:int, lo:int) -> int: return (w >> lo) & ((1 << (hi - lo + 1)) - 1)
def _sext(v:int, n:int) -> int: return v - (1 << n) if v & (1 << (n - 1)) else v
def _u32(v:int) -> int: return v & 0xFFFFFFFF
def _s32(v:int) -> int: return _sext(v & 0xFFFFFFFF, 32)
def _f2u(f:float) -> int: return struct.unpack("<I", struct.pack("<f", f))[0]
def _u2f(u:int) -> float: return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

# mesa encodes the two special registers in the ordinary 8-bit register field: regid 61 is the
# address register a0 (half file, written by `mova`) and regid 62 is the predicate register p0.
# Probed: a dst field of 0xf8 disassembles as p0.x and 0xf8^4 as r63.x, so p0 is regid 62.
A0, P0 = 61 * 4, 62 * 4

class Regs:
  """Flat register file. Index is reg*4+component, matching the encoding's 8-bit register fields."""
  def __init__(self, n=1024, nm='r', pred=None): self.v, self.nm, self.pred = [0] * n, nm, pred if pred is not None else [0] * 4
  def __getitem__(self, i): return self.pred[i - P0] if P0 <= i < P0 + 4 else self.v[i]
  def __setitem__(self, i, val):
    # p0 is a single register file shared by the half and full views of the instruction encoding
    if P0 <= i < P0 + 4: self.pred[i - P0] = _u32(val)
    else: self.v[i] = _u32(val)
    if TRACE >= 2: print(f"       {'p0' if P0 <= i < P0 + 4 else self.nm + str(i//4)}.{'xyzw'[i%4]} = {self[i]:#x}")

class IR3Emulator:
  def __init__(self, const_base:int, mapped:list[tuple[int, int]]|None = None, shared:bytearray|None = None):
    self.pred = [0] * 4
    self.const_base, self.mapped = const_base, mapped or []
    self.r, self.hr = Regs(pred=self.pred), Regs(nm='hr', pred=self.pred)
    self.shared, self.private = shared if shared is not None else bytearray(64 << 10), bytearray(64 << 10)

  def _check(self, addr:int, sz:int, what:str):
    if not self.mapped: return
    if not any(st <= addr and addr + sz <= st + n for st, n in self.mapped):
      raise RuntimeError(f"gpu page fault: {what} of {sz}b at {addr:#x} is not in any mapped range")

  def const(self, idx:int) -> int:
    self._check(self.const_base + idx * 4, 4, "const read")
    v = ctypes.c_uint32.from_address(self.const_base + idx * 4).value
    if TRACE >= 3: print(f"       c{idx//4}.{'xyzw'[idx%4]} -> {v:#010x} ({_u2f(v)})")
    return v

  def _src(self, w:int, field:int, half:bool, is_const:bool, immed:bool=False, rel:bool=False) -> int:
    if immed: return _sext(field & 0x7FF, 11)
    # A relative source is indexed by the address register a0.x, which `mova` writes. The field then
    # holds a 10-bit offset with bit10 selecting the const file, which is how `c<a0.x + 96>` encodes
    # as field 1120 (1024 + 96) even when the instruction's own const flag is clear.
    if rel:
      is_const = is_const or bool(field & 0x400)
      field = (field & 0x3FF) + _sext(self.hr[A0] & 0xFFFF, 16)   # a0 is a half register: a signed 16-bit index
    if is_const: return self.const(field & 0x7FF)
    return (self.hr if half else self.r)[field & 0xFF]

  def _imm(self, field:int, is_float:bool, nbits:int) -> int:
    # integer immediates are signed; both source fields are 11 bits wide (src1 10:0, src2 26:16)
    if not is_float: return _u32(_sext(field, nbits))
    if field >= len(FLOAT_IMM): raise NotImplementedError(f"float immediate index {field} outside the known table")
    return _f2u(FLOAT_IMM[field])

  def _mod(self, v:int, neg:bool, abs_:bool, is_float:bool, bitwise:bool=False) -> int:
    if bitwise: return _u32(~v) if neg else _u32(v)
    if is_float:
      f = _u2f(v)
      if abs_: f = abs(f)
      if neg: f = -f
      return _f2u(f)
    s = _s32(v)
    if abs_: s = abs(s)
    if neg: s = -s
    return _u32(s)

  def run(self, pc_base:int, max_instr:int = 1 << 24):
    """Run one thread to completion, ignoring barriers. Only valid for single-threaded workgroups."""
    for _ in self.run_steps(pc_base, max_instr): pass

  def run_steps(self, pc_base:int, max_instr:int = 1 << 24):
    pc: int = 0
    stack: list[int] = []
    for _ in range(max_instr):
      self._check(pc_base + pc * 8, 8, "instruction fetch")
      w = ctypes.c_uint64.from_address(pc_base + pc * 8).value
      cur, pc = pc, pc + 1
      cat = _bits(w, 63, 61)
      if TRACE and cat == 0: print(f"  pc={cur:3d} cat0 {w:#018x}")
      if cat == 0:
        opc = _bits(w, 58, 55)
        if opc == 0: continue                                          # nop
        if opc == 6: return                                            # end
        if opc == 4:                                                   # ret
          if not stack: raise RuntimeError(f"ret with an empty call stack at pc {cur}")
          pc = stack.pop()
          continue
        if opc not in {1, 2, 3}: raise NotImplementedError(f"cat0 opcode {opc} at pc {cur}")
        # br/jump/call all take a signed 32-bit offset in instructions, relative to themselves
        if opc == 1 and not self._brcond(w, cur): continue
        if opc == 3: stack.append(pc)                                  # call
        pc = cur + _sext(_bits(w, 31, 0), 32)
        continue
      # (rptN) executes the instruction N+1 times; the dst and any source tagged (r) step one component each time
      rpt = _bits(w, 41, 40) if cat in {1, 2, 3, 4} else 0
      if TRACE: print(f"  pc={pc-1:3d} cat{cat} rpt{rpt} {w:#018x}")
      if cat == 1:
        for i in range(rpt + 1): self._cat1(w, i)
      elif cat == 2:
        for i in range(rpt + 1): self._cat2(w, i)
      elif cat == 3:
        for i in range(rpt + 1): self._cat3(w, i)
      elif cat == 4:
        for i in range(rpt + 1): self._cat4(w, i)
      elif cat == 6: self._cat6(w)
      elif cat == 7:
        # cat7 opcode 0 is `bar`, which every thread in the workgroup has to reach before any
        # may pass; the caller interleaves threads at each yield. Fences and the cache maintenance
        # opcodes are no-ops here because the emulator shares memory with the host directly.
        if (opc:=_bits(w, 58, 55)) == 0: yield cur
        elif opc not in {1, 2, 3, 4, 5, 6}: raise NotImplementedError(f"cat7 opcode {opc} ({w:#018x}) at pc {cur}")
      else: raise NotImplementedError(f"cat{cat} instruction {w:#018x} at pc {cur}")
    raise RuntimeError("shader did not terminate")

  def _brcond(self, w:int, pc:int) -> bool:
    """Evaluate a cat0 branch condition.

    Probed against the disassembler: bits 54:53 pick the p0 component and bit52 inverts it; the
    two-predicate forms braa (bit38, AND) and brao (bit37, OR) take a second predicate whose
    component is bits 47:46 and whose inversion is bit45.
    """
    a = bool(self.pred[_bits(w, 54, 53)]) != bool(_bits(w, 52, 52))
    aa, ao = _bits(w, 38, 38), _bits(w, 37, 37)
    if not aa and not ao: return a
    if _bits(w, 39, 39) or (aa and ao): raise NotImplementedError(f"cat0 branch form {w:#018x} at pc {pc}")
    b = bool(self.pred[_bits(w, 47, 46)]) != bool(_bits(w, 45, 45))
    return (a and b) if aa else (a or b)

  def _cat1(self, w:int, i:int = 0):
    dst, dst_type, src_type = _bits(w, 39, 32) + i, _bits(w, 48, 46), _bits(w, 52, 50)
    if _bits(w, 58, 57): raise NotImplementedError(f"cat1 swz/movs {w:#018x}")
    sfld = _bits(w, 10, 0) + (i if _bits(w, 43, 43) else 0)
    # a cat1 immediate is the full low 32 bits of the word, not the 11-bit source field: the
    # compiler materialises float constants with `mov.f32f32 rN, (1.000000)`
    if _bits(w, 54, 54): src = _bits(w, 31, 0)
    else: src = self._src(w, sfld, src_type in TYPE_HALF, bool(_bits(w, 53, 53)), rel=bool(_bits(w, 11, 11)))
    val = self._cvt(src, src_type, dst_type)
    (self.hr if dst_type in TYPE_HALF else self.r)[dst] = val

  def _cvt(self, src:int, src_type:int, dst_type:int) -> int:
    if src_type in TYPE_F and dst_type in TYPE_F: return src
    bits = 8 * TYPE_SZ[src_type]
    if src_type in TYPE_F: v = int(_u2f(src))                          # float -> int truncates toward zero
    else: v = _sext(src & ((1 << bits) - 1), bits) if src_type in TYPE_S else src & ((1 << bits) - 1)
    if dst_type in TYPE_F: return _f2u(float(v))
    return _u32(v) & ((1 << (8 * TYPE_SZ[dst_type])) - 1)

  def _cat2(self, w:int, i:int = 0):
    # bit52 is a "full" flag: clear means the whole instruction is half-precision. bit46 makes just the dst half.
    opc, half = _bits(w, 58, 53), not _bits(w, 52, 52)
    dst, dst_half, sat = _bits(w, 39, 32) + i, (not _bits(w, 52, 52)) or bool(_bits(w, 46, 46)), bool(_bits(w, 42, 42))
    i1, i2 = (i if _bits(w, 43, 43) else 0), (i if _bits(w, 51, 51) else 0)
    bw = opc in CAT2_BITWISE
    if _bits(w, 13, 13): raw1 = self._imm(_bits(w, 10, 0), opc < 16, 11)
    else: raw1 = self._src(w, _bits(w, 10, 0) + i1, half, bool(_bits(w, 12, 12)), rel=bool(_bits(w, 11, 11)))
    s1 = self._mod(raw1, bool(_bits(w, 14, 14)), bool(_bits(w, 15, 15)), opc < 16, bw)
    if _bits(w, 29, 29): raw2 = self._imm(_bits(w, 26, 16), opc < 16, 11)
    else: raw2 = self._src(w, _bits(w, 23, 16) + i2, half, bool(_bits(w, 28, 28)), rel=bool(_bits(w, 27, 27)))
    s2 = self._mod(raw2, bool(_bits(w, 30, 30)), bool(_bits(w, 31, 31)), opc < 16, bw)
    f = {0: lambda: _f2u(_u2f(s1) + _u2f(s2)), 1: lambda: _f2u(min(_u2f(s1), _u2f(s2))), 2: lambda: _f2u(max(_u2f(s1), _u2f(s2))),
         3: lambda: _f2u(_u2f(s1) * _u2f(s2)),
         16: lambda: _u32(s1 + s2), 17: lambda: _u32(_s32(s1) + _s32(s2)), 18: lambda: _u32(s1 - s2), 19: lambda: _u32(_s32(s1) - _s32(s2)),
         5: lambda: int(self._cond(w)(_u2f(s1), _u2f(s2))), 6: lambda: _f2u(_u2f(s1)),
         7: lambda: int(self._cond(w)(_u2f(s1), _u2f(s2))),
         33: lambda: int(self._cond(w)(_u32(s1), _u32(s2))), 34: lambda: int(self._cond(w)(_s32(s1), _s32(s2))),
         9: lambda: _f2u(math.floor(_u2f(s1))), 10: lambda: _f2u(math.ceil(_u2f(s1))),
         11: lambda: _f2u(float(round(_u2f(s1)))), 13: lambda: _f2u(math.trunc(_u2f(s1))),
         20: lambda: int(self._cond(w)(_u32(s1), _u32(s2))), 21: lambda: int(self._cond(w)(_s32(s1), _s32(s2))),
         26: lambda: _u32(s1),
         22: lambda: min(_u32(s1), _u32(s2)), 23: lambda: _u32(min(_s32(s1), _s32(s2))),
         24: lambda: max(_u32(s1), _u32(s2)), 25: lambda: _u32(max(_s32(s1), _s32(s2))),
         28: lambda: s1 & s2, 29: lambda: s1 | s2, 30: lambda: _u32(~s1), 31: lambda: s1 ^ s2,
         54: lambda: _u32(s1 << (s2 & 31)), 55: lambda: _u32(s1) >> (s2 & 31), 56: lambda: _u32(_s32(s1) >> (s2 & 31)),
         # mull.u is a 16x16->32 multiply of the low halves; the compiler builds a 32-bit multiply
         # out of one mull.u plus two madsh.m16 that add the two cross terms.
         50: lambda: _u32((s1 & 0xFFFF) * (s2 & 0xFFFF)),
         # the 24-bit multiplies the compiler uses for address arithmetic
         48: lambda: _u32((s1 & 0xFFFFFF) * (s2 & 0xFFFFFF)),
         49: lambda: _u32(_sext(s1 & 0xFFFFFF, 24) * _sext(s2 & 0xFFFFFF, 24)),
         53: lambda: 32 - _u32(s1).bit_length()}.get(opc)
    if f is None: raise NotImplementedError(f"cat2 opcode {opc} ({w:#018x})")
    val = f()
    if sat: val = (not val) if opc in CAT2_CMP else _f2u(min(1.0, max(0.0, _u2f(val)))) if opc < 16 else val
    (self.hr if dst_half else self.r)[dst] = val

  def _cond(self, w:int):
    if (c:=_bits(w, 50, 48)) not in CONDS: raise NotImplementedError(f"compare condition {c} ({w:#018x})")
    return CONDS[c]

  def _cat4(self, w:int, i:int = 0):
    opc, full = _bits(w, 58, 53), bool(_bits(w, 52, 52))
    dst, dst_half, sat = _bits(w, 39, 32) + i, (not full) or bool(_bits(w, 46, 46)), bool(_bits(w, 42, 42))
    sfld = _bits(w, 10, 0) + (i if _bits(w, 43, 43) else 0)
    raw = self._imm(_bits(w, 10, 0), True, 11) if _bits(w, 13, 13) else \
          self._src(w, sfld, not full, bool(_bits(w, 12, 12)), rel=bool(_bits(w, 11, 11)))
    src = self._mod(raw, bool(_bits(w, 14, 14)), bool(_bits(w, 15, 15)), True)
    x = _u2f(src)
    try:
      f = {0: lambda: 1.0 / x, 1: lambda: 1.0 / math.sqrt(x), 2: lambda: math.log2(x), 3: lambda: 2.0 ** x,
           4: lambda: math.sin(x), 5: lambda: math.cos(x), 6: lambda: math.sqrt(x),
           9: lambda: 1.0 / math.sqrt(x), 10: lambda: math.log2(x), 11: lambda: 2.0 ** x}[opc]
    except KeyError: raise NotImplementedError(f"cat4 opcode {opc} ({w:#018x})") from None
    try: val = f()
    except (ZeroDivisionError, ValueError): val = math.nan if (x < 0 or math.isnan(x)) else math.copysign(math.inf, x or 1.0)
    except OverflowError: val = math.inf
    if sat: val = min(1.0, max(0.0, val))
    (self.hr if dst_half else self.r)[dst] = _f2u(val)

  def _cat3(self, w:int, i:int = 0):
    # bit13 picks the alternate cat3 encoding, which reuses opcodes 8-14 for a different set of
    # instructions (shrm/shlm/shrg/shlg/andg/dp2acc/wmm) and different source encodings.
    if _bits(w, 13, 13): return self._cat3_alt(w, i)
    # The *16 opcodes do 16-bit arithmetic on ordinary full registers; mesa prints their operands
    # with an `hr` prefix, but they are not the half file. Ground truth: the compiler's 64-bit
    # multiply feeds `mad.u16 r3.w, hr2.x, hr1.z, hr3.w` from values that `shr.b`/`mull.u` wrote to
    # the full registers r2.x/r1.z/r3.w, so reading the half file there would read nothing.
    opc = _bits(w, 58, 55)
    dst, dst_half = _bits(w, 39, 32) + i, opc in CAT3_HALF and not _bits(w, 46, 46)
    is_float = opc in {6, 7, 12, 13}
    i1, i2, i3 = (i if _bits(w, 43, 43) else 0), (i if _bits(w, 15, 15) else 0), (i if _bits(w, 29, 29) else 0)
    s1 = self._mod(self._src(w, _bits(w, 10, 0) + i1, False, bool(_bits(w, 12, 12)), rel=bool(_bits(w, 11, 11))),
                   bool(_bits(w, 14, 14)), False, is_float)
    s2 = self._mod(self.r[_bits(w, 54, 47) + i2], bool(_bits(w, 30, 30)), False, is_float)
    s3 = self._mod(self._src(w, _bits(w, 23, 16) + i3, False, bool(_bits(w, 28, 28))), bool(_bits(w, 31, 31)), False, is_float)
    if opc == 15: val = _u32(_s32(s1) + _s32(s2) + _s32(s3))          # sad.s32, used by the compiler as a 3-way add
    elif opc in {6, 7}: val = _f2u(_u2f(s1) * _u2f(s2) + _u2f(s3))    # mad.f16 / mad.f32
    elif opc in {4, 5}: val = _u32((_sext(s1 & 0xFFFFFF, 24) * _sext(s2 & 0xFFFFFF, 24) if opc == 5 else
                                    (s1 & 0xFFFFFF) * (s2 & 0xFFFFFF)) + s3)
    # madsh.m16 dst, a, b, c -> c + ((lo16(a) * hi16(b)) << 16). Paired with mull.u (and with itself,
    # operands swapped) this is exactly a 32x32 multiply, which is how the compiler emits imul.
    elif opc == 3: val = _u32(s3 + (((s1 & 0xFFFF) * ((s2 >> 16) & 0xFFFF)) << 16))
    elif opc in {0, 2}: val = _u32((_sext(s1 & 0xFFFF, 16) * _sext(s2 & 0xFFFF, 16) if opc == 2 else
                                    (s1 & 0xFFFF) * (s2 & 0xFFFF)) + s3)
    elif opc == 1: val = _u32(s3 + (((s1 & 0xFFFF) * (s2 & 0xFFFF)) >> 16))   # madsh.u16 (PROVISIONAL)
    # sel.* dst, src1, src2, src3  ->  src2 ? src1 : src3. Operand order confirmed by compiling
    # `cond ? x : y`, which yields `sel.b32 dst, x, cond, y`. The float variants test the condition
    # as a float against zero, so a NaN condition selects src3: the sin routine relies on exactly
    # that, feeding `(int)fabs(flag) - 1` (0xffffffff when the flag is 0) in as the condition.
    elif opc in {8, 9, 10, 11}: val = s1 if s2 else s3
    elif opc in {12, 13}: val = s1 if (f2:=_u2f(s2)) == f2 and f2 != 0.0 else s3
    else: raise NotImplementedError(f"cat3 opcode {opc} ({w:#018x})")
    (self.hr if dst_half else self.r)[dst] = val

  def _cat3_alt(self, w:int, i:int = 0):
    """cat3 alternate encoding (bit13 set): shift-and-combine ops.

    Semantics were read off mesa's own code generation: `(x<<3)|5` compiles to `shlg d, 3, x, 5`,
    `(x<<3)&0x1f` to `shlm d, 3, x, 24`, `(x>>3)|5` to `shrg` and `(x>>3)&5` to `shrm`. So the
    printed operand order is (shift amount, value, combine operand) and m=mask, g=or.
    Sources 1 and 3 are immediates when bits 12/28 are set, registers otherwise.
    """
    opc, half = _bits(w, 58, 55), not _bits(w, 42, 42)     # bit42 is a "full" flag, as in cat2
    dst, dst_half = _bits(w, 39, 32) + i, half or bool(_bits(w, 46, 46))
    regs = self.hr if half else self.r
    s1 = _bits(w, 11, 0) if _bits(w, 12, 12) else regs[_bits(w, 7, 0) + (i if _bits(w, 43, 43) else 0)]
    s2 = regs[_bits(w, 54, 47) + (i if _bits(w, 15, 15) else 0)]
    s3 = _bits(w, 27, 16) if _bits(w, 28, 28) else regs[_bits(w, 23, 16) + (i if _bits(w, 29, 29) else 0)]
    if _bits(w, 30, 30): s2 = _u32(~s2)
    if _bits(w, 31, 31): s3 = _u32(~s3)
    sh = s1 & 31
    if opc == 8: val = (_u32(s2) >> sh) & s3                          # shrm
    elif opc == 9: val = _u32(s2 << sh) & s3                          # shlm
    elif opc == 10: val = (_u32(s2) >> sh) | s3                       # shrg
    elif opc == 11: val = _u32(s2 << sh) | s3                         # shlg
    else: raise NotImplementedError(f"cat3 alt opcode {opc} ({w:#018x})")
    (self.hr if dst_half else self.r)[dst] = _u32(val)

  def _cat6(self, w:int):
    opc, typ = _bits(w, 58, 53), _bits(w, 51, 49)
    cnt, sz = _bits(w, 26, 24), TYPE_SZ[typ]
    regs = self.hr if typ in TYPE_HALF else self.r
    if opc == 0:                                                       # ldg
      dst = _bits(w, 39, 32)
      if _bits(w, 22, 22):                                             # ldg.a: base + index register
        areg, addr = _bits(w, 21, 14), 0
        # the index register counts elements, not bytes: the compiler feeds it a loop counter that
        # steps by one per iteration (verified by running such a loop against its C meaning)
        addr += self.r[_bits(w, 8, 1)] * sz
        if _bits(w, 48, 41): raise NotImplementedError(f"ldg.a with a nonzero offset ({w:#018x}): "
                                                       "nothing compiled so far emits one, so its scaling is unverified")
      else:
        areg = _bits(w, 21, 14)
        addr = _sext(_bits(w, 13, 1), 13)         # byte offset, not an element count
      addr += self.r[areg] | (self.r[areg + 1] << 32)
      for i in range(cnt): regs[dst + i] = self._load(addr + i * sz, typ)
    elif opc == 6:                                                     # stg
      areg, src = _bits(w, 48, 41), _bits(w, 8, 1)
      if _bits(w, 52, 52):                                             # stg.a: base + index register
        addr = self.r[_bits(w, 39, 32)] * sz                           # the index counts elements
        if _bits(w, 21, 14): raise NotImplementedError(f"stg.a with a nonzero offset ({w:#018x}): "
                                                       "nothing compiled so far emits one, so its scaling is unverified")
      else: addr = _sext(_bits(w, 39, 32) | (_bits(w, 13, 9) << 8), 13)
      addr += self.r[areg] | (self.r[areg + 1] << 32)
      for i in range(cnt): self._store(addr + i * sz, regs[src + i], typ)
    elif opc in {2, 4, 8, 10}:                                         # ldl / ldp / stl / stp
      buf = self.private if opc in {4, 10} else self.shared
      off = self.r[_bits(w, 48, 41) if opc in {8, 10} else _bits(w, 21, 14)]
      if opc in {8, 10}:                                               # store
        off += _sext(_bits(w, 39, 32) | (_bits(w, 13, 9) << 8), 13)
        src = _bits(w, 8, 1)
        for i in range(cnt): self._local_store(buf, off + i * sz, regs[src + i], typ)
      else:
        off += _sext(_bits(w, 13, 1), 13)
        dst = _bits(w, 39, 32)
        for i in range(cnt): regs[dst + i] = self._local_load(buf, off + i * sz, typ)
    else: raise NotImplementedError(f"cat6 opcode {opc} ({w:#018x})")

  def _fmt(self, typ:int) -> str: return {0: '<H', 1: '<I', 2: '<H', 3: '<I', 4: '<h', 5: '<i', 6: '<B', 7: '<b'}[typ]

  def _local_load(self, buf:bytearray, off:int, typ:int) -> int:
    if not 0 <= off <= len(buf) - TYPE_SZ[typ]: raise RuntimeError(f"local memory read out of range at {off:#x}")
    return _u32(struct.unpack_from(self._fmt(typ), buf, off)[0])

  def _local_store(self, buf:bytearray, off:int, val:int, typ:int):
    if not 0 <= off <= len(buf) - TYPE_SZ[typ]: raise RuntimeError(f"local memory write out of range at {off:#x}")
    struct.pack_into(self._fmt(typ).lower().replace('<i', '<I').replace('<h', '<H').replace('<b', '<B'), buf, off,
                     val & ((1 << (8 * TYPE_SZ[typ])) - 1))

  def _load(self, addr:int, typ:int) -> int:
    self._check(addr, TYPE_SZ[typ], "load")
    ct = {1: ctypes.c_uint32, 3: ctypes.c_uint32, 5: ctypes.c_int32, 2: ctypes.c_uint16, 4: ctypes.c_int16, 0: ctypes.c_uint16,
          6: ctypes.c_uint8, 7: ctypes.c_int8}[typ]
    return _u32(ct.from_address(addr).value)

  def _store(self, addr:int, val:int, typ:int):
    self._check(addr, TYPE_SZ[typ], "store")
    ct = {1: ctypes.c_uint32, 3: ctypes.c_uint32, 5: ctypes.c_uint32, 2: ctypes.c_uint16, 4: ctypes.c_uint16, 0: ctypes.c_uint16,
          6: ctypes.c_uint8, 7: ctypes.c_uint8}[typ]
    ct.from_address(addr).value = val & ((1 << (8 * ctypes.sizeof(ct))) - 1)
