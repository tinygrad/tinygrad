from __future__ import annotations
import ctypes, math, os, struct, time
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU

_FLOAT_IMMS = (0.0, .5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0)

def _float_imm(idx:int) -> float:
  # The special-immediate source field is 10 bits wide, but only the low
  # indices are defined.  An index past the table is an unrecognized encoding,
  # not a value, so report it like the other unsupported encodings instead of
  # raising a bare IndexError.
  if idx >= len(_FLOAT_IMMS): raise NotImplementedError(f"A630 float immediate index {idx:#x}")
  return _FLOAT_IMMS[idx]

def _u32(x:int) -> int: return x & 0xffffffff
def _s32(x:int) -> int: return (x & 0xffffffff) - (1 << 32) if x & (1 << 31) else x & 0xffffffff
def _sext(x:int, bits:int) -> int: return x - (1 << bits) if x & (1 << (bits - 1)) else x
def _cat2_signed(x:int, full:bool) -> int: return _s32(x) if full else _sext(x & 0xffff, 16)
def _clz_b(x:int, full:bool) -> int:
  bits = 32 if full else 16
  x &= (1 << bits) - 1
  # IR3 CLZ_B uses all-ones as the zero-input sentinel.  Mesa's 64-bit
  # division lowering relies on this (for example, 31 - clz(0) wraps to 32).
  return 0xffffffff if x == 0 else bits - x.bit_length()
def _cat0_branch_taken(ins:int, gpr:list[int]) -> bool:
  # A6xx CAT0 branch predicates live in p0.x..p0.w (r248..r251).  BR uses
  # COMP1/INV1; BRAO and BRAA combine a second predicate with OR/AND.
  kind = (ins >> 37) & 7
  inv1, comp1 = bool((ins >> 52) & 1), (ins >> 53) & 3
  p1 = bool(gpr[0xf8 + comp1]) ^ inv1
  if kind == 0: return p1
  inv2, comp2 = bool((ins >> 45) & 1), (ins >> 46) & 3
  p2 = bool(gpr[0xf8 + comp2]) ^ inv2
  if kind == 1: return p1 or p2
  if kind == 2: return p1 and p2
  raise NotImplementedError(f"A630 CAT0 branch kind {kind}")
def _cat1_swz(ins:int, sf:list[int], df:list[int]) -> None:
  s0, s1, d0, d1 = ins & 0xff, (ins >> 8) & 0xff, (ins >> 32) & 0xff, (ins >> 16) & 0xff
  v0, v1 = sf[s0], sf[s1]  # SWZ is parallel: source/destination ranges may overlap.
  df[d0], df[d1] = _u32(v0), _u32(v1)
def _cat6_private_offset(ins:int, store:bool) -> int:
  raw = ((((ins >> 9)&0x1f)<<8) | ((ins >> 32)&0xff)) if store else ((ins >> 1)&0x1fff)
  return _sext(raw, 13)
def _mul_s24(a:int, b:int) -> int: return _u32(_sext(a & 0xffffff, 24) * _sext(b & 0xffffff, 24))
def _mull_u(a:int, b:int) -> int: return (a & 0xffff) * (b & 0xffff)
def _madsh_m16(a:int, b:int, c:int) -> int: return _u32((((a & 0xffff) * ((b >> 16) & 0xffff)) << 16) + c)
def _absneg_s(x:int, flags:int, bits:int=32) -> int:
  mask = (1 << bits) - 1
  sx = _sext(x & mask, bits)
  if flags & 2: sx = abs(sx)
  if flags & 1: sx = -sx
  return sx & mask
def _cov_src(src:int, src_type:int, dst_type:int):
  # A6xx CAT1 has no distinct signed-8 source type in this encoding.  Mesa
  # emits TYPE_U8 for 8-bit values and the destination type determines whether
  # the byte is sign-extended (signed/float destinations) or zero-extended.
  if src_type == mesa.TYPE_U8 and dst_type in (mesa.TYPE_S16, mesa.TYPE_S32, mesa.TYPE_F16, mesa.TYPE_F32):
    return _sext(src & 0xff, 8)
  return (_f16(src), _f32(src), src & 0xffff, src, _sext(src & 0xffff,16), _s32(src), src & 0xff, src & 0xff)[src_type]
def _f32(x:int) -> float: return struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0]
def _f32bits(x:float) -> int:
  try: return struct.unpack('<I', struct.pack('<f', float(x)))[0]
  except OverflowError: return 0x7f800000 if x >= 0 else 0xff800000
def _f16(x:int) -> float: return struct.unpack('<e', struct.pack('<H', x & 0xffff))[0]
def _f16bits(x:float) -> int:
  try: return struct.unpack('<H', struct.pack('<e', float(x)))[0]
  except OverflowError: return 0x7c00 if x >= 0 else 0xfc00
def _rcp(x:float) -> float:
  if math.isnan(x): return math.nan
  if x == 0.0: return math.copysign(math.inf, x)
  return 1.0/x
def _rsqrt(x:float) -> float:
  if math.isnan(x) or x < 0.0: return math.nan
  if x == 0.0: return math.copysign(math.inf, x)
  return 1.0/math.sqrt(x)
def _log2(x:float) -> float:
  if math.isnan(x): return math.nan
  if x < 0.0: return math.nan
  if x == 0.0: return -math.inf
  return math.log2(x)
def _exp2(x:float) -> float:
  if math.isnan(x): return math.nan
  if x == math.inf: return math.inf
  if x == -math.inf: return 0.0
  try: return 2.0**x
  except OverflowError: return math.inf
def _sin(x:float) -> float:
  if not math.isfinite(x): return math.nan
  return math.sin(x)
def _cos(x:float) -> float:
  if not math.isfinite(x): return math.nan
  return math.cos(x)
def _sqrt(x:float) -> float:
  if math.isnan(x) or x < 0.0: return math.nan
  return math.sqrt(x)
def _trunc_f(x:float) -> float:
  # IR3 trunc.f truncates toward zero while preserving IEEE non-finite values.
  return x if not math.isfinite(x) else float(math.trunc(x))
def _floor_f(x:float) -> float:
  # Preserve IEEE non-finite values and signed zero; finite nonzero values use
  # mathematical floor (unlike trunc.f, negative fractions round downward).
  return x if not math.isfinite(x) or x == 0.0 else float(math.floor(x))
def _sign_f(x:float) -> float:
  # Matches Mesa IR3 lowering: negative -> -1, zero -> 0, otherwise -> +1.
  # NaN therefore maps to +1 because both (x < 0) and (x == 0) are false.
  return -1.0 if x < 0.0 else 0.0 if x == 0.0 else 1.0
def _sat_f(x:float) -> float:
  x = 0.0 if x < 0.0 else x
  return x if x < 1.0 else 1.0
def _mad_f16(a:float, b:float, c:float, neg:int=0, sat:bool=False) -> float:
  if neg & 1: a = -a
  if neg & 2: b = -b
  if neg & 4: c = -c
  out = a * b + c
  return _sat_f(out) if sat else out
def _cat4_fn(op:int):
  # A6xx gives rsq/log2/exp2 distinct half-precision opcodes (base+8).
  # rcp/sin/cos/sqrt reuse their base opcode and select precision with FULL.
  fn = {0x0:_rcp, 0x1:_rsqrt, 0x2:_log2, 0x3:_exp2, 0x4:_sin, 0x5:_cos, 0x6:_sqrt,
        0x9:_rsqrt, 0xa:_log2, 0xb:_exp2}
  if op not in fn: raise NotImplementedError(f"A630 cat4 opcode {op:#x}")
  return fn[op]
def _fmod(x:float, enc:int) -> float:
  mod = (enc >> 14) & 3
  return -x if mod == 1 else abs(x) if mod == 2 else -abs(x) if mod == 3 else x

class QCOMGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.regs:dict[int, int] = {}
    self.ranges:list[tuple[int, int]] = []
    self.ibs:list[tuple[int, int]] = []
    self.const_addr, self.const_size = 0, 0
    self.shader_addr, self.shader_size = 0, 0

  def map_range(self, vaddr, size): self.ranges.append((int(vaddr), int(vaddr + size)))
  def unmap_range(self, vaddr, size):
    r = (int(vaddr), int(vaddr + size))
    with __import__('contextlib').suppress(ValueError): self.ranges.remove(r)

  def _check(self, addr:int, size:int):
    if not any(st <= addr and addr + size <= en for st,en in self.ranges):
      nearby = sorted(self.ranges, key=lambda r: min(abs(addr-r[0]), abs(addr-r[1])))[:4]
      raise RuntimeError(f"unmapped A630 address {addr:#x}+{size:#x}, nearest={[(hex(st), hex(en)) for st,en in nearby]}")

  def submit_ib(self, addr:int, size:int): self.ibs.append((addr, size))
  def execute(self):
    while self.ibs:
      addr, size = self.ibs.pop(0)
      self._exec_ib(addr, size)

  def _exec_ib(self, addr:int, size:int):
    self._check(addr, size)
    words = list(to_mv(addr, size).cast('I'))
    pos = 0
    while pos < len(words):
      hdr, pos = words[pos], pos + 1
      typ = hdr >> 28
      if typ == 4:
        base, cnt = (hdr >> 8) & 0x3ffff, hdr & 0x7f
        if os.getenv("QCOM_TRACE"): print(f"CP4 reg={base:#x} cnt={cnt}", flush=True)
        for i in range(cnt): self.regs[base+i] = words[pos+i]
        pos += cnt
      elif typ == 7:
        op, cnt = (hdr >> 16) & 0x7f, hdr & 0x3fff
        p, pos = words[pos:pos+cnt], pos+cnt
        if os.getenv("QCOM_TRACE"): print(f"CP7 op={op:#x} cnt={cnt} vals={[hex(x) for x in p]}", flush=True)
        self._exec7(op, p)
      else: raise RuntimeError(f"invalid A6xx PM4 packet type {typ} header={hdr:#x}")

  def _exec7(self, op:int, p:list[int]):
    if op in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER): return
    if op == mesa.CP_LOAD_STATE6_FRAG:
      state_type, block = (p[0] >> 14) & 3, (p[0] >> 18) & 0xf
      addr, units = p[1] | (p[2] << 32), p[0] >> 22
      if os.getenv("QCOM_TRACE"): print(f"LOAD_STATE block={block} type={state_type} addr={addr:#x} units={units}", flush=True)
      if block == mesa.SB6_CS_SHADER and state_type == mesa.ST_CONSTANTS:
        self.const_addr, self.const_size = addr, units * 16
      elif block == mesa.SB6_CS_SHADER and state_type == mesa.ST_SHADER:
        self.shader_addr, self.shader_size = addr, units * 128
      return
    if op == mesa.CP_EVENT_WRITE:
      if len(p) >= 4:
        addr = p[1] | (p[2] << 32)
        self._check(addr, 4)
        to_mv(addr, 4).cast('I')[0] = p[3]
      return
    if op == mesa.CP_WAIT_REG_MEM:
      fn, addr, ref, mask = p[0] & 7, p[1] | (p[2] << 32), p[3], p[4]
      self._check(addr, 4)
      cur = to_mv(addr, 4).cast('I')[0] & mask
      if fn == mesa.WRITE_GE: ok = cur >= ref
      elif fn == mesa.WRITE_EQ: ok = cur == ref
      elif fn == mesa.WRITE_NE: ok = cur != ref
      elif fn == mesa.WRITE_ALWAYS: ok = True
      else: raise RuntimeError(f"unsupported CP_WAIT_REG_MEM function {fn}")
      if not ok: raise RuntimeError(f"mock QCOM wait unsatisfied: {cur:#x} vs {ref:#x}")
      return
    if op == mesa.CP_REG_TO_MEM:
      addr = p[1] | (p[2] << 32)
      self._check(addr, 8)
      to_mv(addr, 8).cast('Q')[0] = int(time.perf_counter() * 19.2e6)
      return
    if op == mesa.CP_EXEC_CS:
      self._exec_cs((p[1], p[2], p[3]))
      return
    raise RuntimeError(f"unsupported A6xx type7 opcode {op:#x}")

  @staticmethod
  def _src(enc:int, gpr:list[int], hreg:list[int], consts:list[int], full:bool) -> int:
    kind = (enc >> 11) & 7
    if kind == 0: return (gpr if full else hreg)[enc & 0xff]
    if kind in (2, 6):
      idx = enc & 0x7ff
      val = consts[idx] if idx < len(consts) else 0
      # IR3 constant operands address the 32-bit constant file directly.  The
      # decoder may print an h-prefixed constant in mixed half/full integer
      # instructions, but that does not mean selecting a 16-bit half of the
      # constant word here.  Mesa's structured decoder likewise materializes
      # CONST sources as full 32-bit entries.
      return val
    if kind == 4: return _sext(enc & 0x7ff, 11)
    if kind == 5: return _f32bits(_float_imm(enc & 0x3ff))
    raise NotImplementedError(f"A630 multisrc encoding {enc:#x}")

  @staticmethod
  def _float_src(enc:int, gpr:list[int], hreg:list[int], consts:list[int], full:bool) -> float:
    kind = (enc >> 11) & 7
    # Special float immediates denote a value, not a raw 32-bit register word.
    # In half instructions Mesa prints these as h(...); interpreting the low
    # 16 bits of their fp32 representation as fp16 corrupts the value.
    if kind == 5: return _float_imm(enc & 0x3ff)
    val = QCOMGPU._src(enc, gpr, hreg, consts, full)
    # Freedreno keeps half constant-register values as 32-bit floats for
    # floating-point opcodes.  Half GPR/immediate sources remain fp16.
    return _f32(val) if full or kind in (2, 6) else _f16(val)

  @staticmethod
  def _cat3_src(enc:int, regs:list[int], consts:list[int], immediate:bool=False) -> int:
    if (enc >> 8) & 0x1f == 0: return regs[enc & 0xff]
    if enc & 0x1000:
      idx = enc & 0xfff
      # A6xx CAT3 has two source encodings selected by instruction bit 13.
      # Normal CAT3 treats this form as a constant-file index, while CAT3-alt
      # (shrm/shlm/shrg/shlg/andg) treats it as an inline immediate.
      if immediate: return idx
      return consts[idx] if idx < len(consts) else 0
    raise NotImplementedError(f"A630 cat3 source encoding {enc:#x}")

  @staticmethod
  def _cat3_f16_src(enc:int, hreg:list[int], consts:list[int]) -> float:
    raw = QCOMGPU._cat3_src(enc, hreg, consts)
    # As with CAT2 floating half operands, constant-file values are 32-bit
    # floats while half GPR values contain fp16 bits.
    return _f32(raw) if enc & 0x1000 else _f16(raw)

  @staticmethod
  def _cat1_src(ins:int, mode:int, sf:list[int], consts:list[int], si:int) -> int:
    # CAT1 source selection: mode 2 = inline 32-bit immediate, mode 1 = 11-bit
    # constant-file index (out-of-range reads default to 0, matching IR3's
    # constant file behavior), otherwise a register from the selected file.
    if mode == 2: return ins & 0xffffffff
    if mode == 1: return consts[ins & 0x7ff] if (ins & 0x7ff) < len(consts) else 0
    return sf[si]

  def _run_thread(self, local_id:tuple[int,int,int], group_id:tuple[int,int,int], shared:bytearray, shader:bytes, consts:list[int]):
    gpr, hreg, pc = [0]*256, [0]*256, 0
    # A6xx private memory is per work-item.  Keep it separate from workgroup
    # shared memory; cumprod backward is the first broad backend workload that
    # spills enough temporaries for Mesa to emit ldp/stp.
    private = bytearray(32*1024)
    # predt/predf snapshot p0.x when the region begins.  Since this emulator
    # runs one work-item at a time, the A6xx execution mask reduces to a
    # per-thread boolean until prede closes the region.
    pred_mode, pred_value = 0, False
    cfg = self.regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0)
    wgid = (cfg >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT) & 0xff
    lid = (cfg >> mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT) & 0xff
    if wgid < 0xfc: gpr[wgid:wgid+3] = group_id
    if lid < 0xfc: gpr[lid:lid+3] = local_id
    trace_thread = bool(os.getenv("QCOM_TRACE")) and local_id[0] < 3 and local_id[1:] == (0,0) and group_id == (0,0,0)

    while pc * 8 < len(shader):
      ins = struct.unpack_from('<Q', shader, pc*8)[0]
      cat, dst = (ins >> 61) & 7, (ins >> 32) & 0xff
      if cat == 0:
        op = (ins >> 55) & 0x3f
        if op == 1 and _cat0_branch_taken(ins, gpr):
          pc += _s32(ins & 0xffffffff)
          continue
        if op == 2:
          pc += _s32(ins & 0xffffffff)
          continue
        if op == 6: break
        if op == 0xd: pred_mode, pred_value = 1, bool(gpr[0xf8])  # predt
        elif op == 0xe: pred_mode, pred_value = 2, bool(gpr[0xf8])  # predf
        elif op == 0xf: pred_mode = 0  # prede
      elif pred_mode and (pred_value if pred_mode == 1 else not pred_value) is False:
        pc += 1
        continue
      elif cat == 1:
        src_type, dst_type, mode = (ins >> 50) & 7, (ins >> 46) & 7, (ins >> 53) & 3
        sf, df = (hreg if src_type in (0,2,4,6) else gpr), (hreg if dst_type in (0,2,4,6) else gpr)
        # CAT1 SWZ is a two-source/two-destination parallel move.  Bit 58
        # selects the SWZ encoding on A6xx; treating it as an ordinary MOV
        # loses DST1 and is especially destructive when Mesa uses SWZ to swap
        # the two 32-bit halves of a 64-bit pointer.
        if (ins >> 58) & 1:
          if src_type != dst_type: raise NotImplementedError(f"A630 cat1 swz type conversion {src_type}->{dst_type} pc={pc}")
          _cat1_swz(ins, sf, df)
          pc += 1
          continue
        for rpt in range(((ins >> 40) & 3) + 1):
          si = (ins & 0xff) + (rpt if (ins >> 43) & 1 else 0)
          src = self._cat1_src(ins, mode, sf, consts, si)
          if src_type != dst_type:
            val = _cov_src(src, src_type, dst_type)
            cvt = (_f16bits, _f32bits, lambda x:int(x)&0xffff, lambda x:_u32(int(x)), lambda x:int(x)&0xffff,
                   lambda x:_u32(int(x)), lambda x:int(x)&0xff, lambda x:int(x)&0xff)[dst_type]
            src = cvt(val)
          df[dst+rpt] = _u32(src)
      elif cat == 2:
        full, conv, op = bool((ins >> 52) & 1), bool((ins >> 46) & 1), (ins >> 53) & 0x3f
        for rpt in range(((ins >> 40) & 3) + 1):
          ae, be = (ins & 0xffff) + (rpt if (ins >> 43)&1 else 0), ((ins >> 16)&0xffff) + (rpt if (ins >> 51)&1 else 0)
          a, b = self._src(ae,gpr,hreg,consts,full), self._src(be,gpr,hreg,consts,full)
          fa, fb = _fmod(self._float_src(ae,gpr,hreg,consts,full), ae), _fmod(self._float_src(be,gpr,hreg,consts,full), be)
          outf = _f16bits if full == conv and dst <= 0xf7 else _f32bits
          if op == 0x00: out = outf(fa + fb)
          elif op == 0x01: out = outf(min(fa, fb))
          elif op == 0x02: out = outf(max(fa, fb))
          elif op == 0x03: out = outf(fa * fb)
          elif op == 0x04: out = outf(_sign_f(fa))
          elif op == 0x06: out = outf(fa)
          elif op == 0x09: out = outf(_floor_f(fa))
          elif op == 0x0d: out = outf(_trunc_f(fa))
          elif op == 0x05:
            cond = (ins >> 48) & 7
            out = int((fa < fb, fa <= fb, fa > fb, fa >= fb, fa == fb, fa != fb)[cond])
          elif op == 0x10: out = a + b
          elif op in (0x12,0x13): out = a - b
          elif op == 0x14:
            cond = (ins >> 48) & 7
            out = int((a < b, a <= b, a > b, a >= b, a == b, a != b)[cond])
          elif op == 0x15:
            cond, sa, sb = (ins >> 48) & 7, _cat2_signed(a, full), _cat2_signed(b, full)
            out = int((sa < sb, sa <= sb, sa > sb, sa >= sb, sa == sb, sa != sb)[cond])
          elif op == 0x16: out = min(a, b)
          elif op == 0x17: out = _u32(min(_cat2_signed(a, full), _cat2_signed(b, full)))
          elif op == 0x18: out = max(a, b)
          elif op == 0x19: out = _u32(max(_cat2_signed(a, full), _cat2_signed(b, full)))
          elif op == 0x1a: out = _absneg_s(a, (ae >> 14) & 3, 32 if full else 16)
          elif op == 0x1c: out = a & b
          elif op == 0x1d: out = a | b
          elif op == 0x1e: out = ~a
          elif op == 0x1f: out = a ^ b
          elif op == 0x31: out = _mul_s24(a, b)
          elif op == 0x32: out = _mull_u(a, b)
          elif op == 0x35: out = _clz_b(a, full)
          elif op == 0x36: out = a << (b & 31)
          elif op == 0x37: out = a >> (b & 31)
          elif op == 0x38: out = _cat2_signed(a, full) >> (b & 31)
          else: raise NotImplementedError(f"A630 cat2 opcode {op:#x} pc={pc}")
          (hreg if full == conv and dst <= 0xf7 else gpr)[dst+rpt] = _u32(out)
      elif cat == 3:
        op, alt = (ins >> 55) & 0xf, bool((ins >> 13) & 1)
        for rpt in range(((ins >> 40) & 3) + 1):
          ae = (ins & 0x1fff) + (rpt if (ins >> 43) & 1 else 0)
          bi = ((ins >> 47) & 0xff) + (rpt if (ins >> 15) & 1 else 0)
          ce = ((ins >> 16) & 0x1fff) + (rpt if (ins >> 29) & 1 else 0)
          if alt:
            # Mesa's CAT3-alt format (bit 13 set) reuses opcode values 8..c
            # but changes SRC1/SRC3 to the immediate-capable source encoding.
            # FULL (bit 42) describes SRC2 precision; DST_CONV (bit 46) flips
            # the destination precision relative to SRC2.  Half-file shift
            # ops are common in fp16/fp8 lowering, so routing these through
            # the full GPR file corrupts unrelated full registers (including
            # 64-bit pointer high words).
            full, conv = bool((ins >> 42) & 1), bool((ins >> 46) & 1)
            src_regs = gpr if full else hreg
            dst_half = (not full) ^ conv
            dst_regs = hreg if dst_half else gpr
            a = self._cat3_src(ae, src_regs, consts, immediate=True)
            b = src_regs[bi]
            c = self._cat3_src(ce, src_regs, consts, immediate=True)
            if op == 0x8: out = (b >> (a & 31)) & c       # shrm
            elif op == 0x9: out = (b << (a & 31)) & c    # shlm
            elif op == 0xa: out = (b >> (a & 31)) | c    # shrg
            elif op == 0xb: out = (b << (a & 31)) | c    # shlg
            elif op == 0xc: out = (a & b) | c             # andg
            else: raise NotImplementedError(f"A630 cat3-alt opcode {op:#x} pc={pc}")
            dst_regs[dst+rpt] = (_u32(out) & 0xffff) if dst_half else _u32(out)
            continue

          b, c = gpr[bi], self._cat3_src(ce, gpr, consts)
          if op == 0x3:
            a = consts[ae & 0xfff] if ae & 0x1000 and (ae & 0xfff) < len(consts) else gpr[ae & 0xff]
            # madsh.m16 accumulates the cross term formed by SRC1.low16 and
            # SRC2.high16, shifted into the upper half of the 32-bit result.
            # This is used heavily by tinygrad's 64-bit multiply decomposition.
            out = _madsh_m16(a, b, c)
          elif op == 0x6:
            af = self._cat3_f16_src(ae, hreg, consts)
            bf = _f16(hreg[bi])
            cf = self._cat3_f16_src(ce, hreg, consts)
            neg = ((ins >> 14) & 1) | (((ins >> 30) & 1) << 1) | (((ins >> 31) & 1) << 2)
            hreg[dst+rpt] = _f16bits(_mad_f16(af, bf, cf, neg, bool((ins >> 42) & 1)))
            continue
          elif op == 0x7:
            af = _f32(self._cat3_src(ae, gpr, consts))
            bf = _f32(gpr[bi])
            cf = _f32(self._cat3_src(ce, gpr, consts))
            neg = ((ins >> 14) & 1) | (((ins >> 30) & 1) << 1) | (((ins >> 31) & 1) << 2)
            val = _mad_f16(af, bf, cf, neg, bool((ins >> 42) & 1))
            # Normal CAT3 mad.f32 can convert its float32 result to a half-file
            # destination.  Mesa exposes raw bit 46 as DST_HALF for this opcode.
            if (ins >> 46) & 1: hreg[dst+rpt] = _f16bits(val)
            else: gpr[dst+rpt] = _f32bits(val)
            continue
          elif op == 0x8:
            # Normal opcode 8 is sel.b16. All three sources are half-file
            # values (constant sources still address full 32-bit const words,
            # with the destination write selecting the low 16 bits).
            a, b, c = self._cat3_src(ae, hreg, consts), hreg[bi], self._cat3_src(ce, hreg, consts)
            hreg[dst+rpt] = (a if b else c) & 0xffff
            continue
          elif op == 0x9:
            a = self._cat3_src(ae, gpr, consts)
            out = a if b else c
          else: raise NotImplementedError(f"A630 cat3 opcode {op:#x} pc={pc}")
          gpr[dst+rpt] = _u32(out)
      elif cat == 4:
        full, op, enc = bool((ins >> 52)&1), (ins >> 53)&0x3f, ins & 0xffff
        try: fn = _cat4_fn(op)
        except NotImplementedError as e: raise NotImplementedError(f"{e} pc={pc}") from e
        for rpt in range(((ins >> 40) & 3) + 1):
          src_enc = enc + (rpt if (ins >> 43) & 1 else 0)
          x = _fmod(self._float_src(src_enc,gpr,hreg,consts,full), src_enc)
          val = fn(x)
          if (ins >> 42) & 1: val = _sat_f(val)
          (gpr if full else hreg)[dst+rpt] = (_f32bits if full else _f16bits)(val)
      elif cat == 6:
        op, typ, size = (ins >> 54)&0x1f, (ins >> 49)&7, (ins >> 24)&7
        width = 2 if typ in (0,2,4) else 1 if typ == 6 else 4
        if op == 0:
          ar = (ins >> 14)&0xff
          addr = gpr[ar] | (gpr[(ar+1)&0xff] << 32)
          off = _sext((ins >> 1)&0x1fff, 13) * width
          try: self._check(addr+off, size*width)
          except RuntimeError as e:
            raise RuntimeError(f"{e}; pc={pc} cat6=ldg ar={ar} lo={gpr[ar]:#x} hi={gpr[(ar+1)&0xff]:#x} off={off} typ={typ} size={size}") from e
          out_regs = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): out_regs[dst+i] = int.from_bytes(ctypes.string_at(addr+off+i*width, width), 'little')
        elif op == 3:
          src, ar = (ins >> 1)&0xff, (ins >> 41)&0xff
          addr = gpr[ar] | (gpr[(ar+1)&0xff] << 32)
          off = _sext((((ins >> 9)&0x1f)<<8) | ((ins >> 32)&0xff), 13) * width
          try: self._check(addr+off, size*width)
          except RuntimeError as e:
            raise RuntimeError(f"{e}; pc={pc} cat6=stg ar={ar} lo={gpr[ar]:#x} hi={gpr[(ar+1)&0xff]:#x}"
                               f" off={off} typ={typ} size={size} src={src}") from e
          sf = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): ctypes.memmove(addr+off+i*width, struct.pack('<I', sf[src+i])[:width], width)
        elif op == 1:
          ar = (ins >> 14)&0xff
          addr = gpr[ar]
          if addr + size*width > len(shared): raise RuntimeError(f"A630 local load OOB {addr:#x}+{size*width:#x}")
          out_regs = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): out_regs[dst+i] = int.from_bytes(shared[addr+i*width:addr+(i+1)*width], 'little')
        elif op == 2:
          # ldp offsets are byte offsets (Mesa disassembles these as p[rN+off]),
          # unlike global ldg offsets which are scaled by the element width.
          ar = (ins >> 14)&0xff
          off = _cat6_private_offset(ins, store=False)
          addr = gpr[ar] + off
          if addr < 0 or addr + size*width > len(private): raise RuntimeError(f"A630 private load OOB {addr:#x}+{size*width:#x}")
          out_regs = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): out_regs[dst+i] = int.from_bytes(private[addr+i*width:addr+(i+1)*width], 'little')
        elif op == 4:
          src, ar = (ins >> 1)&0xff, (ins >> 41)&0xff
          addr = gpr[ar]
          if addr + size*width > len(shared): raise RuntimeError(f"A630 local store OOB {addr:#x}+{size*width:#x}")
          sf = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): shared[addr+i*width:addr+(i+1)*width] = struct.pack('<I', sf[src+i])[:width]
        elif op == 5:
          src, ar = (ins >> 1)&0xff, (ins >> 41)&0xff
          off = _cat6_private_offset(ins, store=True)
          addr = gpr[ar] + off
          if addr < 0 or addr + size*width > len(private): raise RuntimeError(f"A630 private store OOB {addr:#x}+{size*width:#x}")
          sf = hreg if typ in (0,2,4,6) else gpr
          for i in range(size): private[addr+i*width:addr+(i+1)*width] = struct.pack('<I', sf[src+i])[:width]
        else: raise NotImplementedError(f"A630 cat6 opcode {op:#x} pc={pc}")
      elif cat == 7:
        # Barrier: cooperative scheduler resumes all threads after each barrier.
        yield
      else: raise NotImplementedError(f"A630 instruction category {cat} pc={pc}")
      if trace_thread and (pc <= 32 or int(os.getenv("QCOM_TRACE", "1")) >= 3):
        print(f"IR3 lane={local_id[0]} pc={pc:02d} r0={gpr[0:4]} r1={gpr[4:8]} r2={gpr[8:12]} r3={gpr[12:16]} r4={gpr[16:20]} "
              f"r8={gpr[32:36]} r9={gpr[36:40]} r10={gpr[40:44]} "
              f"h0={hreg[0:4]} h2={hreg[8:12]} h3={hreg[12:16]} h4={hreg[16:20]} h5={hreg[20:24]} h6={hreg[24:28]}", flush=True)
      pc += 1

  def _exec_cs(self, groups:tuple[int,int,int]):
    if not self.shader_addr: raise RuntimeError("CP_EXEC_CS without shader")
    shader = bytes(to_mv(self.shader_addr, self.shader_size))
    consts = list(to_mv(self.const_addr, self.const_size).cast('I')) if self.const_addr and self.const_size else []
    n0 = self.regs.get(mesa.REG_A6XX_SP_CS_NDRANGE_0, 0)
    local = (((n0 >> 2)&0x3ff)+1, ((n0 >> 12)&0x3ff)+1, ((n0 >> 22)&0x3ff)+1)
    if os.getenv("QCOM_TRACE"):
      print(f"EXEC_CS groups={groups} local={local} shader={self.shader_addr:#x}+{self.shader_size:#x}"
            f" const={self.const_addr:#x}+{self.const_size:#x}", flush=True)
      print(f"CONSTS {[hex(x) for x in consts[:16]]}", flush=True)
      if int(os.getenv("QCOM_TRACE", "1")) > 1:
        from tinygrad.runtime.support.compiler_mesa import disas_adreno
        disas_adreno(shader)
    for gz in range(groups[2]):
      for gy in range(groups[1]):
        for gx in range(groups[0]):
          shared = bytearray(32*1024)
          threads = [self._run_thread((lx,ly,lz),(gx,gy,gz),shared,shader,consts)
                     for lz in range(local[2]) for ly in range(local[1]) for lx in range(local[0])]
          while threads:
            waiting = []
            for th in threads:
              try:
                next(th)
                waiting.append(th)
              except StopIteration: pass
            threads = waiting
