import ctypes, math, os, struct

def _u32(x:int) -> int: return x & 0xffffffff
def _s32(x:int) -> int: return x - (1 << 32) if x & (1 << 31) else x
def _sext(x:int, bits:int) -> int: return x - (1 << bits) if x & (1 << (bits - 1)) else x
def _f32(x:int) -> float: return struct.unpack("f", struct.pack("I", x & 0xffffffff))[0]
def _f32bits(x:float) -> int: return struct.unpack("I", struct.pack("f", x))[0]
def _fmod(x:float, enc:int) -> float:
  return [-x, abs(x), -abs(x)][((enc >> 14) & 3) - 1] if (enc >> 14) & 3 else x

class A6XXEmulator:
  def __init__(self):
    self.constants:list[int] = []
    self.shader = b""
    self.regs:dict[int, int] = {}
    self.ranges:list[tuple[int, int]] = []

  def map_range(self, addr:int, size:int): self.ranges.append((addr, addr + size))
  def check_range(self, addr:int, size:int):
    if not any(start <= addr and addr + size <= end for start, end in self.ranges): raise RuntimeError(f"unmapped A6XX address {addr:#x}+{size:#x}")

  def load_constants(self, addr:int, size:int):
    self.constants = list(struct.unpack(f"{size // 4}I", ctypes.string_at(addr, size)))

  def load_shader(self, addr:int, size:int): self.shader = ctypes.string_at(addr, size)

  def write_regs(self, base:int, vals:tuple[int, ...]):
    for i, val in enumerate(vals): self.regs[base + i] = val

  @staticmethod
  def _src(enc:int, gpr:list[int], hreg:list[int], consts:list[int], full:bool) -> int:
    kind = (enc >> 11) & 0x7
    if kind == 0: return (gpr if full else hreg)[enc & 0xff]
    if kind in (2, 6):
      v = consts[enc & 0x7ff]
      return v if full else v >> (16 if enc & 1 else 0) & 0xffff
    if kind == 4: return _sext(enc & 0x7ff, 11)
    if kind == 5:
      flut = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1 / math.pi, math.log(2), math.log2(math.e), math.log10(2), math.log2(10), 4.0]
      return _f32bits(flut[enc & 0x3ff])
    raise NotImplementedError(f"A6XX multisrc encoding {enc:#x}")

  @staticmethod
  def _cat3_src(enc:int, gpr:list[int]) -> int:
    if (enc >> 8) & 0x1f == 0: return gpr[enc & 0xff]
    if enc & 0x1000: return enc & 0xfff
    raise NotImplementedError(f"A6XX cat3 source encoding {enc:#x}")

  def _run_thread(self, global_id:int):
    gpr, hreg, pc = [0] * 256, [0] * 256, 0
    gpr[0] = gpr[48 * 4] = global_id
    if (localid:=((self.regs.get(0xb997, 0) >> 24) & 0xff)) < 64: gpr[localid * 4] = global_id
    while pc * 8 < len(self.shader):
      ins = struct.unpack_from("Q", self.shader, pc * 8)[0]
      cat, dst = (ins >> 61) & 0x7, (ins >> 32) & 0xff
      if cat == 0:
        if (ins >> 55) & 0xf == 6: break
      elif cat == 1:
        src_type, dst_type, mode = (ins >> 50) & 0x7, (ins >> 46) & 0x7, (ins >> 53) & 0x3
        src_file = hreg if src_type in (0, 2, 4, 6) else gpr
        src = ins & 0xffffffff if mode == 2 else self.constants[ins & 0x7ff] if mode == 1 else src_file[ins & 0xff]
        if src_type in (2, 4): src &= 0xffff
        if src_type == 4: src = _sext(src, 16)
        if src_type == 5: src = _s32(src)
        (hreg if dst_type in (0, 2, 4, 6) else gpr)[dst] = _u32(src)
      elif cat == 2:
        full, conv = bool((ins >> 52) & 1), bool((ins >> 46) & 1)
        op = (ins >> 53) & 0x3f
        for rpt in range(((ins >> 40) & 0x3) + 1):
          aenc = (ins & 0xffff) + (rpt if (ins >> 43) & 1 else 0)
          benc = ((ins >> 16) & 0xffff) + (rpt if (ins >> 51) & 1 else 0)
          a, b = self._src(aenc, gpr, hreg, self.constants, full), self._src(benc, gpr, hreg, self.constants, full)
          fa, fb = _fmod(_f32(a), aenc), _fmod(_f32(b), benc)
          if op == 0x00: out = _f32bits(fa + fb)
          elif op == 0x01: out = _f32bits(min(fa, fb))
          elif op == 0x02: out = _f32bits(max(fa, fb))
          elif op == 0x03: out = _f32bits(fa * fb)
          elif op == 0x05:
            cond = (ins >> 48) & 0x7
            out = int([fa < fb, fa <= fb, fa > fb, fa >= fb, fa == fb, fa != fb][cond])
          elif op == 0x10: out = a + b
          elif op == 0x14:
            cond = (ins >> 48) & 0x7
            out = int([a < b, a <= b, a > b, a >= b, a == b, a != b][cond])
          elif op == 0x38: out = _s32(a) >> (b & 31)
          elif op == 0x37: out = a >> (b & 31)
          elif op == 0x36: out = a << (b & 31)
          else: raise NotImplementedError(f"A6XX cat2 opcode {op:#x} at {pc}")
          (hreg if full == conv and dst <= 0xf7 else gpr)[dst + rpt] = _u32(out)
      elif cat == 3:
        op = (ins >> 55) & 0xf
        a = self._cat3_src(ins & 0x1fff, gpr)
        b, c = gpr[(ins >> 47) & 0xff], self._cat3_src((ins >> 16) & 0x1fff, gpr)
        if op == 0x9:
          out = a if b else c
          if os.getenv("QCOM_TRACE") and global_id < 2: print(f"thread {global_id} sel {a:#x} if {b:#x} else {c:#x} -> {out:#x}")
        elif op == 0xa: out = (b >> (a & 31)) | c
        else: raise NotImplementedError(f"A6XX cat3 opcode {op:#x} at {pc}")
        gpr[dst] = _u32(out)
      elif cat == 4:
        full, conv, op = bool((ins >> 52) & 1), bool((ins >> 46) & 1), (ins >> 53) & 0x3f
        for rpt in range(((ins >> 40) & 0x3) + 1):
          enc = (ins & 0xffff) + (rpt if (ins >> 43) & 1 else 0)
          x = _fmod(_f32(self._src(enc, gpr, hreg, self.constants, full)), enc)
          out = [lambda:1/x, lambda:1/math.sqrt(x), lambda:math.log2(x), lambda:2**x,
                 lambda:math.sin(x), lambda:math.cos(x), lambda:math.sqrt(x)][op]()
          if os.getenv("QCOM_TRACE") and global_id < 2: print(f"thread {global_id} cat4 op={op} x={x} out={out}")
          (hreg if full == conv and dst <= 0xf7 else gpr)[dst + rpt] = _f32bits(out)
      elif cat == 6:
        op = (ins >> 54) & 0x1f
        if op == 0:
          addr_lo, size, typ = (ins >> 14) & 0xff, (ins >> 24) & 0x7, (ins >> 49) & 0x7
          addr = gpr[addr_lo] | gpr[(addr_lo + 1) & 0xff] << 32
          off, width = _sext((ins >> 1) & 0x1fff, 13), 2 if typ in (0, 2, 4) else 1 if typ == 6 else 4
          self.check_range(addr + off * width, size * width)
          for i in range(size): gpr[dst + i] = int.from_bytes(ctypes.string_at(addr + (off + i) * width, width), "little")
        elif op == 3:
          src, addr_lo = (ins >> 1) & 0xff, (ins >> 41) & 0xff
          addr = gpr[addr_lo] | gpr[(addr_lo + 1) & 0xff] << 32
          off = _sext(((ins >> 9) & 0x1f) << 8 | ((ins >> 32) & 0xff), 13)
          size, typ = (ins >> 24) & 0x7, (ins >> 49) & 0x7
          width = 2 if typ in (0, 2, 4) else 1 if typ == 6 else 4
          if os.getenv("QCOM_TRACE") and global_id < 2: print(f"thread {global_id} stg addr={addr:#x} src={src} value={gpr[src]:#x}")
          self.check_range(addr + off * width, size * width)
          for i in range(size): ctypes.memmove(addr + (off + i) * width, struct.pack("I", gpr[src + i])[:width], width)
        else: raise NotImplementedError(f"A6XX cat6 opcode {op:#x} at {pc}")
      else: raise NotImplementedError(f"A6XX category {cat} at {pc}")
      pc += 1

  def exec_cs(self, groups:tuple[int, int, int]):
    cfg = self.regs[0xb990]
    local = (((cfg >> 2) & 0x3ff) + 1, ((cfg >> 12) & 0x3ff) + 1, ((cfg >> 22) & 0x3ff) + 1)
    for z in range(groups[2] * local[2]):
      for y in range(groups[1] * local[1]):
        for x in range(groups[0] * local[0]): self._run_thread(x)
