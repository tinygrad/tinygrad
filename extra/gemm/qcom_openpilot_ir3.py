"""Small IR3 encoding helpers used by the OpenPilot QCOM graph patches."""
import re, struct

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def plain_name(name:str) -> str: return ANSI_RE.sub("", name)

def _freg(name:str|int) -> int:
  if isinstance(name, int): return name
  register, component = name.replace("r", "").split(".")
  return int(register) * 4 + "xyzw".index(component)

def _pack(lo:int, hi:int) -> bytes: return struct.pack("<II", lo & 0xffffffff, hi & 0xffffffff)

def nop() -> bytes: return _pack(0, 0)

def branch(offset:int) -> bytes: return struct.pack("<iI", offset, 0x00900000)

def mov_f32(dst:str, src:str, rpt:int=0, sy:bool=False, ss:bool=False, r:bool=False) -> bytes:
  return _pack(_freg(src), (0x30044000 if sy else 0x20044000) | (0x1000 if ss else 0) |
               (0x800 if r else 0) | ((rpt & 0x7f) << 8) | _freg(dst))

def add_s(dst:str, src:str, imm:int, ss:bool=False) -> bytes:
  lo = ((0x27 if imm < 0 else 0x20) << 24) | ((imm & 0xff) << 16) | _freg(src)
  return _pack(lo, 0x42300000 | (0x1000 if ss else 0) | _freg(dst))

def mad_f32(dst:str, src1:str, src2:str, src3:str, rpt:int=0, sy:bool=False, r:bool=False) -> bytes:
  d, s1, s2, s3 = _freg(dst), _freg(src1), _freg(src2), _freg(src3)
  hi = ((0x73 if sy else 0x63) << 24) | (0x80 << 16) | ((s2 >> 1) << 16) | (((s2 & 1) << 7 | (rpt & 0x7f)) << 8) | d
  lo = (0x20000000 if r else 0) | (s3 << 16) | (0x8000 if r else 0) | s1
  return _pack(lo, hi)

def isam_f32(dst:str, coord:str, tex:int=0, samp:int=0) -> bytes:
  return _pack((tex * 2) << 24 | ((samp & 7) << 21) | (_freg(coord) * 2 + 1), 0xa0001f00 | _freg(dst))
