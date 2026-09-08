# A6xx emulator
# 64 wide lockstep
# Cat0 (br/jump/park) stays in Python
# ALU/mem is one clang kernel per encoding
#
# Buffers:
#   0 gpr    gpr[tid*256 + reg]     full 32-bit file
#   1 h      h[tid*256 + reg]       half 16-bit file (same Mesa _ index, not aliased)
#   2 lds    a630 cs_shared_mem_size
#   3 pvt    pvt[tid*32768 + off]   per-fiber private
#   4 vmem   identity map, INDEX is host VA
#   5 caddr  const file pointer
#   6 wst    wave mmap: pc, act, pmode, pmask, park stack
#   7 enc    packed DST/SRC/OFF per instruction
from __future__ import annotations
import ctypes, functools, hashlib, itertools, math, mmap, os, struct
from typing import Literal
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.helpers import Context, mv_address, to_mv, unwrap
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen import to_program
from tinygrad.runtime.autogen import mesa, libc

# a630 fd_dev_info: threadsize_base=64, cs_shared_mem_size=32*1024 (wave_granularity/fibers_per_sp are occupancy, not emu width)
WAVE, CS_SHARED_MEM_SIZE, PARK, _JP, DONE, BAR = 64, 32 * 1024, 64, 0xffffffff, 1, 2
_ONES64 = (1 << 64) - 1

### HOST MAPS

def _host_buf(nitems: int, fmt: Literal["I", "i", "B", "Q"]) -> tuple[int, mmap.mmap, memoryview]:
  nbytes = max(nitems * {"I": 4, "i": 4, "B": 1, "Q": 8}[fmt], 8)
  m = mmap.mmap(-1, nbytes)
  addr = mv_address(m)
  return addr, m, to_mv(addr, nbytes).cast(fmt)

### MESA DECODE

def _cstr(p) -> str:
  if not p: return ""
  return ctypes.string_at(p).decode()

_DEC_OP = {"DST", "SRC1", "SRC2", "SRC3"}
_OP_FIELD = {"GPR", "SWIZ", "SRC", "CONST", "IMMED", "HALF", "LAST", "ABSNEG", "SRC_R"}

@functools.cache
def decode_shader(code: bytes, gpu_id=630) -> list[dict]:
  instrs: list[dict] = []
  cur: dict = {}
  op: str|None = None

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def pre(_data, n, _instr):
    nonlocal cur, op
    cur, op = {"n": n}, None

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value))
  def field(_data, name, val):
    nonlocal op
    nm = _cstr(name).split(":")[0]
    x = _cstr(val.contents.str) if val.contents.str else val.contents.num
    if nm == "NAME":
      if isinstance(x, str) and x and not x.startswith("#"):
        old = cur.get("NAME")
        if old is None or x.startswith(old): cur["NAME"] = x
    elif nm in _DEC_OP:
      if op != nm: op, cur[nm] = nm, {"_": x}
    elif nm == "SRC" and op == "DST":
      op, cur["SRC"] = "SRC", {"_": x}
    elif op is not None and nm in _OP_FIELD:
      cur[op][nm] = x
    else:
      cur[nm] = x

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def post(_data, _n, _instr):
    instrs.append(cur)

  fp = libc.fopen(os.devnull.encode(), b"w")
  if not fp: raise OSError("fopen /dev/null")
  try:
    buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    mesa.ir3_isa_disasm(buf, len(code), ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE)),
      mesa.struct_isa_decode_options(gpu_id, True, 0, False, field_cb=field, pre_instr_cb=pre, post_instr_cb=post))
  finally:
    libc.fclose(fp)
  return instrs

def _iname(raw: dict) -> str|None:
  name = raw.get("NAME")
  if name in (None, "") and "SRC_TYPE" in raw:
    name = "sct" if "DST3" in raw else "gat" if "SRC3" in raw and "DST0" in raw else "swz" if "DST0" in raw else "cov"
  elif isinstance(name, str) and name.startswith("swz"): name = "swz"
  return name

### IMMEDIATES

def _u32(x: int) -> int: return x & 0xffffffff
def _s11(x: int) -> int:
  x = int(x)
  return x - 0x800 if 0x400 <= x <= 0x7ff else x
def _s32(x: int) -> int: return (x := _u32(x)) - 0x100000000 if x >= 0x80000000 else x
def _f32bits(x: float) -> int: return struct.unpack("I", struct.pack("f", x))[0]
def _f16bits(x: float) -> int:
  try: return struct.unpack("H", struct.pack("e", x))[0]
  except OverflowError: return 0x7c00 if x > 0 else 0xfc00

_FLUT_F = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi,
  1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0)
_FLUT = [_f32bits(x) for x in _FLUT_F]
_FLUT16 = [_f16bits(x) for x in _FLUT_F]

### TYPES

_TSIZE = {mesa.TYPE_F16: 2, mesa.TYPE_F32: 4, mesa.TYPE_U16: 2, mesa.TYPE_U32: 4,
          mesa.TYPE_S16: 2, mesa.TYPE_S32: 4, mesa.TYPE_U8: 1}
# ISA #type-half: f16/u16/s16/u8 data is the half file. Mesa derives TYPE_HALF from TYPE but
# omits it from stp/stl display, so decode leaves TYPE_HALF unset on those stores.
_HALF_TYPE = {mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8, mesa.TYPE_U8_32}
def _half_mem(ins: dict) -> bool:
  t = ins.get("TYPE", mesa.TYPE_U32)
  if t in _HALF_TYPE: return True
  return bool(ins.get("TYPE_HALF") or ins.get("DST_HALF"))

def _cat3_src(ins: dict, k: str) -> dict:
  d = ins[k]
  if k == "SRC2" and "HALF" not in d and (ins.get("SRC1", {}).get("HALF") or ins.get("SRC3", {}).get("HALF")):
    return {**d, "HALF": 1}
  return d

### CS LAUNCH

def cs_ndrange(regs: dict[int, int]):
  n0 = regs.get(mesa.REG_A6XX_SP_CS_NDRANGE_0, 0)
  ls = (((n0 >> 2) & 0x3ff) + 1, ((n0 >> 12) & 0x3ff) + 1, ((n0 >> 22) & 0x3ff) + 1)
  ng = (regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X, 1), regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y, 1),
        regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z, 1))
  cfg = regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0)
  wgid = (cfg >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT) & 0xff
  wgsz = (cfg >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID__SHIFT) & 0xff
  lid = (cfg >> mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT) & 0xff
  return ls, ng, lid, wgid, wgsz

def cs_init_consts(const_addr: int, wgsz: int, ls, ng):
  def store_xyz(idx: int, xyz):
    if idx == 0xfc or not const_addr: return
    mv = to_mv(const_addr + idx * 4, 12).cast("I")
    mv[0], mv[1], mv[2] = int(xyz[0]), int(xyz[1]), int(xyz[2])
  store_xyz(wgsz, ls)
  if wgsz != 0xfc and wgsz >= 8: store_xyz(wgsz - 8, ng)

def cs_init_ids(gpr, lid: int, wgid: int, lx: int, ly: int, lz: int, gx: int, gy: int, gz: int):
  if lid != 0xfc: gpr[lid], gpr[lid + 1], gpr[lid + 2] = lx, ly, lz
  if wgid != 0xfc: gpr[wgid], gpr[wgid + 1], gpr[wgid + 2] = gx, gy, gz

### UOP ALU

def _c(v, dt=dtypes.uint32): return UOp.const(v, dt)

def _to_u32(v: UOp) -> UOp:
  if v.dtype == dtypes.uint32: return v
  if v.dtype == dtypes.float32: return v.bitcast(dtypes.uint32)
  if v.dtype == dtypes.half: return v.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if v.dtype == dtypes.int32: return v.bitcast(dtypes.uint32)
  if v.dtype == dtypes.bool: return v.where(_c(1), _c(0))
  return v.cast(dtypes.uint32)

def _cmp(cond, a: UOp, b: UOp, flt: bool) -> UOp:
  if cond == mesa.IR3_COND_LT: return a < b
  if cond == mesa.IR3_COND_GT: return b < a
  if cond == mesa.IR3_COND_EQ: return a.eq(b)
  if cond == mesa.IR3_COND_NE: return a.ne(b)
  if cond == mesa.IR3_COND_LE: return (a < b) | a.eq(b) if flt else (b < a).ne(True)
  if cond == mesa.IR3_COND_GE: return (b < a) | a.eq(b) if flt else (a < b).ne(True)
  raise RuntimeError(f"unhandled ir3 cond {cond}")

def _floor(x: UOp) -> UOp:
  t = UOp(Ops.TRUNC, src=(x,))
  return ((x < _c(0, x.dtype)) & x.ne(t)).where(t - _c(1, x.dtype), t)

def _sign(x: UOp) -> UOp:
  z, one = _c(0, x.dtype), _c(1, x.dtype)
  return (x < z).where(-one, x.ne(z).where(one, z))

def _clz(x: UOp) -> UOp:
  n = _c(0)
  for bits in (16, 8, 4, 2, 1):
    t = (x >> _c(32 - bits)).eq(_c(0))
    n = n + t.where(_c(bits), _c(0))
    x = t.where(x << _c(bits), x)
  return x.eq(_c(0)).where(_c(0xffffffff), n)

_ALU_F = {"add.f": lambda a, b: a + b, "mul.f": lambda a, b: a * b,
          "max.f": lambda a, b: (a < b).where(b, a), "min.f": lambda a, b: (a < b).where(a, b)}
_UNARY_F = {
  "sqrt": lambda x: UOp(Ops.SQRT, src=(x,)), "exp2": lambda x: UOp(Ops.EXP2, src=(x,)),
  "log2": lambda x: UOp(Ops.LOG2, src=(x,)), "rcp": lambda x: UOp(Ops.RECIPROCAL, src=(x,)),
  "sin": lambda x: UOp(Ops.SIN, src=(x,)),
  "rsq": lambda x: UOp(Ops.RECIPROCAL, src=(UOp(Ops.SQRT, src=(x,)),)),
  "hexp2": lambda x: UOp(Ops.EXP2, src=(x,)), "hlog2": lambda x: UOp(Ops.LOG2, src=(x,)),
  "hrsq": lambda x: UOp(Ops.RECIPROCAL, src=(UOp(Ops.SQRT, src=(x,)),)),
}
_UNARY_F1 = {"trunc.f": lambda x: UOp(Ops.TRUNC, src=(x,)), "floor.f": _floor, "sign.f": _sign}
_ALU_I = {
  "add.u": lambda a, b: a + b, "sub.u": lambda a, b: a - b,
  "and.b": lambda a, b: a & b, "or.b": lambda a, b: a | b, "xor.b": lambda a, b: a ^ b,
  "shl.b": lambda a, b: a << (b & _c(31)), "shr.b": lambda a, b: a >> (b & _c(31)),
  "ashr.b": lambda a, b: (a.bitcast(dtypes.int32) >> (b & _c(31)).cast(dtypes.int32)).bitcast(dtypes.uint32),
  "min.u": lambda a, b: (a < b).where(a, b), "max.u": lambda a, b: (a < b).where(b, a),
  "mul.u24": lambda a, b: (a & _c(0xffffff)) * (b & _c(0xffffff)),
  "mull.u": lambda a, b: (a & _c(0xffff)) * (b & _c(0xffff)),
}
_CTRL = {"nop", "end", "bar", "jump", "br", "brao", "braa", "predt", "predf", "prede", None}
_MEMOPS = {"ldg", "stg", "stl", "ldl", "stp", "ldp"}
_FLUT_OPS = set(_ALU_F) | set(_UNARY_F) | set(_UNARY_F1) | {"absneg.f", "mad.f32", "mad.f16", "cmps.f"}
_HALF_OK = set(_ALU_F) | set(_UNARY_F1) | {"absneg.f"}
_KIND_SKIP = {"n", "SY", "SS", "JP", "NOP", "UL", "EI", "LAST", "OFF", "SWIZ", "_kind"}
_KIND_LIFT = {"_", "GPR", "SRC", "CONST", "IMMED"}
_OP_KEYS = {"DST", "SRC", "SRC1", "SRC2", "SRC3", "DST0", "DST1", "SRC0"}
_LAYOUT_KEYS = ("DST", "SRC", "SRC1", "SRC2", "SRC3", "DST0", "DST1", "SRC0")
_ENC_KEYS = _LAYOUT_KEYS + ("OFF",)
_ENC_STRIDE, _ENC_SLOT = len(_ENC_KEYS), {k: i for i, k in enumerate(_ENC_KEYS)}

def _kind(raw: dict):
  if (hit:=raw.get("_kind")) is not None: return hit
  def rec(obj, in_op=False):
    if isinstance(obj, dict):
      out = {}
      for k, v in obj.items():
        if k in _KIND_SKIP: continue
        if in_op and k in _KIND_LIFT: out[k] = True
        else: out[k] = rec(v, in_op or k in _OP_KEYS)
      return tuple(sorted(out.items()))
    if isinstance(obj, (list, tuple)): return tuple(rec(v, in_op) for v in obj)
    if isinstance(obj, (bool, str, type(None), float)): return obj
    if isinstance(obj, int): return int(obj)
    try: return int(obj)
    except (TypeError, ValueError): return str(obj)
  raw["_kind"] = ret = rec(raw)
  return ret

def _layout_keys(raw: dict) -> list[str]:
  keys = [k for k in _LAYOUT_KEYS if k in raw]
  if _iname(raw) in _MEMOPS: keys.append("OFF")
  return keys

def _pack_one(raw: dict, name: str, k: str) -> int:
  if k == "OFF": return int(raw.get("OFF", 0)) & 0xffffffff
  x = raw[k]
  if k in ("DST", "DST0", "DST1") or not isinstance(x, dict):
    return int(x if not isinstance(x, dict) else x["_"]) & 0xffffffff
  if "IMMED" in x:
    imm = int(x["IMMED"])
    if name == "cov": return imm & 0xffffffff
    if name in _FLUT_OPS:
      hf = bool(x.get("HALF")) and "CONST" not in x
      return int((_FLUT16 if hf else _FLUT)[imm]) & 0xffffffff
    return _s11(imm) & 0xffffffff
  if "CONST" in x:
    return int(x.get("SRC", int(x["CONST"]) * 4 + int(x.get("SWIZ", 0)))) & 0xffffffff
  return int(unwrap(x.get("SRC", x.get("_")))) & 0xffffffff

def _pack_named(raw: dict) -> dict[str, int]:
  name = _iname(raw) or ""
  return {k: _pack_one(raw, name, k) for k in _layout_keys(raw)}

### CTX

class _Ctx:
  def __init__(self, nt: int):
    self.nt = nt
    self.stores: list[UOp] = []
    self.pval: dict[str, UOp] = {}
    self.raw: dict = {}
    self.rpt, self.ins_name = 0, ""
    self.tid = UOp.range(nt, 0, dtype=dtypes.int)
    self.gpr = UOp.param(0, dtypes.uint32, self.nt * 256)
    self.h = UOp.param(1, dtypes.uint32, self.nt * 256)
    self.lds = UOp.param(2, dtypes.uint8, CS_SHARED_MEM_SIZE)
    self.pvt = UOp.param(3, dtypes.uint8, self.nt * 32768)
    self.vmem = UOp.param(4, dtypes.uint32, 1 << 40)
    self.caddr = UOp.param(5, dtypes.uint64, 1)
    self.wst = UOp.param(6, dtypes.uint32, 64)
    self.enc = UOp.param(7, dtypes.uint32, 1 << 18)
    def u64(lo: int) -> UOp:
      return (self.wst.index(_c(lo)).load().cast(dtypes.uint64) |
              (self.wst.index(_c(lo + 1)).load().cast(dtypes.uint64) << _c(32, dtypes.uint64)))
    self._pc = self.wst.index(0).load()
    act, pmode, pmask = u64(2), self.wst.index(4).load(), u64(6)
    lanes = pmode.eq(_c(1)).where(act & pmask, pmode.eq(_c(2)).where(act & (pmask ^ _c(_ONES64, dtypes.uint64)), act))
    lanes = lanes & _c((1 << self.nt) - 1, dtypes.uint64)
    self.active = ((lanes >> self.tid.cast(dtypes.uint64)) & _c(1, dtypes.uint64)).ne(_c(0, dtypes.uint64))

  def bind(self, raw: dict):
    self.raw = raw
    pci = self._pc.cast(dtypes.int) * _c(_ENC_STRIDE, dtypes.int)
    for k in _layout_keys(raw): self.pval[k] = self.enc.index(pci + _c(_ENC_SLOT[k], dtypes.int)).load()

  def _as_int(self, r: UOp|int) -> UOp:
    if isinstance(r, UOp): return r if r.dtype == dtypes.int else r.cast(dtypes.int)
    return _c(int(r), dtypes.int)

  def _idx(self, r: UOp|int) -> UOp: return self.tid * 256 + self._as_int(r)

  def _src_rel(self, k: str) -> bool:
    d = self.raw.get(k)
    return bool(isinstance(d, dict) and (d.get("SRC_R") or self.raw.get(k+"_R")))

  def ridx(self, key: str, extra: int=0) -> UOp:
    inc = extra + (self.rpt if key in ("DST", "DST0", "DST1") or self._src_rel(key) else 0)
    p = self.pval[key].cast(dtypes.int)
    return p + _c(inc, dtypes.int) if inc else p

  def gpr_r(self, r: UOp|int) -> UOp: return self.gpr.index(self._idx(r)).load()
  def h_r(self, r: UOp|int) -> UOp: return self.h.index(self._idx(r)).load() & _c(0xffff)

  def _st(self, buf: UOp, idx: UOp, val: UOp) -> UOp:
    st = buf.index(idx.valid(self.active)).store(val)
    self.stores.append(st)
    return st

  def wr(self, ins: dict, val: UOp):
    v, idx = _to_u32(val), self._idx(self.ridx("DST"))
    if ins.get("DST_HALF"):
      self.h = self.h.after(self._st(self.h, idx, v & _c(0xffff)))
    else:
      self.gpr = self.gpr.after(self._st(self.gpr, idx, v))

  def wr_reg(self, half: bool, r: UOp|int, val: UOp):
    v, idx = _to_u32(val), self._idx(r)
    if half: self.h = self.h.after(self._st(self.h, idx, v & _c(0xffff)))
    else: self.gpr = self.gpr.after(self._st(self.gpr, idx, v))

  def src(self, ins: dict, k: str, flut: bool=False, half: bool=False) -> UOp:
    d = ins[k]
    if self.ins_name in ("andg", "shrg", "shlg", "shrm", "shlm", "mad.f16") and k in ("SRC1", "SRC2", "SRC3") and isinstance(d, dict):
      d = _cat3_src(ins, k)
    if self.ins_name == "sel.b16" and isinstance(d, dict) and "HALF" not in d: d = {**d, "HALF": 1}
    if isinstance(d, dict) and "IMMED" in d:
      x = self.pval[k]
    elif isinstance(d, dict) and "CONST" in d:
      i = self.pval[k]
      if self.rpt and self._src_rel(k): i = i + _c(self.rpt)
      x = self._ld_vmem(self.caddr.index(0).load() + i.cast(dtypes.uint64) * _c(4, dtypes.uint64), 4)
    else:
      idx = self.ridx(k) if k in self.pval else self._as_int(d if not isinstance(d, dict) else int(unwrap(d.get("SRC", d.get("_")))))
      x = self.h_r(idx) if isinstance(d, dict) and d.get("HALF") else self.gpr_r(idx)
    an = int(d.get("ABSNEG", 0)) if isinstance(d, dict) else 0
    if not an: return x
    if flut:
      if half:
        if an & 2: x = x & _c(0x7fff)
        if an & 1: x = x ^ _c(0x8000)
      else:
        if an & 2: x = x & _c(0x7fffffff)
        if an & 1: x = x ^ _c(0x80000000)
      return x
    xi = x.bitcast(dtypes.int32)
    if an & 2: xi = (xi < _c(0, dtypes.int32)).where(-xi, xi)
    if an & 1: xi = -xi
    return xi.bitcast(dtypes.uint32)

  def fsrc(self, ins: dict, k: str) -> UOp:
    d = ins[k]
    if self.ins_name == "mad.f16" and k in ("SRC1", "SRC2", "SRC3") and isinstance(d, dict):
      d = _cat3_src(ins, k)
    hf = bool(isinstance(d, dict) and d.get("HALF")) and not (isinstance(d, dict) and "CONST" in d)
    bits = self.src(ins, k, True, hf)
    if hf: return (bits & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
    return bits.bitcast(dtypes.float32)

  def _sat_f(self, val: UOp) -> UOp:
    if not self.sat: return val
    if val.dtype != dtypes.float32: val = val.cast(dtypes.float32)
    z, o = _c(0, dtypes.float32), _c(1, dtypes.float32)
    val = (val < z).where(z, val)
    return (val < o).where(val, o)

  def fwr(self, ins: dict, val: UOp):
    if val.dtype != dtypes.float32: val = val.cast(dtypes.float32)
    val = self._sat_f(val)
    if ins.get("DST_HALF"): self.wr(ins, val.cast(dtypes.half))
    else: self.wr(ins, val)

  def gaddr(self, s: UOp|int, off: UOp|int) -> UOp:
    s = self._as_int(s)
    lo, hi = self.gpr_r(s), self.gpr_r(s + _c(1, dtypes.int))
    offi = off.bitcast(dtypes.int32) if isinstance(off, UOp) else _c(_s32(int(off)), dtypes.int32)
    offu = offi.cast(dtypes.int64).bitcast(dtypes.uint64)
    return (lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << _c(32, dtypes.uint64))) + offu

  def _ld_vmem(self, addr: UOp, n: int) -> UOp:
    x = _c(0)
    for j in range(n):
      a = addr + _c(j, dtypes.uint64)
      word = self.vmem.index((a >> _c(2, dtypes.uint64)).valid(self.active)).load()
      sh = (a & _c(3, dtypes.uint64)).cast(dtypes.uint32) * _c(8)
      x = x | ((word >> sh) & _c(0xff)) << _c(8 * j)
    return x

  def _st_vmem(self, addr: UOp, n: int, val: UOp):
    vmem, val = self.vmem, _to_u32(val)
    for j in range(n):
      a = addr + _c(j, dtypes.uint64)
      idx = a >> _c(2, dtypes.uint64)
      word, sh = vmem.index(idx.valid(self.active)).load(), (a & _c(3, dtypes.uint64)).cast(dtypes.uint32) * _c(8)
      mask, byte = _c(0xff) << sh, (val >> _c(8 * j)) & _c(0xff)
      st = vmem.index(idx.valid(self.active)).store((word & (mask ^ _c(0xffffffff))) | (byte << sh))
      self.stores.append(st)
      vmem = vmem.after(st)
    self.vmem = vmem

  def _ld_bytes(self, buf: UOp, idx0: UOp, n: int) -> tuple[UOp, UOp]:
    x = _c(0)
    for j in range(n):
      b = buf.index((idx0 + j).valid(self.active)).load()
      x = x | (b.cast(dtypes.uint32) << _c(8 * j))
    return x, buf

  def _st_bytes(self, buf: UOp, idx0: UOp, n: int, val: UOp) -> UOp:
    val = _to_u32(val)
    for j in range(n):
      byte = ((val >> _c(8 * j)) & _c(0xff)).cast(dtypes.uint8)
      st = buf.index((idx0 + j).valid(self.active)).store(byte)
      self.stores.append(st)
      buf = buf.after(st)
    return buf

  def _alu_f(self, ins, raw):
    self.fwr(ins, _ALU_F[self.ins_name](self.fsrc(ins, "SRC1"), self.fsrc(ins, "SRC2")))

  def _absneg_f(self, ins, raw):
    self.fwr(ins, self.fsrc(ins, "SRC1"))

  def _unary_f1(self, ins, raw):
    self.fwr(ins, _UNARY_F1[self.ins_name](self.fsrc(ins, "SRC1")))

  def _alu_i(self, ins, raw):
    name, a, b = self.ins_name, self.src(ins, "SRC1"), self.src(ins, "SRC2")
    d1 = ins.get("SRC1")
    hf = bool(isinstance(d1, dict) and d1.get("HALF"))
    if name in ("add.u", "sub.u") and hf:
      r = ((a & _c(0xffff)) + (b & _c(0xffff))) if name == "add.u" else ((a & _c(0xffff)) - (b & _c(0xffff)))
      self.wr(ins, r & _c(0xffff))
      return
    if name in ("min.s", "max.s"):
      if hf:
        ia = (a & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
        ib = (b & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
      else:
        ia, ib = a.bitcast(dtypes.int32), b.bitcast(dtypes.int32)
      self.wr(ins, ((ia < ib).where(ia, ib) if name == "min.s" else (ia < ib).where(ib, ia)).bitcast(dtypes.uint32))
      return
    if name == "mul.s24":
      sa = ((a << _c(8)).bitcast(dtypes.int32) >> _c(8, dtypes.int32))
      sb = ((b << _c(8)).bitcast(dtypes.int32) >> _c(8, dtypes.int32))
      self.wr(ins, (sa * sb).bitcast(dtypes.uint32))
      return
    self.wr(ins, _ALU_I[name](a, b))

  def _absneg_s(self, ins, raw): self.wr(ins, self.src(ins, "SRC1"))
  def _not_b(self, ins, raw): self.wr(ins, self.src(ins, "SRC1") ^ _c(0xffffffff))
  def _clz_b(self, ins, raw): self.wr(ins, _clz(self.src(ins, "SRC1")))
  def _unary_f(self, ins, raw): self.fwr(ins, _UNARY_F[self.ins_name](self.fsrc(ins, "SRC")))

  def _shift3(self, ins, raw):
    name = self.ins_name
    sh, src, extra = (self.src(ins, k) for k in ("SRC1", "SRC2", "SRC3"))
    t = (src >> (sh & _c(31))) if name in ("shrg", "shrm") else (src << (sh & _c(31)))
    self.wr(ins, (t | extra) if name in ("shrg", "shlg") else (t & extra))

  def _andg(self, ins, raw):
    a, b, extra = (self.src(ins, k) for k in ("SRC1", "SRC2", "SRC3"))
    self.wr(ins, (a & b) | extra)

  def _madsh(self, ins, raw):
    a, b, c = self.src(ins, "SRC1"), self.src(ins, "SRC2"), self.src(ins, "SRC3")
    self.wr(ins, (((a & _c(0xffff)) * ((b >> _c(16)) & _c(0xffff))) << _c(16)) + c)

  def _cmps(self, ins, raw):
    name = self.ins_name
    if name == "cmps.f":
      a, b, flt = self.fsrc(ins, "SRC1"), self.fsrc(ins, "SRC2"), True
    else:
      a, b = self.src(ins, "SRC1"), self.src(ins, "SRC2")
      hf = bool(ins.get("SRC1", {}).get("HALF"))
      if name == "cmps.s":
        if hf:
          a = (a & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
          b = (b & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
        else:
          a, b = a.bitcast(dtypes.int32), b.bitcast(dtypes.int32)
      elif hf:
        a, b = a & _c(0xffff), b & _c(0xffff)
      flt = False
    self.wr(ins, _cmp(ins["COND"], a, b, flt))

  def _sel(self, ins, raw):
    s2 = self.src(ins, "SRC2")
    self.wr(ins, s2.ne(_c(0)).where(self.src(ins, "SRC1"), self.src(ins, "SRC3")))

  def _mad_f32(self, ins, raw):
    def madsrc(k: str) -> UOp:
      x = self.src(ins, k, True)
      if ins.get(k+"_NEG"): x = x ^ _c(0x80000000)
      return x.bitcast(dtypes.float32)
    self.fwr(ins, madsrc("SRC1") * madsrc("SRC2") + madsrc("SRC3"))

  def _mad_f16(self, ins, raw):
    def madsrc(k: str) -> UOp:
      x = self.fsrc(ins, k)
      return -x if ins.get(k+"_NEG") else x
    self.fwr(ins, madsrc("SRC1") * madsrc("SRC2") + madsrc("SRC3"))

  def _swz(self, ins, raw):
    if raw["SRC_TYPE"] != raw["DST_TYPE"]: raise RuntimeError(f"unhandled ir3 swz types {raw}")
    half = bool(raw.get("DST_HALF") or raw.get("HALF"))
    def s(k: str) -> UOp:
      x = raw[k] if k in raw else ins[k]
      if isinstance(x, dict): return self.src(ins if k in ins else raw, k)
      idx = self.ridx(k)
      return self.h_r(idx) if half else self.gpr_r(idx)
    a, b = s("SRC0"), s("SRC1")
    self.wr_reg(half, self.ridx("DST0"), a)
    self.wr_reg(half, self.ridx("DST1"), b)

  def _cov(self, ins, raw):
    src = self.pval["SRC"] if "IMMED" in ins["SRC"] else self.src(ins, "SRC")
    st, dt, rnd = ins["SRC_TYPE"], ins["DST_TYPE"], ins.get("ROUND", 0)
    if st in (mesa.TYPE_F16, mesa.TYPE_F32):
      f = (src & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32) if st == mesa.TYPE_F16 else src.bitcast(dtypes.float32)
      if dt == mesa.TYPE_F16: self.wr(ins, f.cast(dtypes.half))
      elif dt == mesa.TYPE_F32: self.wr(ins, f)
      elif dt in (mesa.TYPE_U16, mesa.TYPE_U32, mesa.TYPE_S16, mesa.TYPE_S32):
        if rnd != mesa.ROUND_ZERO: raise RuntimeError(f"unhandled ir3 cov round {ins}")
        ti = UOp(Ops.TRUNC, src=(f,)).cast(dtypes.int64).cast(dtypes.uint32)
        self.wr(ins, ti & _c(0xffff) if dt in (mesa.TYPE_U16, mesa.TYPE_S16) else ti)
      else: raise RuntimeError(f"unhandled ir3 {self.ins_name} {ins}")
    else:
      sint = st in (mesa.TYPE_S16, mesa.TYPE_S32)
      if st == mesa.TYPE_U16: src = src & _c(0xffff)
      elif st == mesa.TYPE_S16: src = (src & _c(0xffff)).cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32).bitcast(dtypes.uint32)
      elif st == mesa.TYPE_S32: pass
      elif st in (mesa.TYPE_U8, mesa.TYPE_U8_32):
        src = src & _c(0xff)
        if dt in (mesa.TYPE_S16, mesa.TYPE_S32, mesa.TYPE_F16, mesa.TYPE_F32):
          src = ((src << _c(24)).bitcast(dtypes.int32) >> _c(24, dtypes.int32)).bitcast(dtypes.uint32)
          sint = True
      if dt == mesa.TYPE_F16:
        si = src.bitcast(dtypes.int32) if sint else src
        self.wr(ins, si.cast(dtypes.float32).cast(dtypes.half))
      elif dt == mesa.TYPE_F32:
        si = src.bitcast(dtypes.int32) if sint else src
        self.wr(ins, si.cast(dtypes.float32))
      else: self.wr(ins, src)

  def _memsz(self, ins):
    return int(ins.get("SIZE", 1)), _TSIZE[ins.get("TYPE", mesa.TYPE_U32)], _half_mem(ins)

  def _ldg(self, ins, raw):
    n, ts, hm = self._memsz(ins)
    addr, dst = self.gaddr(self.ridx("SRC1"), self.pval["OFF"]), self.ridx("DST")
    for i in range(n): self.wr_reg(hm, dst + _c(i, dtypes.int), self._ld_vmem(addr + _c(i * ts, dtypes.uint64), ts))

  def _stg(self, ins, raw):
    n, ts, hm = self._memsz(ins)
    data, addr = self.ridx("SRC3"), self.gaddr(self.ridx("SRC1"), self.pval["OFF"])
    for i in range(n):
      di = data + _c(i, dtypes.int)
      self._st_vmem(addr + _c(i * ts, dtypes.uint64), ts, self.h_r(di) if hm else self.gpr_r(di))

  def _off13(self) -> UOp:
    o = self.pval["OFF"] & _c(0x1fff)
    return (o ^ _c(0x1000)) - _c(0x1000)

  def _st_local(self, ins, raw):
    n, ts, hm = self._memsz(ins)
    a0, data = self.gpr_r(self.ridx("DST")) + self._off13(), self.ridx("SRC")
    for i in range(n):
      val = self.h_r(data + _c(i, dtypes.int)) if hm else self.gpr_r(data + _c(i, dtypes.int))
      if self.ins_name == "stl":
        self.lds = self._st_bytes(self.lds, (a0 + _c(i * ts)).cast(dtypes.int) & _c(CS_SHARED_MEM_SIZE - 1, dtypes.int), ts, val)
      else:
        self.pvt = self._st_bytes(self.pvt, self.tid * 32768 + (a0 + _c(i * ts)).cast(dtypes.int), ts, val)

  def _ld_local(self, ins, raw):
    n, ts, hm = self._memsz(ins)
    a0, dst = self.gpr_r(self.ridx("SRC")) + self._off13(), self.ridx("DST")
    for i in range(n):
      if self.ins_name == "ldl": val, _ = self._ld_bytes(self.lds, (a0 + _c(i * ts)).cast(dtypes.int) & _c(CS_SHARED_MEM_SIZE - 1, dtypes.int), ts)
      else: val, _ = self._ld_bytes(self.pvt, self.tid * 32768 + (a0 + _c(i * ts)).cast(dtypes.int), ts)
      self.wr_reg(hm, dst + _c(i, dtypes.int), val)

  _OPS = {
    **dict.fromkeys(_ALU_F, _alu_f),
    **dict.fromkeys(_UNARY_F1, _unary_f1),
    **dict.fromkeys(_UNARY_F, _unary_f),
    **dict.fromkeys((*_ALU_I, "min.s", "max.s", "mul.s24"), _alu_i),
    **dict.fromkeys(("cmps.u", "cmps.s", "cmps.f"), _cmps),
    **dict.fromkeys(("sel.b32", "sel.b16"), _sel),
    **dict.fromkeys(("shrg", "shlg", "shrm", "shlm"), _shift3),
    "absneg.f": _absneg_f,
    "absneg.s": _absneg_s, "not.b": _not_b, "clz.b": _clz_b,
    "andg": _andg, "madsh.m16": _madsh,
    "mad.f32": _mad_f32, "mad.f16": _mad_f16,
    "swz": _swz, "cov": _cov,
    "ldg": _ldg, "stg": _stg,
    "stl": _st_local, "stp": _st_local,
    "ldl": _ld_local, "ldp": _ld_local,
  }

  def exec_ins(self, name: str, ins: dict, raw: dict):
    self.sat, self.ins_name = bool(raw.get("SAT")), name
    self.half = (not name.startswith("cmps")) and name.endswith(".f") and (
      any(isinstance(v, dict) and v.get("HALF") for v in ins.values()) or ins.get("DST_HALF"))
    if self.half and name not in _HALF_OK: raise RuntimeError(f"unhandled ir3 half {name} {ins}")
    if (fn:=self._OPS.get(name)) is None: raise RuntimeError(f"unhandled ir3 {name} {ins}")
    fn(self, ins, raw)

### COMPILE

def _compile(raw: dict, nt: int) -> UOp:
  name = _iname(raw)
  if name in _CTRL or name == "getone":
    raise RuntimeError(f"not a compiled op {name}")
  nrep = (raw.get("REPEAT") or 0) + 1
  if name in _MEMOPS and nrep != 1:
    raise RuntimeError(f"unhandled ir3 rpt {name} {raw}")
  ctx = _Ctx(nt)
  ctx.bind(raw)
  for rpt in range(nrep):
    ctx.rpt = rpt
    ctx.exec_ins(name or "", raw, raw)
  if not ctx.stores: raise RuntimeError(f"ir3 uop produced no stores {name} {raw}")
  kn = "ir3w_" + hashlib.sha1(repr((_kind(raw), nt)).encode()).hexdigest()[:12]
  body = UOp.sink(UOp.group(*ctx.stores).end(ctx.tid), ctx.wst.index(0).store(ctx._pc + _c(1)))
  return body.replace(arg=KernelInfo(name=kn)).rtag(1)

def _make_enc(instrs: list[dict]) -> tuple[object, int]:
  n = max(len(instrs), 1)
  arr = (ctypes.c_uint32 * (n * _ENC_STRIDE))()
  for pc, raw in enumerate(instrs):
    packed, base = _pack_named(raw), pc * _ENC_STRIDE
    for k, v in packed.items(): arr[base + _ENC_SLOT[k]] = v
  return arr, ctypes.addressof(arr)

_runners: dict[tuple, tuple] = {}
def _get_runner(raw: dict, nt: int):
  key = (_kind(raw), nt)
  if key not in _runners:
    name = _iname(raw)
    try:
      sink = _compile(raw, nt)
      with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        prg = to_program(sink, Device["CPU"].renderer)
        runtime = get_runtime("CPU", prg)
    except Exception as e:
      raise RuntimeError(f"ir3 uop compile {name} {raw}: {type(e).__name__}: {e}") from e
    gl = tuple(prg.arg.globals)
    _runners[key] = (runtime.fxn, gl, gl == tuple(range(8)))
  return _runners[key]

### WORKGROUP

class _WorkGroup:
  def __init__(self, nt: int):
    self.nt = nt
    self._maps: list = []
    def hb(n, fmt):
      addr, m, mv = _host_buf(n, fmt)
      self._maps.append(m)
      return addr, mv
    self.gpr_addr, self.gpr_mv = hb(nt * 256, "I")
    self.h_addr, _ = hb(nt * 256, "I")
    self.lds_addr, _ = hb(CS_SHARED_MEM_SIZE, "B")
    self.pvt_addr, _ = hb(nt * 32768, "B")
    self.ca_addr, self.ca_mv = hb(1, "Q")

  def reset(self, lds=False, pvt=False):
    ctypes.memset(self.gpr_addr, 0, self.nt * 256 * 4)
    ctypes.memset(self.h_addr, 0, self.nt * 256 * 4)
    if lds: ctypes.memset(self.lds_addr, 0, CS_SHARED_MEM_SIZE)
    if pvt: ctypes.memset(self.pvt_addr, 0, self.nt * 32768)

  def c_bufs(self, base: int, wst_addr: int, enc_addr: int) -> list:
    return [ctypes.c_uint64(a) for a in (
      self.gpr_addr + base * 256 * 4, self.h_addr + base * 256 * 4, self.lds_addr, self.pvt_addr + base * 32768,
      0, self.ca_addr, wst_addr, enc_addr)]

_wg_cache: dict[int, _WorkGroup] = {}
def _wg(nt: int) -> _WorkGroup:
  if nt not in _wg_cache: _wg_cache[nt] = _WorkGroup(nt)
  return _wg_cache[nt]

def _run_wave_op(raw: dict, nt: int, c_bufs: list):
  fxn, gl, contig = _get_runner(raw, nt)
  fxn(*c_bufs) if contig else fxn(*[c_bufs[g] for g in gl])

def _pred_taken(gpr_mv, tid: int, ins: dict, invk: str, compk: str) -> bool:
  t = gpr_mv[tid * 256 + 248 + int(ins.get(compk, 0))]
  return (not t) if ins.get(invk) else bool(t)

### WAVE STATE

_WPC, _WACT, _WPMO, _WPMA, _WPPC = 0, 8, 16, 24, 32
_WPMK = _WPPC + PARK * 4
_WSP, _WFLG, _WBYTES = _WPMK + PARK * 8, _WPMK + PARK * 8 + 4, _WPMK + PARK * 8 + 8

def _wmv(addr: int, off: int, n: int, fmt: Literal["I", "Q"]):
  return to_mv(addr + off, n * {"I": 4, "Q": 8}[fmt]).cast(fmt)

class _Wave:
  __slots__ = ("base", "nlanes", "all", "keep", "addr", "c_bufs", "park_limit",
               "pc_mv", "act_mv", "pmode_mv", "pmask_mv", "ppc_mv", "pmsk_mv", "sp_mv", "flg_mv")
  def __init__(self, base: int, nlanes: int):
    self.base, self.nlanes, self.all, self.keep = base, nlanes, (1 << nlanes) - 1, []
    self.c_bufs: list|None = None
    self.park_limit = PARK
    addr, m, _ = _host_buf(_WBYTES, "B")
    self.addr = addr
    self.keep.append(m)
    self.pc_mv, self.act_mv = _wmv(addr, _WPC, 1, "I"), _wmv(addr, _WACT, 1, "Q")
    self.pmode_mv, self.pmask_mv = _wmv(addr, _WPMO, 1, "I"), _wmv(addr, _WPMA, 1, "Q")
    self.ppc_mv, self.pmsk_mv = _wmv(addr, _WPPC, PARK, "I"), _wmv(addr, _WPMK, PARK, "Q")
    self.sp_mv, self.flg_mv = _wmv(addr, _WSP, 1, "I"), _wmv(addr, _WFLG, 1, "I")
    self.reset()

  @property
  def exec_mask(self) -> int:
    em, mode, pmask = int(self.act_mv[0]), int(self.pmode_mv[0]), int(self.pmask_mv[0])
    if mode == 1: em &= pmask
    elif mode == 2: em &= ~pmask
    return em & self.all

  def reset(self):
    self.pc_mv[0], self.act_mv[0], self.sp_mv[0], self.flg_mv[0] = 0, self.all, 0, 0
    self.pmode_mv[0], self.pmask_mv[0] = 0, 0

  def wake(self, pc: int, jp=False):
    sp, act, n = int(self.sp_mv[0]), int(self.act_mv[0]), 0
    pcs, ms = [0]*PARK, [0]*PARK
    for i in range(sp):
      ppc, m = int(self.ppc_mv[i]), int(self.pmsk_mv[i])
      if ppc == (pc & 0xffffffff) or (jp and ppc == _JP): act |= m
      else:
        pcs[n], ms[n], n = ppc, m, n+1
    self.act_mv[0], self.sp_mv[0] = act, n
    for i in range(n): self.ppc_mv[i], self.pmsk_mv[i] = pcs[i], ms[i]

  def pop_parked(self) -> bool:
    sp = int(self.sp_mv[0])
    if sp == 0: return False
    sp -= 1
    self.pc_mv[0], self.act_mv[0], self.sp_mv[0] = int(self.ppc_mv[sp]), int(self.pmsk_mv[sp]), sp
    return True

  def park(self, dest: int, mask: int):
    dest = dest & 0xffffffff
    sp = int(self.sp_mv[0])
    for i in range(sp):
      if int(self.ppc_mv[i]) == dest:
        self.pmsk_mv[i] = int(self.pmsk_mv[i]) | mask
        return
    if sp >= self.park_limit: raise RuntimeError("ir3 park overflow")
    self.ppc_mv[sp], self.pmsk_mv[sp], self.sp_mv[0] = dest, mask, sp + 1

  def goto(self, dest: int, act: int, signed_off: int):
    dest, act = dest & 0xffffffff, act & self.all
    if signed_off <= 0:
      self.pc_mv[0], self.act_mv[0] = dest, act
      return
    sp, cur = int(self.sp_mv[0]), int(self.pc_mv[0])
    rp = None
    for i in range(sp):
      ppc = int(self.ppc_mv[i])
      if cur < ppc < dest and (rp is None or ppc < rp): rp = ppc
    if rp is None:
      self.pc_mv[0], self.act_mv[0] = dest, act
      return
    wm, n, pcs, ms = 0, 0, [0]*PARK, [0]*PARK
    for i in range(sp):
      ppc, m = int(self.ppc_mv[i]), int(self.pmsk_mv[i])
      if ppc == rp: wm |= m
      else: pcs[n], ms[n], n = ppc, m, n+1
    for i in range(n): self.ppc_mv[i], self.pmsk_mv[i] = pcs[i], ms[i]
    self.sp_mv[0] = n
    self.park(dest, act)
    self.pc_mv[0], self.act_mv[0] = rp, wm & self.all

def _ballot(wave: _Wave, wg, raw: dict, name: str|None) -> int:
  taken, em = 0, wave.exec_mask
  for i in range(wave.nlanes):
    if not (em & (1 << i)): continue
    t = _pred_taken(wg.gpr_mv, wave.base + i, raw, "INV1", "COMP1")
    if name != "br":
      t2 = _pred_taken(wg.gpr_mv, wave.base + i, raw, "INV2", "COMP2")
      t = (t or t2) if name == "brao" else (t and t2)
    if t: taken |= 1 << i
  return taken

def _snap_p0(wave: _Wave, wg) -> int:
  bits, em = 0, wave.exec_mask
  for i in range(wave.nlanes):
    if (em & (1 << i)) and wg.gpr_mv[(wave.base + i) * 256 + 248]: bits |= 1 << i
  return bits

def _exec_cat0(wave: _Wave, wg, raw: dict, name: str|None) -> str|None:
  pc, act = int(wave.pc_mv[0]), int(wave.act_mv[0])
  if name in ("nop", None):
    wave.pc_mv[0] = (pc + 1) & 0xffffffff
  elif name == "bar":
    wave.pc_mv[0] = (pc + 1) & 0xffffffff
    wave.flg_mv[0] = int(wave.flg_mv[0]) | BAR
    return "bar"
  elif name == "end":
    if not wave.pop_parked():
      wave.act_mv[0], wave.flg_mv[0] = 0, DONE
  elif name == "jump":
    off = _s32(raw["IMMED"])
    wave.goto(pc + off, act, off)
  elif name in ("br", "brao", "braa"):
    off, em = _s32(raw["IMMED"]), wave.exec_mask
    taken = _ballot(wave, wg, raw, name)
    tpc, fpc, fall, idle = (pc + off) & 0xffffffff, (pc + 1) & 0xffffffff, em ^ taken, act & ~em
    both = taken != 0 and fall != 0
    if both:
      if off <= 1:
        wave.park(fpc, fall)
        wave.pc_mv[0], wave.act_mv[0] = tpc, taken | idle
      else:
        wave.park(tpc, taken)
        wave.pc_mv[0], wave.act_mv[0] = fpc, fall | idle
    else:
      wave.goto(tpc if taken else fpc, act, off)
  elif name in ("predt", "predf"):
    wave.pmask_mv[0], wave.pmode_mv[0] = _snap_p0(wave, wg), 1 if name == "predt" else 2
    wave.pc_mv[0] = (pc + 1) & 0xffffffff
  elif name == "prede":
    wave.pmode_mv[0] = 0
    wave.pc_mv[0] = (pc + 1) & 0xffffffff
  elif name == "getone":
    off, em = _s32(raw["IMMED"]), wave.exec_mask
    one = (em & -em) & wave.all
    rest = em & ~one
    if em:
      if rest: wave.park(_JP, rest)
      wave.goto(pc + off, one | (act & ~em), off)
    else:
      wave.pc_mv[0] = (pc + 1) & 0xffffffff
  else:
    raise RuntimeError(f"unhandled ir3 {name} {raw}")
  return None

def _is_cat0(raw: dict, name: str|None) -> bool:
  return name in _CTRL or (name in ("nop", None) and "SRC_TYPE" not in raw) or name == "getone"

def _step_wave(wave: _Wave, instrs: list[dict], wg, nins: int):
  for _ in range(1_000_000):
    flg = int(wave.flg_mv[0])
    if flg & DONE or flg & BAR: return
    if int(wave.act_mv[0]) == 0 and not wave.pop_parked():
      wave.flg_mv[0] = DONE
      return
    pc = int(wave.pc_mv[0])
    if pc < 0 or pc >= nins:
      if wave.pop_parked(): continue
      wave.flg_mv[0] = DONE
      return
    raw = instrs[pc]
    wave.wake(pc, jp=bool(raw.get("JP")))
    name = _iname(raw)
    if _is_cat0(raw, name):
      if _exec_cat0(wave, wg, raw, name) == "bar": return
    else:
      _run_wave_op(raw, wave.nlanes, unwrap(wave.c_bufs))
  raise RuntimeError("ir3 wave exceeded 1M instructions")

def _run_waves(waves: list[_Wave], instrs: list[dict], wg):
  nins = len(instrs)
  for _ in range(10_000_000):
    if all(int(w.flg_mv[0]) & DONE for w in waves): return
    wait = [w for w in waves if not (int(w.flg_mv[0]) & DONE)]
    if wait and all(int(w.flg_mv[0]) & BAR for w in wait):
      for w in wait: w.flg_mv[0] = int(w.flg_mv[0]) & ~BAR
    for w in wait: _step_wave(w, instrs, wg, nins)
  raise RuntimeError("ir3 wave exceeded 10M barrier rounds")

_wave_cache: dict[int, list[_Wave]] = {}
def _waves(wg) -> list[_Wave]:
  if (ws:=_wave_cache.get(wg.nt)) is None:
    _wave_cache[wg.nt] = ws = [_Wave(b, min(WAVE, wg.nt - b)) for b in range(0, wg.nt, WAVE)]
  else:
    for w in ws: w.reset()
  return ws

def run_shader(base: int, sz: int, const_addr: int, regs: dict[int, int], branchstack: int):
  instrs = decode_shader(bytes(to_mv(base, sz)))
  ls, ng, lid, wgid, wgsz = cs_ndrange(regs)
  nt = ls[0] * ls[1] * ls[2]
  if nt == 0: return
  wg = _wg(nt)
  names = {_iname(i) for i in instrs}
  _enc_addr_gc_pin, enc_addr = _make_enc(instrs)
  wg.ca_mv[0] = const_addr
  lids = list(itertools.product(range(ls[2]), range(ls[1]), range(ls[0])))
  park = min(PARK, branchstack)
  for gz, gy, gx in itertools.product(range(ng[2]), range(ng[1]), range(ng[0])):
    cs_init_consts(const_addr, wgsz, ls, ng)
    wg.reset(lds=bool(names & {"ldl", "stl"}), pvt=bool(names & {"ldp", "stp"}))
    for tid, (lz, ly, lx) in enumerate(lids):
      cs_init_ids(wg.gpr_mv[tid * 256:(tid + 1) * 256], lid, wgid, lx, ly, lz, gx, gy, gz)
    waves = _waves(wg)
    for w in waves:
      w.c_bufs, w.park_limit = wg.c_bufs(w.base, w.addr, enc_addr), park
    _run_waves(waves, instrs, wg)
