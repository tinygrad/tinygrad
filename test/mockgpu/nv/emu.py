# PTX emulator used by mockgpu, replaces gpuocelot.
# PTX is parsed once per module, then executed SIMT style: every register is a numpy array with one lane per thread and the
# whole grid runs at once. Lanes are stepped by the lowest program counter first, which reconverges divergent branches and
# makes bar.sync a no-op (a barrier can only be the lowest pc once every lane has reached it).
# All memory (params, shared, globals) is real host memory, so addresses in registers are plain host pointers.
from __future__ import annotations
import ctypes, functools, math, re
from typing import Callable
import numpy as np

# ***** parsing *****

class Inst:
  __slots__ = ("pred", "neg", "op", "src", "line")
  def __init__(self, pred:str|None, neg:bool, op:list[str], src:list, line:str):
    self.pred, self.neg, self.op, self.src, self.line = pred, neg, op, src, line
  def __repr__(self): return self.line

def _split(s:str, sep:str) -> list[str]:
  # split on sep at bracket depth 0
  ret, depth, cur = [], 0, ""
  for c in s:
    if c in "[{(": depth += 1
    elif c in "]})": depth -= 1
    if c == sep and depth == 0:
      ret.append(cur)
      cur = ""
    else: cur += c
  if cur.strip(): ret.append(cur)
  return [x.strip() for x in ret]

def _statements(body:str):
  # yields statements, "{" and "}" scope markers. braces inside a statement (vector operands) stay in the statement.
  cur = ""
  for c in body:
    if c in "{}" and not cur.strip(): yield c
    elif c == ";" or (c == ":" and re.fullmatch(r"[\w$]+", cur.strip())):  # a label ends a statement even without a semicolon
      yield cur.strip() + (":" if c == ":" else "")
      cur = ""
    else: cur += c

def _imm(tok:str):
  if tok.startswith("0f") or tok.startswith("0F"): return int(tok[2:], 16)
  if tok.startswith("0d") or tok.startswith("0D"): return int(tok[2:], 16)
  if re.fullmatch(r"[-+]?\d*\.\d*([eE][-+]?\d+)?", tok) or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", tok): return ("f", float(tok))
  return int(tok[:-1] if tok[-1] in "Uu" else tok, 0)  # only U is a suffix, F is a hex digit

def _operand(tok:str):
  tok = tok.strip()
  if tok.startswith("{"): return ("vec", [_operand(x) for x in _split(tok[1:tok.rindex("}")], ",")])
  if tok.startswith("["):
    inner = tok[1:tok.rindex("]")].strip()
    # the offset carries its own sign, so "[%rd1+-1024]" is a valid way to write a negative displacement
    if (m := re.fullmatch(r"(.+?)\s*([-+])\s*([-+]?\w+)\s*", inner)):
      return ("mem", _operand(m.group(1)), int(m.group(3), 0) * (-1 if m.group(2) == "-" else 1))
    return ("mem", _operand(inner), 0)
  if tok.startswith("%"):
    if (name := tok[1:]).split(".")[0] in SREGS: return ("sreg", name)
    return ("reg", name)
  if (m := re.fullmatch(r"([\w$]+)\s*\[\s*(\d+)\s*\]", tok)): return ("sym", m.group(1), int(m.group(2)))
  # a ptx identifier can never start with a digit or a sign, so anything that does is a literal
  return ("imm", _imm(tok)) if tok[0].isdigit() or tok[0] in "+-" else ("sym", tok, 0)

SREGS = {"tid", "ntid", "ctaid", "nctaid", "laneid", "warpid", "nwarpid", "gridid", "smid", "nsmid", "clock", "clock64",
         "lanemask_eq", "lanemask_le", "lanemask_lt", "lanemask_ge", "lanemask_gt", "warpsize", "envreg0", "dynamic_smem_size"}

DECL = re.compile(r"^\.(?:extern\s+|visible\s+|weak\s+)*\.?(shared|global|const|local)\b.*?\.(?:align\s+\d+\s+)?"
                  r"\.(\w+)\s+([\w$]+)(?:\[(\d*)\])?(?:\s*=\s*(.+))?$", re.S)
TYPE_SZ = {"b8": 1, "s8": 1, "u8": 1, "b16": 2, "s16": 2, "u16": 2, "f16": 2, "b32": 4, "s32": 4, "u32": 4, "f32": 4,
           "b64": 8, "s64": 8, "u64": 8, "f64": 8, "pred": 1, "f16x2": 4, "b128": 16}

class Kernel:
  def __init__(self, name:str, params:list[tuple[str,int]], insts:list[Inst], labels:dict[str,int], decls:list[tuple[str,str,int,bytes]],
               regs:dict[str,int]):
    self.name, self.params, self.insts, self.labels, self.decls, self.regs = name, params, insts, labels, decls, regs
    self.statics: dict[str, ctypes.Array] = {}  # module scope .global/.const, allocated once and kept

def _parse_decl(st:str) -> tuple[str,str,int,bytes]|None:
  if (m := DECL.match(st.strip())) is None: return None
  space, ty, name, cnt, init = m.groups()
  sz = TYPE_SZ.get(ty, 1) * (int(cnt) if cnt else 1)
  data = b""
  if init is not None and init.strip().startswith("{"):
    vals = [int(x.strip(), 0) for x in _split(init.strip()[1:init.rindex("}")], ",") if x.strip()]
    data = b"".join(v.to_bytes(TYPE_SZ.get(ty, 1), "little", signed=v < 0) for v in vals)
  return space, name, sz, data

REG = re.compile(r"^\.reg\s*\.(\w+)\s+(.+)$")  # nvrtc writes ".reg.b16" with no space in inline asm

def _parse_regs(st:str, regs:dict[str,int]):
  if (m := REG.match(st)) is None: return
  w = 1 if m.group(1) == "pred" else TYPE_SZ[m.group(1)] * 8
  for nm in _split(m.group(2), ","):
    if (mm := re.fullmatch(r"([\w$]+)<(\d+)>", nm := nm.strip().lstrip("%"))): regs.update({f"{mm.group(1)}{i}": w for i in range(int(mm.group(2)))})
    else: regs[nm] = w

def _parse_body(body:str) -> tuple[list[Inst], dict[str,int], list[tuple[str,str,int,bytes]], dict[str,int]]:
  insts:list[Inst] = []
  labels:dict[str,int] = {}
  decls:list[tuple[str,str,int,bytes]] = []
  regs:dict[str,int] = {}
  for st in _statements(body):
    st = " ".join(st.split())
    if not st or st in "{}": continue
    while (m := re.match(r"^([\w$]+)\s*:", st)):  # labels
      labels[m.group(1)] = len(insts)
      st = st[m.end():].strip()
    if not st: continue
    if st.startswith("."):
      if (d := _parse_decl(st)) is not None: decls.append(d)
      else: _parse_regs(st, regs)
      continue
    pred, neg = None, False
    if st.startswith("@"):
      p, st = st[1:].split(None, 1)
      neg, pred = p.startswith("!"), p.lstrip("!%")
    # the opcode is the leading run of name characters, nvrtc does not always put a space before the first operand
    op = re.match(r"[\w.]+", st).group(0)  # type: ignore[union-attr]
    insts.append(Inst(pred, neg, op.split("."), [_operand(x) for x in _split(st[len(op):], ",")], st))
  return insts, labels, decls, regs

def _as_reg(o, regs:dict[str,int], named:set[str]):
  if o[0] == "vec": return ("vec", [_as_reg(x, regs, named) for x in o[1]])
  if o[0] == "mem": return ("mem", _as_reg(o[1], regs, named), o[2])
  return ("reg", o[1]) if o[0] == "sym" and o[1] in regs and o[1] not in named else o

ENTRY = re.compile(r"\.entry\s+([\w$]+)\s*(?:\(([^)]*)\))?([^{]*)\{", re.S)

@functools.lru_cache(maxsize=256)
def parse_ptx(src:str) -> Kernel:
  src = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", src, flags=re.S))
  if (m := ENTRY.search(src)) is None: raise RuntimeError("no .entry in ptx")
  params = []
  for p in _split(m.group(2) or "", ","):
    if (pm := re.search(r"\.(\w+)\s+([\w$]+)(?:\[(\d*)\])?$", p.strip())) is None: raise RuntimeError(f"bad param {p}")
    params.append((pm.group(2), TYPE_SZ.get(pm.group(1), 8) * (int(pm.group(3)) if pm.group(3) else 1)))
  depth, i = 0, m.end() - 1
  while True:
    if src[i] == "{": depth += 1
    elif src[i] == "}" and (depth := depth - 1) == 0: break
    i += 1
  insts, labels, decls, regs = _parse_body(src[m.end():i])
  for st in _statements(src[:m.start()]):  # module scope declarations
    if (d := _parse_decl(st.strip())) is not None: decls.append(d)
  # nvrtc's inline asm declares its own temporaries with no % prefix, so they parse as symbols until the .reg is seen.
  # names that a param or a memory declaration already owns stay symbols.
  named = {n for n, _ in params} | {d[1] for d in decls}
  for ins in insts: ins.src = [_as_reg(o, regs, named) for o in ins.src]
  return Kernel(m.group(1), params, insts, labels, decls, regs)


# ***** values *****

UI = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
UI_CT = {8: ctypes.c_uint8, 16: ctypes.c_uint16, 32: ctypes.c_uint32, 64: ctypes.c_uint64}
SI = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
FL = {16: np.float16, 32: np.float32, 64: np.float64}

@functools.lru_cache(maxsize=None)
def _ty(s:str) -> tuple[str,int]:
  # "B" is bfloat16, which numpy has no dtype for: it is kept as 16 raw bits and widened to f32 to compute with
  return {"pred": ("p", 1), "f16x2": ("b", 32), "bf16": ("B", 16)}.get(s) or (s[0], int(s[1:]))
def _npty(kind:str, w:int): return np.float32 if kind == "B" else FL[w] if kind == "f" else SI[w] if kind == "s" else UI[w]
def _isfloat(kind:str) -> bool: return kind in "fB"

def _bf16_to_f32(bits:np.ndarray) -> np.ndarray: return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
def _f32_to_bf16(v:np.ndarray) -> np.ndarray:
  u = np.asarray(v, dtype=np.float32).view(np.uint32)
  rounded = (u + np.uint32(0x7fff) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)  # round to nearest even
  return np.where(np.isnan(v), np.uint32(0x7fc0), rounded).astype(np.uint16)
def _bits(ty:tuple[str,int]) -> tuple[str,int]: return ty if ty[0] == "p" else ("b", ty[1])  # a predicate has no bit pattern form
def _regty(w:int): return UI[max(w, 8)]

def _immval(v, kind:str, w:int):
  if kind == "p": return np.bool_(v if isinstance(v, int) else v[1])
  if isinstance(v, tuple): return _npty(kind, w)(v[1])  # a decimal literal is a value, not a bit pattern
  return np.array(v & ((1 << w) - 1), dtype=UI[w]).view(_npty(kind, w))

class Ctx:
  """One vectorized launch chunk: nthreads lanes, each register an array of that length."""
  def __init__(self, kern:Kernel, sym:dict, sregs:dict, n:int):
    self.kern, self.sym, self.sregs, self.n = kern, sym, sregs, n
    self.regs:dict[str, np.ndarray] = {}

  def arr(self, name:str) -> np.ndarray:
    if (a := self.regs.get(name)) is None: a = self.regs[name] = np.zeros(self.n, dtype=_regty(self.kern.regs.get(name, 64)))
    return a

  def count(self, m) -> int: return self.n if m is None else int(m.sum())

  def rd(self, o, ty:tuple[str,int], m):
    kind, w = ty
    if o[0] == "imm": return _immval(o[1], kind, w)
    if o[0] == "reg": v = self.arr(o[1]) if m is None else self.arr(o[1])[m]
    elif o[0] == "sreg": v = self._sel(self.sregs[o[1]], m)
    elif o[0] == "sym": v = self._sel(self.sym[o[1]], m) + np.uint64(o[2])
    else: raise RuntimeError(f"cannot read operand {o}")
    if kind == "p": return v.astype(np.uint8).astype(bool)  # predicates live one per byte
    if kind == "B": return _bf16_to_f32(v.astype(np.uint16))
    return v.astype(UI[w]).view(_npty(kind, w))  # a register may be wider than the instruction reads

  def _sel(self, v, m): return (v if m is None else v[m]) if isinstance(v, np.ndarray) else np.uint64(v)

  def addr(self, o, m) -> np.ndarray:
    return np.broadcast_to(self.rd(o[1], ("u", 64), m), (self.count(m),)) + np.uint64(o[2] & 0xffffffffffffffff)

  def wr(self, o, v, ty:tuple[str,int], m):
    kind, w = ty
    if kind == "p": bits = np.asarray(v, dtype=bool).astype(np.uint8)
    elif kind == "B": bits = _f32_to_bf16(np.asarray(v, dtype=np.float32))
    elif kind == "f": bits = np.asarray(v, dtype=FL[w]).view(UI[w])
    elif kind == "s": bits = np.asarray(v).astype(SI[w]).view(UI[w])
    else: bits = np.asarray(v).astype(UI[w])
    a = self.arr(o[1])
    if (rw := a.dtype.itemsize * 8) > w and kind != "p":  # writing a narrow signed value into a wider register sign extends
      bits = bits.view(SI[w]).astype(SI[rw]).view(UI[rw]) if kind == "s" else bits.astype(UI[rw])
    if m is None: a[:] = bits
    else: a[m] = bits

# ***** memory *****
# every buffer lives in real host memory, so a load is a gather over the byte range the active lanes touch

def _span(addrs:np.ndarray, n:int) -> tuple[np.ndarray, np.ndarray, bool]:
  """A view of the memory the lanes touch, the per lane index into it, and whether that view is already n byte typed."""
  lo = int(addrs.min())
  off = (addrs - np.uint64(lo)).astype(np.int64)
  hi = lo + int(off.max()) + n
  # naturally aligned accesses, which is nearly all of them, index a typed view directly instead of gathering bytes
  if lo % n == 0 and not (off % n).any():
    return np.ctypeslib.as_array(ctypes.cast(lo, ctypes.POINTER(UI_CT[n * 8])), shape=((hi - lo) // n,)), off // n, True
  buf = np.ctypeslib.as_array((ctypes.c_ubyte * (hi - lo)).from_address(lo))
  return buf, off[:, None] + np.arange(n, dtype=np.int64), False

def _gather(addrs:np.ndarray, n:int) -> np.ndarray:
  buf, idx, typed = _span(addrs, n)
  return buf[idx] if typed else np.ascontiguousarray(buf[idx]).view(UI[n * 8]).reshape(-1)

def _scatter(addrs:np.ndarray, n:int, vals:np.ndarray):
  buf, idx, typed = _span(addrs, n)
  if typed: buf[idx] = vals.astype(UI[n * 8])
  else: buf[idx] = np.ascontiguousarray(vals.astype(UI[n * 8])).view(np.uint8).reshape(-1, n)

# ***** instructions *****

def _idiv(a:np.ndarray, b:np.ndarray, signed:bool) -> np.ndarray:
  nz = b != 0
  bs = np.where(nz, b, np.ones_like(b))
  q = a // bs
  # numpy floors, ptx truncates toward zero: they differ by one whenever the signs differ and the division is inexact.
  # done without abs() because abs(INT_MIN) overflows.
  if signed: q = (q + (((a % bs) != 0) & ((a < 0) != (bs < 0)))).astype(a.dtype)
  return np.where(nz, q, np.zeros_like(q))

def _mulhi64(a:np.ndarray, b:np.ndarray, signed:bool) -> np.ndarray:
  """The high 64 bits of a 64x64 product, built from 32 bit halves because numpy has no 128 bit integer."""
  ua, ub = a.astype(np.uint64), b.astype(np.uint64)
  msk, sh = np.uint64(0xffffffff), np.uint64(32)
  a0, a1, b0, b1 = ua & msk, ua >> sh, ub & msk, ub >> sh
  t = a1 * b0 + ((a0 * b0) >> sh)
  hi = a1 * b1 + (t >> sh) + ((a0 * b1 + (t & msk)) >> sh)
  if signed:  # fix up the unsigned result: subtract the operand whose partner is negative
    hi = hi - np.where(a < 0, ub, np.uint64(0)) - np.where(b < 0, ua, np.uint64(0))
    return hi.view(np.int64)
  return hi

def _shift(a:np.ndarray, b:np.ndarray, w:int, right:bool, signed:bool) -> np.ndarray:
  if int(b.max()) < w:  # the common case needs no clamping and no masking of over wide shifts
    sh = b.astype(a.dtype)
    return (a >> sh) if right else (a << sh)
  sh = np.minimum(b.astype(np.uint64), np.uint64(w - 1)).astype(np.uint64)
  v = (a >> sh.astype(a.dtype)) if right else (a << sh.astype(a.dtype))
  # shifting by more than the width gives 0, except for an arithmetic shift which saturates to the sign
  return v if right and signed else np.where(b.astype(np.uint64) >= np.uint64(w), np.zeros_like(v), v)

def _bitlen(a:np.ndarray, w:int) -> np.ndarray:
  x = a.copy()
  for s in [1 << i for i in range(w.bit_length() - 1)]: x |= x >> np.array(s, dtype=a.dtype)
  return np.bitwise_count(x)

def _compare(cmp:str, a:np.ndarray, b:np.ndarray, kind:str) -> np.ndarray:
  if not _isfloat(kind): return CMP[cmp](a, b)
  nan = np.isnan(a) | np.isnan(b)  # ptx has ordered and unordered float comparisons, numpy only has ordered ones
  return (CMP[UCMP[cmp]](a, b) | nan) if cmp in UCMP else nan if cmp == "nan" else ~nan if cmp == "num" else (CMP[cmp](a, b) & ~nan)

CMP:dict[str,Callable] = {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal, "gt": np.greater,
                          "ge": np.greater_equal, "lo": np.less, "ls": np.less_equal, "hi": np.greater, "hs": np.greater_equal}
UCMP = {"equ": "eq", "neu": "ne", "ltu": "lt", "leu": "le", "gtu": "gt", "geu": "ge"}
BOOL:dict[str,Callable] = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}
RND:dict[str,Callable] = {"rzi": np.trunc, "rmi": np.floor, "rpi": np.ceil, "rni": np.rint}
FLOAT1:dict[str,Callable] = {"sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x), "rcp": lambda x: 1.0 / x, "ex2": np.exp2,
                             "lg2": np.log2, "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "abs": np.abs, "neg": np.negative}
ATOMIC:dict[str,Callable] = {"add": lambda o,b: o + b, "min": np.minimum, "max": np.maximum, "and": np.bitwise_and,
                             "or": np.bitwise_or, "xor": np.bitwise_xor, "exch": lambda o,b: b}
IGNORE = ("membar", "fence", "nanosleep", "pmevent", "trap", "prefetch", "prefetchu", "griddepcontrol", "nop", "bar", "barrier")

def _step(ctx:Ctx, ins:Inst, m) -> int|None:
  """Runs one instruction for the lanes in m. Returns a branch target pc, -1 to retire the lanes, or None to fall through."""
  o, src, base = ins.op, ins.src, ins.op[0]
  if base in ("ret", "exit"): return -1
  if base == "bra": return ctx.kern.labels[src[-1][1]]
  if base in IGNORE: return None
  if base == "mov":
    bty = _bits(_ty(o[-1]))
    if bty[0] == "p":
      ctx.wr(src[0], ctx.rd(src[1], bty, m), bty, m)
      return None
    w = bty[1]
    if src[0][0] == "vec":  # unpack a wide register into parts
      v, ew = ctx.rd(src[1], ("b", w), m), w // len(src[0][1])
      for i, e in enumerate(src[0][1]): ctx.wr(e, (v >> np.array(i * ew, dtype=v.dtype)).astype(UI[ew]), ("b", ew), m)
    elif src[1][0] == "vec":  # pack parts into a wide register
      ew = w // len(src[1][1])
      parts = [ctx.rd(e, ("b", ew), m).astype(UI[w]) << np.array(i * ew, dtype=UI[w]) for i, e in enumerate(src[1][1])]
      ctx.wr(src[0], functools.reduce(np.bitwise_or, parts), ("b", w), m)
    else: ctx.wr(src[0], ctx.rd(src[1], ("b", w), m), ("b", w), m)
    return None
  if base == "cvta":
    ctx.wr(src[0], ctx.rd(src[1], ("u", 64), m), ("u", 64), m)
    return None
  if base in ("ld", "st"):
    ty = _ty(o[-1])
    if ty[1] > 64: raise RuntimeError(f"unsupported access width in: {ins}")
    n, mi = ty[1] // 8, (1 if base == "ld" else 0)
    vec = next((int(x[1:]) for x in o if re.fullmatch(r"v\d", x)), 1)
    addr = ctx.addr(src[mi], m)
    dsts = src[1 - mi][1] if src[1 - mi][0] == "vec" else [src[1 - mi]]
    assert len(dsts) == vec, f"vector width mismatch in {ins}"
    for i, d in enumerate(dsts):
      ea = addr + np.uint64(i * n)
      if base == "ld": ctx.wr(d, _gather(ea, n).view(_npty(ty[0], ty[1])), ty, m)
      else: _scatter(ea, n, ctx.rd(d, ("b", ty[1]), m))
    return None
  if base in ("setp", "set"):
    ty = _ty(o[-1])
    res = _compare(o[1], ctx.rd(src[1], ty, m), ctx.rd(src[2], ty, m), ty[0])
    if o[2] in BOOL: res = BOOL[o[2]](res, ctx.rd(src[3], ("p", 1), m))
    if base == "setp":
      ctx.wr(src[0], res, ("p", 1), m)
      return None
    # set writes a value rather than a predicate: 1.0 for a float destination, all ones for an integer one
    dt = _ty(o[-2])
    if _isfloat(dt[0]): ctx.wr(src[0], np.where(res, 1.0, 0.0), dt, m)
    else: ctx.wr(src[0], np.where(res, np.array((1 << dt[1]) - 1, dtype=UI[dt[1]]), np.array(0, dtype=UI[dt[1]])), ("b", dt[1]), m)
    return None
  if base in ("selp", "slct"):
    ty = _bits(_ty(o[-2] if base == "slct" else o[-1]))
    c = ctx.rd(src[3], ("p", 1), m) if base == "selp" else ctx.rd(src[3], _ty(o[-1]), m) >= 0
    ctx.wr(src[0], np.where(c, ctx.rd(src[1], ty, m), ctx.rd(src[2], ty, m)), ty, m)
    return None
  if base == "cvt":
    dt, st = _ty(o[-2]), _ty(o[-1])
    v = ctx.rd(src[1], st, m)
    if _isfloat(st[0]) and dt[0] in "su":
      lo, hi = (-(1 << (dt[1] - 1)), (1 << (dt[1] - 1)) - 1) if dt[0] == "s" else (0, (1 << dt[1]) - 1)
      r = RND[next((x for x in o if x in RND), "rzi")](v.astype(np.float64))
      v = np.where(np.isnan(r), 0, np.clip(r, float(lo), float(hi)))  # ptx saturates out of range conversions
    elif _isfloat(dt[0]):
      # cvt.rmi/rpi/rzi/rni between float types rounds to an integral value, it does not convert to an integer
      if _isfloat(st[0]) and (rnd := next((x for x in o if x in RND), None)) is not None: v = RND[rnd](v)
      v = v.astype(_npty(dt[0], dt[1]))
      if "sat" in o: v = np.where(np.isnan(v), v, np.clip(v, 0.0, 1.0)).astype(_npty(dt[0], dt[1]))
    ctx.wr(src[0], v, dt, m)
    return None
  if base in ("and", "or", "xor", "not", "cnot", "shl", "shr"):
    ty = _ty(o[-1])
    if ty[0] == "p":
      a = ctx.rd(src[1], ty, m)
      ctx.wr(src[0], ~a if base == "not" else BOOL[base](a, ctx.rd(src[2], ty, m)), ty, m)
      return None
    w = ty[1]
    a = ctx.rd(src[1], ("b", w), m)
    if base == "not": v = ~a
    elif base == "cnot": v = (a == 0).astype(UI[w])
    elif base in ("shl", "shr"):
      sa = ctx.rd(src[1], ty, m) if ty[0] == "s" else a
      v = _shift(sa, ctx.rd(src[2], ("u", 32), m), w, base == "shr", ty[0] == "s")
    else: v = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[base](a, ctx.rd(src[2], ("b", w), m))
    ctx.wr(src[0], v, ("b", w) if ty[0] != "s" else ty, m)
    return None
  if base in ("add", "sub", "mul", "mad", "fma", "div", "rem", "min", "max"):
    ty = _ty(o[-1])
    kind, w = ty
    wide = "wide" in o
    dty = ((kind, w * 2) if wide else ty)
    a, b = ctx.rd(src[1], ty, m), ctx.rd(src[2], ty, m)
    hi64 = "hi" in o and kind != "f" and w == 64
    if (wide or ("hi" in o and kind != "f")) and not hi64: a, b = (x.astype(_npty(kind, w * 2)) for x in (a, b))
    if base == "add": v = a + b
    elif base == "sub": v = a - b
    elif base == "mul": v = a * b
    elif base == "div": v = a / b if kind == "f" else _idiv(a, b, kind == "s")
    elif base == "rem": v = a - b * _idiv(a, b, kind == "s")
    elif base == "min": v = np.fmin(a, b)  # ptx min/max return the non-NaN operand
    elif base == "max": v = np.fmax(a, b)
    else:
      c = ctx.rd(src[3], dty, m)
      # fma.rn rounds once, so compute the f32 product and sum in f64 where both are exact
      v = (a.astype(np.float64) * b + c).astype(FL[w]) if kind == "f" and w < 64 else a * b + c
    if hi64: v = _mulhi64(a, b, kind == "s")
    elif "hi" in o and kind != "f": v = (v >> np.array(w, dtype=v.dtype)).astype(_npty(kind, w))
    ctx.wr(src[0], v, dty, m)
    return None
  if base in FLOAT1:
    ty = _ty(o[-1])
    ctx.wr(src[0], FLOAT1[base](ctx.rd(src[1], ty, m)), ty, m)
    return None
  if base in ("popc", "clz", "brev", "bfe", "prmt"):
    ty = _ty(o[-1])
    w = ty[1]
    a = ctx.rd(src[1], ("b", w), m)
    if base == "popc": ctx.wr(src[0], np.bitwise_count(a), ("u", 32), m)
    elif base == "clz": ctx.wr(src[0], w - _bitlen(a, w), ("u", 32), m)
    elif base == "brev":
      v = np.zeros_like(a)
      for i in range(w): v |= ((a >> np.array(i, dtype=a.dtype)) & np.array(1, dtype=a.dtype)) << np.array(w - 1 - i, dtype=a.dtype)
      ctx.wr(src[0], v, ("b", w), m)
    elif base == "bfe":
      pos, ln = (ctx.rd(src[i], ("u", 32), m).astype(UI[w]) for i in (2, 3))
      sh = np.minimum(pos, np.array(w, dtype=UI[w]))
      ln = np.minimum(ln, np.array(w, dtype=UI[w]) - sh)
      v = np.where(ln == 0, np.zeros_like(a), (a >> sh) & ((np.array(1, dtype=UI[w]) << ln) - np.array(1, dtype=UI[w])))
      if ty[0] == "s":  # a signed extract sign extends from the field's top bit
        sign = np.where(ln == 0, np.zeros_like(v), (v >> np.maximum(ln - np.array(1, dtype=UI[w]), 0)) & np.array(1, dtype=UI[w]))
        v = np.where(sign.astype(bool), v | ~((np.array(1, dtype=UI[w]) << ln) - np.array(1, dtype=UI[w])), v)
      ctx.wr(src[0], v, ("b", w), m)
    else:
      ab = ctx.rd(src[1], ("b", 32), m).astype(np.uint64) | (ctx.rd(src[2], ("b", 32), m).astype(np.uint64) << np.uint64(32))
      c = ctx.rd(src[3], ("b", 32), m).astype(np.uint64)
      v = np.zeros(ctx.count(m), dtype=np.uint32)
      for i in range(4):
        sel = (c >> np.uint64(4 * i)) & np.uint64(7)
        v |= (((ab >> (sel * np.uint64(8))) & np.uint64(0xff)) << np.uint64(8 * i)).astype(np.uint32)
      ctx.wr(src[0], v, ("b", 32), m)
    return None
  if base == "shf":
    # funnel shift: shift the 64 bit value {hi:lo} and keep one 32 bit half
    n = ctx.rd(src[3], ("u", 32), m).astype(np.uint64)
    n = np.minimum(n, np.uint64(32)) if "clamp" in o else n & np.uint64(31)
    v = ctx.rd(src[1], ("b", 32), m).astype(np.uint64) | (ctx.rd(src[2], ("b", 32), m).astype(np.uint64) << np.uint64(32))
    ctx.wr(src[0], ((v << n) >> np.uint64(32)) if o[1] == "l" else (v >> n), ("b", 32), m)
    return None
  if base == "bfi":
    ty = _ty(o[-1])
    w = ty[1]
    one, full = np.array(1, dtype=UI[w]), np.array((1 << w) - 1, dtype=UI[w])
    a, b = ctx.rd(src[1], ("b", w), m), ctx.rd(src[2], ("b", w), m)
    pos = np.minimum(ctx.rd(src[3], ("u", 32), m).astype(UI[w]), np.array(w, dtype=UI[w]))
    ln = np.minimum(ctx.rd(src[4], ("u", 32), m).astype(UI[w]), np.array(w, dtype=UI[w]) - pos)
    low = np.where(ln >= np.array(w, dtype=UI[w]), full, (one << np.minimum(ln, np.array(w - 1, dtype=UI[w]))) - one)
    mask = low << np.minimum(pos, np.array(w - 1, dtype=UI[w]))
    ctx.wr(src[0], (b & ~mask) | ((a << np.minimum(pos, np.array(w - 1, dtype=UI[w]))) & mask), ("b", w), m)
    return None
  if base in ("atom", "red"):
    ty = _ty(o[-1])
    di = 1 if base == "atom" else 0
    n, addr = ty[1] // 8, ctx.addr(src[di], m)
    b = ctx.rd(src[di + 1], ty, m)
    opn = next(x for x in o if x in ATOMIC)
    old = np.empty(addr.size, dtype=_npty(ty[0], ty[1]))
    for i in range(addr.size):  # lanes can collide, so atomics are done one lane at a time
      one = addr[i:i + 1]
      old[i] = cur = _gather(one, n).view(_npty(ty[0], ty[1]))[0]
      _scatter(one, n, np.asarray(ATOMIC[opn](cur, b[i]), dtype=_npty(ty[0], ty[1])).reshape(1).view(UI[ty[1]]))
    if base == "atom": ctx.wr(src[0], old, ty, m)
    return None
  if base == "mma":
    _mma(ctx, ins, m)
    return None
  raise RuntimeError(f"unsupported ptx instruction: {ins}")

# ***** launch *****

class Launch:
  """Owns the memory a kernel launch needs: the param buffer, module scope globals and per block shared memory."""
  def __init__(self, kern:Kernel, args:list):
    self.kern = kern
    self.sym: dict[str, int|np.ndarray] = {}
    off, self.offs = 0, []
    for name, sz in kern.params:
      off = round_up(off, min(sz, 8))
      self.offs.append(off)
      off += sz
    self.pbuf = ctypes.create_string_buffer(max(off, 1))
    for (name, sz), o, val in zip(kern.params, self.offs, args):
      ctypes.memmove(ctypes.addressof(self.pbuf) + o, ((val or 0) & ((1 << (sz * 8)) - 1)).to_bytes(sz, "little"), sz)
    for i, (name, _) in enumerate(kern.params): self.sym[name] = ctypes.addressof(self.pbuf) + self.offs[i]
    for space, name, sz, data in kern.decls:
      if space in ("shared", "local"): continue  # these are per block / per thread, allocated per chunk below
      if name not in kern.statics: kern.statics[name] = ctypes.create_string_buffer(data.ljust(max(sz, len(data), 1), b"\x00"))
      self.sym[name] = ctypes.addressof(kern.statics[name])

  def scoped(self, nblocks:int, blocks:np.ndarray, smem:int) -> dict:
    """Per chunk memory: .shared gets one copy per block, .local one per lane (nvrtc spills sin/cos to a .local depot)."""
    sym = dict(self.sym)
    self.bufs = []  # kept alive for the duration of the chunk
    for space, name, sz, _ in self.kern.decls:
      if space not in ("shared", "local"): continue
      n, stride = (nblocks, max(sz or smem, 1)) if space == "shared" else (blocks.size, round_up(max(sz, 1), 16))
      self.bufs.append(buf := ctypes.create_string_buffer(stride * n))
      idx = blocks if space == "shared" else np.arange(blocks.size, dtype=np.uint64)
      sym[name] = np.uint64(ctypes.addressof(buf)) + idx * np.uint64(stride)
    return sym

def round_up(x:int, a:int) -> int: return (x + a - 1) // a * a

def _sregs(t:np.ndarray, block:tuple[int,int,int], grid:tuple[int,int,int], smem:int) -> dict:
  bs = np.uint64(block[0] * block[1] * block[2])
  b, l = t // bs, t % bs
  sr:dict = {"warpsize": 32, "nwarpid": 64, "gridid": 0, "smid": 0, "nsmid": 1, "clock": 0, "clock64": 0, "envreg0": 0, "dynamic_smem_size": smem,
             "laneid": l % np.uint64(32), "warpid": l // np.uint64(32)}
  for i, d in enumerate("xyz"):
    div = np.uint64(math.prod(block[:i]) or 1)
    sr[f"tid.{d}"] = (l // div) % np.uint64(block[i])
    sr[f"ctaid.{d}"] = (b // np.uint64(math.prod(grid[:i]) or 1)) % np.uint64(grid[i])
    sr[f"ntid.{d}"], sr[f"nctaid.{d}"] = block[i], grid[i]
  lane = (sr["laneid"]).astype(np.uint64)
  one = np.uint64(1)
  sr["lanemask_eq"] = one << lane
  sr["lanemask_lt"] = (one << lane) - one
  sr["lanemask_le"] = (one << (lane + one)) - one
  sr["lanemask_ge"] = ~sr["lanemask_lt"] & np.uint64(0xffffffff)
  sr["lanemask_gt"] = ~sr["lanemask_le"] & np.uint64(0xffffffff)
  return sr

CHUNK = 1 << 20  # lanes executed at once, keeps a kernel's register arrays to a sane size

def ptx_run(source:ctypes.c_char_p, n_args:int, args, blck_x:int, blck_y:int, blck_z:int, grid_x:int, grid_y:int, grid_z:int, shared_mem_size:int):
  kern = parse_ptx(ctypes.string_at(source).decode())
  block, grid = (blck_x, blck_y, blck_z), (grid_x, grid_y, grid_z)
  lc = Launch(kern, [args[i] for i in range(n_args)])
  bs, nblocks = block[0] * block[1] * block[2], grid[0] * grid[1] * grid[2]
  per_chunk = max(1, CHUNK // max(bs, 1))
  with np.errstate(all="ignore"):
    for start in range(0, nblocks, per_chunk):
      nb = min(per_chunk, nblocks - start)
      if (t := np.arange(start * bs, (start + nb) * bs, dtype=np.uint64)).size == 0: continue
      ctx = Ctx(kern, lc.scoped(nb, t // np.uint64(bs) - np.uint64(start), shared_mem_size), _sregs(t, block, grid, shared_mem_size), t.size)
      _execute(kern, ctx)

def _execute(kern:Kernel, ctx:Ctx):
  # fast path: until a branch splits the lanes they all share one pc, so no per lane pc bookkeeping is needed
  upc = 0
  while upc < len(kern.insts):
    ins = kern.insts[upc]
    m = None
    if ins.pred is not None:
      pv = ctx.arr(ins.pred).astype(bool)
      if ins.neg: pv = ~pv
      if not pv.any():
        upc += 1
        continue
      if not pv.all():
        if ins.op[0] in ("bra", "ret", "exit"): break  # the lanes take different paths from here
        m = pv
    if (tgt := _step(ctx, ins, m)) is None: upc += 1
    elif tgt == -1: return
    else: upc = tgt
  else: return

  # general path: step the lowest pc first, which reconverges the lanes and makes bar.sync implicit
  pc = np.full(ctx.n, upc, dtype=np.int32)
  alive = np.ones(ctx.n, dtype=bool)
  while True:
    live = pc[alive]
    if live.size == 0: return
    cur = int(live.min())
    m = alive & (pc == cur)
    ins = kern.insts[cur]
    me = m
    if ins.pred is not None:
      pv = ctx.arr(ins.pred).astype(bool)
      me = m & (~pv if ins.neg else pv)
    pc[m] = cur + 1
    if not me.any(): continue
    if (tgt := _step(ctx, ins, None if me.all() else me)) is None: continue
    if tgt == -1: alive[me] = False
    else: pc[me] = tgt


# ***** tensor cores *****
# https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions
# a warp holds each matrix spread over its 32 lanes; groupID/threadID_in_group below are the doc's names for the lane split

def _unpack(raw:np.ndarray, dt:str) -> np.ndarray:
  if dt == "f16": return raw.view(np.uint16).reshape(-1, 2).view(np.float16).astype(np.float32)
  if dt == "bf16": return (raw.view(np.uint16).reshape(-1, 2).astype(np.uint32) << np.uint32(16)).view(np.float32)
  return raw.view(np.float32).reshape(-1, 1)  # tf32 and f32 are stored as plain f32 bits

def _pack(vals:np.ndarray, dt:str) -> np.ndarray:
  if dt == "f16": return np.ascontiguousarray(vals.astype(np.float16)).view(np.uint16).reshape(-1, 2).view(np.uint32).reshape(-1)
  return np.ascontiguousarray(vals.astype(np.float32)).view(np.uint32).reshape(-1)

def _lanes(kind:str, i:int, per:int, gid:np.ndarray, tig:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  reg, j = i // per, i % per
  if kind == "a": return (gid + 8 * (reg % 2), tig * per + j + (8 if per == 2 else 4) * (reg // 2))
  if kind == "b": return (tig * per + j + (8 if per == 2 else 4) * reg, gid)
  return (gid + 8 * (i // 2), tig * 2 + i % 2)  # the accumulator is always 4 values over a 16x8 tile

def _veclist(o) -> list: return o[1] if o[0] == "vec" else [o]  # a one element fragment can be written without braces

def _mma(ctx:Ctx, ins:Inst, m) -> None:
  o, src = ins.op, ins.src
  if m is not None: raise RuntimeError("mma needs a converged warp")
  M, N, K = (int(x) for x in re.findall(r"\d+", next(p for p in o if re.fullmatch(r"m\d+n\d+k\d+", p))))
  d_ty, a_ty, b_ty, c_ty = o[-4], o[-3], o[-2], o[-1]
  n = ctx.n
  assert n % 32 == 0, f"mma needs whole warps, got {n} lanes"
  nw = n // 32
  lane = np.arange(32)
  gid, tig = lane >> 2, lane & 3

  def load(ops, dt, kind, count):
    per = 2 if dt in ("f16", "bf16") else 1
    vals = np.concatenate([_unpack(ctx.rd(r, ("b", 32), None), dt) for r in ops], axis=1)
    mat = np.zeros((nw, M if kind == "a" else K if kind == "b" else M, K if kind == "a" else N), dtype=np.float32)
    for i in range(count):
      r, c = _lanes(kind, i, per, gid, tig)
      mat[:, r, c] = vals[:, i].reshape(nw, 32)
    return mat

  a = load(_veclist(src[1]), a_ty, "a", M * K // 32)
  b = load(_veclist(src[2]), b_ty, "b", K * N // 32)
  c = load(_veclist(src[3]), c_ty, "c", M * N // 32)
  d = np.matmul(a, b) + c

  per = 2 if d_ty == "f16" else 1
  out = np.empty((nw, 32, M * N // 32), dtype=np.float32)
  for i in range(M * N // 32):
    r, cc = _lanes("c", i, per, gid, tig)
    out[:, :, i] = d[:, r, cc]
  flat = out.reshape(n, -1)
  for ri, reg in enumerate(_veclist(src[0])): ctx.wr(reg, _pack(flat[:, ri * per:(ri + 1) * per], d_ty), ("b", 32), None)
