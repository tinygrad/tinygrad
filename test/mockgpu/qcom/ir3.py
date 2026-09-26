import ctypes, hashlib, math, os, pathlib, re, shutil, struct, subprocess, tempfile
from tinygrad.helpers import DEBUG
from tinygrad.runtime.autogen import mesa, libc

OP = {n:i for i,n in enumerate("NOP END BAR PREDT PREDF PREDE COV ADDU SUBU SHL SHR ASHR XOR AND OR MULL SHRG MADSH "
  "ADDF MULF MADF MAXF MINF FMOV RCP SQRT LOG2 EXP2 SIN COS FLOOR CMPU CMPS CMPF SEL LD ST ISAM STIB BR JUMP SWZ "
  "IMOV NOT BRAO BRAA MINU MAXU MINS MAXS SHLG SHRM SHLM ANDG RSQ CEIL TRUNC RNDNE MULS24 MULU24 CLZ SIGN".split())}
F_FULL, F_HALF, F_CONST, F_IMM, F_PRED = range(5)
TY = {"f16":0, "f32":1, "u16":2, "u32":3, "s16":4, "s32":5, "u8":6}
COND = {"lt":0, "le":1, "gt":2, "ge":3, "eq":4, "ne":5}
SW = {"x":0, "y":1, "z":2, "w":3}
FLUT = {"0.0":0.0, "0.5":0.5, "1.0":1.0, "2.0":2.0, "4.0":4.0, "e":math.e, "pi":math.pi, "1/pi":1/math.pi,
        "1/log2(e)":1/math.log2(math.e), "log2(e)":math.log2(math.e), "1/log2(10)":1/math.log2(10), "log2(10)":math.log2(10)}
ALU = {"add.u":"ADDU", "sub.u":"SUBU", "add.s":"ADDU", "sub.s":"SUBU", "shl.b":"SHL", "shr.b":"SHR", "ashr.b":"ASHR", "xor.b":"XOR",
       "and.b":"AND", "or.b":"OR", "mull.u":"MULL", "mul.u":"MULL", "shrg":"SHRG", "madsh.m16":"MADSH", "add.f":"ADDF", "mul.f":"MULF",
       "mad.f32":"MADF", "mad.f16":"MADF", "max.f":"MAXF", "min.f":"MINF", "absneg.f":"FMOV", "rcp":"RCP", "sqrt":"SQRT", "log2":"LOG2",
       "exp2":"EXP2", "sin":"SIN", "cos":"COS", "floor.f":"FLOOR", "sel.b32":"SEL", "absneg.s":"IMOV", "not.b":"NOT", "min.u":"MINU",
       "max.u":"MAXU", "min.s":"MINS", "max.s":"MAXS", "shlg":"SHLG", "shrm":"SHRM", "shlm":"SHLM", "andg":"ANDG", "rsq":"RSQ", "hrsq":"RSQ",
       "hlog2":"LOG2", "hexp2":"EXP2", "ceil.f":"CEIL", "trunc.f":"TRUNC", "rndne.f":"RNDNE", "sel.b16":"SEL", "mul.s24":"MULS24",
       "mul.u24":"MULU24", "clz.b":"CLZ", "sign.f":"SIGN"}
_TEX = ("FMT6_16_16_16_16_FLOAT", "A6XX_TEX_CONST_0_FMT__MASK", "A6XX_TEX_CONST_0_FMT__SHIFT", "A6XX_TEX_CONST_1_WIDTH__MASK",
        "A6XX_TEX_CONST_1_WIDTH__SHIFT", "A6XX_TEX_CONST_1_HEIGHT__MASK", "A6XX_TEX_CONST_1_HEIGHT__SHIFT",
        "A6XX_TEX_CONST_2_PITCH__MASK", "A6XX_TEX_CONST_2_PITCH__SHIFT")

class Inst(ctypes.Structure):
  _fields_ = [("op", ctypes.c_uint32), ("dst", ctypes.c_uint32), ("extra", ctypes.c_uint32), ("src_file", ctypes.c_uint32),
              ("src_mod", ctypes.c_uint32), ("src", ctypes.c_uint32 * 4), ("ab", ctypes.c_uint32)]

def _fbits(f): return struct.unpack("I", struct.pack("f", f))[0]

def _src(tok):
  mod = 0
  while tok[:3] == "(r)" or tok.startswith(("(neg)", "(absneg)", "(abs)", "!")):
    if tok.startswith("!"): mod, tok = mod | 8, tok[1:]
    elif tok.startswith("(absneg)"): mod, tok = mod | 6, tok[8:]
    elif tok.startswith("(neg)"): mod, tok = mod | 2, tok[5:]
    elif tok.startswith("(abs)"): mod, tok = mod | 4, tok[5:]
    else: mod, tok = mod | 1, tok[3:]
  if tok.startswith("(") and tok.endswith(")") and tok[1:-1] in FLUT: return F_IMM, mod, _fbits(FLUT[tok[1:-1]])
  if tok.startswith("h(") and tok.endswith(")"):
    inner = tok[2:-1]
    if inner in FLUT: return F_IMM, mod, _fbits(FLUT[inner])
    v = int(inner, 0)
    if v < 0 or v > 0xffff: v &= 0xffffffff
    elif v & 0x8000: v |= 0xffff0000
    return F_IMM, mod, v
  if (m:=re.fullmatch(r"hr(\d+)\.([xyzw])", tok)): return F_HALF, mod, int(m.group(1))*4 + SW[m.group(2)]
  if (m:=re.fullmatch(r"r(\d+)\.([xyzw])", tok)): return F_FULL, mod, int(m.group(1))*4 + SW[m.group(2)]
  if (m:=re.fullmatch(r"h?c(\d+)\.([xyzw])", tok)): return F_CONST, mod, int(m.group(1))*4 + SW[m.group(2)]
  if (m:=re.fullmatch(r"p(\d+)\.([xyzw])", tok)): return F_PRED, mod, SW[m.group(2)]
  if re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|\d+)", tok): return F_IMM, mod, int(tok, 0) & 0xffffffff
  raise RuntimeError(f"bad ir3 operand {tok}")

def _dst(tok):
  f, _, v = _src(tok)
  return (f << 16) | v

def _commas(s):
  out, buf, d = [], "", 0
  for ch in s:
    if ch == "[": d += 1
    elif ch == "]": d -= 1
    if ch == "," and not d:
      out.append(buf.strip())
      buf = ""
    else: buf += ch
  if buf.strip(): out.append(buf.strip())
  return out

def _tys(suf):
  for n in sorted(TY, key=len, reverse=True):
    if suf.startswith(n) and suf[len(n):] in TY: return TY[n], TY[suf[len(n):]]
  raise RuntimeError(f"bad ir3 type {suf}")

def _mem(tok):
  m = re.fullmatch(r"([glp])\[([^\]]+)\]", tok)
  if not m: raise RuntimeError(f"bad ir3 mem {tok}")
  inner, off = m.group(2), 0
  if "+" in inner:
    inner, off_s = inner.split("+")
    off = int(off_s, 0)
  return {"g": 0, "l": 1, "p": 2}[m.group(1)], _src(inner), off

def _half_tok(op, tok):
  if ".u8_32" in op or not any(t in op for t in (".u8", ".s16", ".u16", ".f16")): return tok
  i = 0
  while tok.startswith(("(r)", "(neg)", "(absneg)", "(abs)", "!"), i): i = i + 1 if tok[i] == "!" else tok.find(")", i) + 1
  return tok[:i] + "h" + tok[i:] if tok.startswith("r", i) else tok

def _inst(op, dst, extra, srcs, ab=0, repeat=0):
  file = mod = 0
  vals = [0, 0, 0, 0]
  for i, (f, m, v) in enumerate(srcs[:4]):
    file |= (f & 0xff) << (8*i)
    mod |= (m & 0xff) << (8*i)
    vals[i] = v & 0xffffffff
  return Inst(op | (repeat << 24), dst, extra, file, mod, (ctypes.c_uint32 * 4)(*vals), ab)

def _encode(op, args, repeat):
  if op in ALU: return _inst(OP[ALU[op]], _dst(args[0]), 0, [_src(a) for a in args[1:]], repeat=repeat)
  if op.startswith(("mov.", "cov.")):
    st, dt = _tys(op.split(".", 1)[1])
    return _inst(OP["COV"], _dst(args[0]), 0, [_src(args[1])], (dt << 8) | st, repeat)
  if op.startswith("cmps."):
    _, kind, cond = op.split(".")
    return _inst(OP["CMP" + kind.upper()], _dst(args[0]), COND[cond], [_src(args[1]), _src(args[2])], repeat=repeat)
  if op.startswith(("ldg.", "stg.", "ldl.", "stl.", "ldp.", "stp.")):
    el = 2 if "16" in op else 1 if "8" in op else 4
    if op.startswith(("stg", "stl", "stp")):
      space, addr, off = _mem(args[0])
      return _inst(OP["ST"], 0, (off << 16) | int(args[2]), [_src(_half_tok(op, args[1])), addr], (el << 16) | space, repeat)
    space, addr, off = _mem(args[1])
    return _inst(OP["LD"], _dst(_half_tok(op, args[0])), (off << 16) | int(args[2]), [addr], (el << 16) | space, repeat)
  if op.startswith("isam"):
    m = re.fullmatch(r"\((?:f16|f32|u32|u16)\)\(([xyzw]+)\)(.+)", args[0])
    if not m: raise RuntimeError(f"bad isam {args}")
    dim = 2 if ".2d" in op else 3 if ".3d" in op else 1
    tex = int(args[3][2:]) if len(args) > 3 and args[3].startswith("t#") else 0
    return _inst(OP["ISAM"], _dst(m.group(2)), (dim << 8) | len(m.group(1)), [_src(args[1])], tex, repeat)
  if op.startswith("stib"):
    bits = op.split(".")
    dim = next((int(b[0]) for b in bits if b in ("1d", "2d", "3d")), 2)
    ncomp = next((int(b) for b in bits if b.isdigit()), 4)
    slot = int(args[2], 0) if len(args) > 2 else 0
    return _inst(OP["STIB"], 0, (dim << 8) | ncomp, [_src(args[0]), _src(args[1])], slot, repeat)
  if op in ("bar.g", "bar"): return _inst(OP["BAR"], 0, 0, [])
  if op.startswith("swz."):
    st, dt = _tys(op.split(".", 1)[1])
    return _inst(OP["SWZ"], _dst(args[0]), _dst(args[1]), [_src(args[2]), _src(args[3])], (dt << 8) | st, repeat)
  if op == "br": return _inst(OP["BR"], 0, int(args[1].replace("#", ""), 0) & 0xffffffff, [_src(args[0])])
  if op in ("brao", "braa"): return _inst(OP[op.upper()], 0, int(args[2].replace("#", ""), 0) & 0xffffffff, [_src(args[0]), _src(args[1])])
  if op == "jump": return _inst(OP["JUMP"], 0, int(args[0].replace("#", ""), 0) & 0xffffffff, [])
  if op in ("predt", "predf", "prede", "end"): return _inst(OP[op.upper()], 0, 0, [])
  raise RuntimeError(f"unhandled ir3 op {op} {args}")

def parse_line(line):
  s = line.strip()
  repeat, sat = 0, 0
  while s.startswith("("):
    end = s.find(")")
    tag, s = s[1:end], s[end+1:].strip()
    if tag.startswith("rpt"): repeat = int(tag[3:] or "0")
    elif tag == "sat": sat = 1
    elif tag.startswith("nop") or tag in ("sy", "ss", "jp", "eq", "ul"): continue
    else:
      s = f"({tag}){s}"
      break
  if not s: return None
  if s == "nop": return _inst(OP["NOP"], 0, 0, [])
  op, _, rest = s.partition(" ")
  ins = _encode(op, _commas(rest), repeat)
  if sat: ins.op |= 1 << 16
  return ins

def disasm(image):
  raw = os.memfd_create("ir3")
  fp = libc.fdopen(raw, b"w+")
  try:
    buf = ctypes.create_string_buffer(image)
    opts = mesa.struct_isa_decode_options(630, False, 0, False)
    mesa.ir3_isa_disasm(ctypes.addressof(buf), len(image), ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE)), ctypes.pointer(opts))
    libc.fflush(fp)
    os.lseek(raw, 0, os.SEEK_SET)
    return [ln.strip() for ln in os.read(raw, 1 << 20).decode().splitlines() if ln.strip()]
  finally: libc.fclose(fp)

_lib: ctypes.CDLL|None = None
def _so():
  global _lib
  if _lib is not None: return _lib
  srcp = pathlib.Path(__file__).with_name("ir3emu.c")
  defs = [f"-D{n}={getattr(mesa, n)}" for n in _TEX]
  so = pathlib.Path(tempfile.gettempdir()) / f"ir3emu_{hashlib.md5(srcp.read_bytes() + ''.join(defs).encode()).hexdigest()[:8]}.so"
  if not so.exists():
    cc = shutil.which("clang") or shutil.which("gcc")
    assert cc is not None
    tmp = so.with_name(f".{so.name}.{os.getpid()}")
    subprocess.check_call([cc, "-O2", "-fPIC", "-shared", "-ffp-contract=off", *defs, "-o", str(tmp), str(srcp), "-lm"])
    os.replace(tmp, so)
  _lib = ctypes.CDLL(str(so))
  _lib.ir3_launch.restype = ctypes.c_int
  _lib.ir3_set_maps.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
  _lib.ir3_launch.argtypes = [ctypes.POINTER(Inst), ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]
  return _lib

_cache: dict[bytes, tuple] = {}
def compile_image(image):
  if (hit:=_cache.get(image)) is not None: return hit
  insts, lines = [], disasm(image)
  if DEBUG >= 5: print("\n".join(lines[:80]))
  for line in lines:
    if (ins:=parse_line(line)) is None: continue
    insts.append(ins)
    if (ins.op & 0xff) == OP["END"]: break
  _cache[image] = ((Inst * max(1, len(insts)))(*insts), len(insts))
  return _cache[image]

def launch(image, consts, gx, gy, gz, lx, ly, lz, lid, wgid, shared, psz, tex, uav, maps=()):
  arr, n = compile_image(image)
  words = (ctypes.c_uint32 * (max(1, len(consts)//4)))()
  raw = bytes(consts)
  if raw: ctypes.memmove(words, raw, len(raw) - len(raw) % 4)
  lib, spans = _so(), (ctypes.c_uint64 * (2 * len(maps) or 1))()
  for i, (b, s) in enumerate(maps): spans[2 * i], spans[2 * i + 1] = b, s
  lib.ir3_set_maps(spans, len(maps))
  if (rc:=lib.ir3_launch(arr, n, words, len(consts)//4, gx, gy, gz, lx, ly, lz, lid, wgid, shared, psz, tex, uav)):
    raise RuntimeError(f"ir3 launch failed ({rc})")
