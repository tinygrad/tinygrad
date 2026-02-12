# Instruction selection for AMD GPUs via pcode-derived PatternMatcher
# Parses AMD pcode specs into UOp templates, normalizes them, and converts to UPat patterns
# that rewrite renderer-level UOps into Ops.INS with arg=Inst objects

import functools
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.emu import parse_pcode
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3 import ins as rdna3_ins

# sentinel UOps representing source operands (typed as u32, like real registers)
_SENTINEL = {f'S{i}': UOp(Ops.DEFINE_VAR, dtypes.uint32, arg=(f'S{i}', 0, 0xFFFFFFFF)) for i in range(4)}
_SENTINEL_SET = set(_SENTINEL.values())

# only parse ALU-relevant opcode types (memory ops need different sentinels)
_ALU_ENUM_TYPES = frozenset({'SOP1Op', 'SOP2Op', 'SOPCOp', 'SOPKOp', 'VOP1Op', 'VOP2Op',
                             'VOP3Op', 'VOP3POp', 'VOP3SDOp', 'VOPCOp', 'VINTERPOp'})
# SOP types use SGPRs — skip for LLVM inline asm renderer (VGPR-only)
_SOP_ENUM_TYPES = frozenset({'SOP1Op', 'SOP2Op', 'SOPCOp', 'SOPKOp'})

# opcode enum class name -> list of Inst class suffixes to try
_VARIANT_SUFFIXES = ['', '_SDST']

def make_inst(opcode):
  """Create an Inst object with just the opcode set (registers defaulted)."""
  base_name = type(opcode).__name__[:-2]
  for suffix in _VARIANT_SUFFIXES:
    cls = getattr(rdna3_ins, base_name + suffix, None)
    if cls is None: continue
    try: return cls(op=opcode)
    except (RuntimeError, TypeError): continue
  raise RuntimeError(f"no Inst class found for {opcode}")

# ═══════════════════════════════════════════════════════════════
# Normalization: strip register-model artifacts from pcode UOps
# ═══════════════════════════════════════════════════════════════

def normalize(uop, _cache=None):
  """Strip register-model artifacts from pcode UOps to match renderer-level UOps.

  Pcode models registers as typeless u32 words and uses BITCAST/CAST to reinterpret.
  The renderer's UOps are natively typed — we strip these artifacts:
    - BITCAST(f32, sentinel_u32) -> sentinel typed as f32
    - CAST(i32, sentinel_u32) -> sentinel typed as i32 (same-size reinterpret)
    - CAST(u64, sentinel_u32) -> sentinel typed as u64 (widening for 64-bit ops)
    - BITCAST(T, x) where x.dtype == T -> x (identity bitcast)
    - AND(x, mask) where mask is shift masking -> x (hardware does this implicitly)
  """
  if _cache is None: _cache = {}
  if id(uop) in _cache: return _cache[id(uop)]

  # first recurse so children are normalized before we check patterns
  new_src = tuple(normalize(s, _cache) for s in uop.src)
  uop = uop if new_src == uop.src else uop.replace(src=new_src)

  # BITCAST or CAST on a sentinel -> sentinel with target dtype
  if uop.op in (Ops.BITCAST, Ops.CAST) and len(uop.src) == 1 and uop.src[0] in _SENTINEL_SET:
    result = uop.src[0].replace(dtype=uop.dtype)
    _cache[id(uop)] = result
    return result

  # identity BITCAST: BITCAST(T, x) where x already has dtype T
  if uop.op == Ops.BITCAST and len(uop.src) == 1 and uop.src[0].dtype == uop.dtype:
    _cache[id(uop)] = uop.src[0]
    return uop.src[0]

  # shift masking: AND(sentinel, 31) or AND(sentinel, 63) -> sentinel (hardware masks shift amounts)
  if uop.op == Ops.AND and len(uop.src) == 2:
    if uop.src[1].op == Ops.CONST and uop.src[1].arg in (31, 63) and uop.src[0].op == Ops.DEFINE_VAR:
      _cache[id(uop)] = uop.src[0]
      return uop.src[0]

  _cache[id(uop)] = uop
  return uop

def _count_nodes(uop, _seen=None):
  if _seen is None: _seen = set()
  if id(uop) in _seen: return 0
  _seen.add(id(uop))
  return 1 + sum(_count_nodes(s, _seen) for s in uop.src)

# ═══════════════════════════════════════════════════════════════
# UOp template -> UPat conversion
# ═══════════════════════════════════════════════════════════════

def uop_to_upat(uop, _seen=None):
  """Convert a normalized UOp template into a matchable UPat pattern."""
  if _seen is None: _seen = {}
  if id(uop) in _seen: return _seen[id(uop)]
  if uop.op == Ops.DEFINE_VAR and isinstance(uop.arg, tuple) and uop.arg[0] in _SENTINEL:
    result = UPat.var(uop.arg[0], dtype=uop.dtype)
    _seen[id(uop)] = result
    return result
  if uop.op in (Ops.CONST, Ops.VCONST):
    result = UPat(uop.op, uop.dtype, arg=uop.arg)
    _seen[id(uop)] = result
    return result
  src = tuple(uop_to_upat(s, _seen) for s in uop.src) if uop.src else None
  result = UPat(uop.op, uop.dtype, src=src)
  _seen[id(uop)] = result
  return result

# ═══════════════════════════════════════════════════════════════
# Pattern classification and selection
# ═══════════════════════════════════════════════════════════════

def _select_best_opcode(opcodes):
  """Prefer shorter encodings: VOP1/VOP2 > SOP > VOP3/VOPC."""
  _PREF = {'VOP1Op': 0, 'VOP2Op': 0, 'SOP1Op': 1, 'SOP2Op': 1, 'SOPCOp': 1, 'VOPCOp': 2, 'VOP3Op': 3, 'VOP3SDOp': 3, 'VOP3POp': 4}
  return min(opcodes, key=lambda oc: (_PREF.get(type(oc).__name__, 9), oc.value))

def _is_direct_alu(norm_uop):
  """Check if normalized UOp is a direct ALU: op(sentinels...) with no intermediate ops."""
  if norm_uop.op not in GroupOp.ALU and norm_uop.op not in {Ops.CAST, Ops.BITCAST}: return False
  return all(s.op == Ops.DEFINE_VAR for s in norm_uop.src)

def _pattern_key(uop, _seen=None):
  """Structural fingerprint for a normalized UOp template (sentinels become var placeholders)."""
  if _seen is None: _seen = {}
  if id(uop) in _seen: return _seen[id(uop)]
  if uop.op == Ops.DEFINE_VAR and isinstance(uop.arg, tuple) and uop.arg[0] in _SENTINEL:
    result = f'var({uop.arg[0]},{uop.dtype})'
  elif uop.op in (Ops.CONST, Ops.VCONST):
    result = f'const({uop.arg},{uop.dtype})'
  else:
    children = ','.join(_pattern_key(s, _seen) for s in uop.src)
    result = f'{uop.op}({uop.dtype},{children})'
  _seen[id(uop)] = result
  return result

def _runtime_key(uop, _var_counter=None, _seen=None):
  """Compute a structural key from a matched UOp at runtime (real data, not sentinels).
  Leaf UOps (non-ALU with no recognized children) are treated as variables."""
  if _seen is None: _seen = {}
  if _var_counter is None: _var_counter = [0]
  uid = id(uop)
  if uid in _seen: return _seen[uid]
  _ALU_OPS = GroupOp.ALU | {Ops.CAST, Ops.BITCAST, Ops.WHERE}
  if uop.op in (Ops.CONST, Ops.VCONST):
    result = f'const({uop.arg},{uop.dtype})'
  elif uop.op not in _ALU_OPS:
    result = f'var(S{_var_counter[0]},{uop.dtype})'
    _var_counter[0] += 1
  else:
    children = ','.join(_runtime_key(s, _var_counter, _seen) for s in uop.src)
    result = f'{uop.op}({uop.dtype},{children})'
  _seen[uid] = result
  return result

# ═══════════════════════════════════════════════════════════════
# Build tables from pcode
# ═══════════════════════════════════════════════════════════════

def _parse_pcode_patterns(pcode_dict, vgpr_only=False):
  """Parse ALU pcode entries, normalize, and categorize into direct vs structural."""
  # key: (op, src_dtypes_tuple, dst_dtype) -> [opcodes]
  direct: dict[tuple, list] = {}
  structural: list[tuple] = []  # [(opcode, norm_uop)]
  allowed_types = _ALU_ENUM_TYPES - _SOP_ENUM_TYPES if vgpr_only else _ALU_ENUM_TYPES

  for opcode, pcode_str in pcode_dict.items():
    if type(opcode).__name__ not in allowed_types: continue
    try: env, assigns = parse_pcode(pcode_str, dict(_SENTINEL))
    except Exception: continue
    d0 = next(((n, u) for n, u in assigns if n.startswith('D0')), None)
    if d0 is None: continue
    _, uop = d0
    if _count_nodes(uop) > 5: continue
    norm = normalize(uop)
    if norm.op == Ops.DEFINE_VAR or norm.op in (Ops.CONST, Ops.VCONST): continue

    if _is_direct_alu(norm):
      src_dtypes = tuple(s.dtype for s in norm.src)
      key = (norm.op, src_dtypes, norm.dtype)
      direct.setdefault(key, []).append(opcode)
    else:
      structural.append((opcode, norm))

  # pick best opcode for each direct pattern
  direct_best = {k: _select_best_opcode(v) for k, v in direct.items()}

  # deduplicate structural patterns by shape, pick best
  seen: dict[str, list] = {}
  for opcode, norm in structural:
    key = _pattern_key(norm)
    seen.setdefault(key, []).append((opcode, norm))
  structural_best = []
  for key, group in seen.items():
    best = _select_best_opcode([oc for oc, _ in group])
    best_norm = next(n for oc, n in group if oc == best)
    structural_best.append((best, best_norm, key))

  return direct_best, structural_best

# ═══════════════════════════════════════════════════════════════
# Global tables (populated by build_isel_patterns, used by callbacks)
# ═══════════════════════════════════════════════════════════════

# direct: (op, src_dtypes, dst_dtype) -> Inst
_DIRECT_TABLE: dict[tuple, object] = {}
# structural: pattern_key_string -> Inst
_STRUCTURAL_TABLE: dict[str, object] = {}

# ops that LLVM handles natively (compares write VCC, not VGPRs)
_SKIP_OPS = frozenset({Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ})

def _isel_direct(m):
  """Callback for direct ALU: look up Inst by (op, src_dtypes, dtype)."""
  if m.op in _SKIP_OPS or m.dtype == dtypes.bool: return None
  src_dtypes = tuple(s.dtype for s in m.src)
  inst = _DIRECT_TABLE.get((m.op, src_dtypes, m.dtype))
  if inst is None: return None
  return UOp(Ops.INS, m.dtype, m.src, arg=inst)

def _isel_structural(m, **kwargs):
  """Callback for structural patterns: compute runtime key, look up Inst."""
  if m.op in _SKIP_OPS or m.dtype == dtypes.bool: return None
  key = _runtime_key(m)
  inst = _STRUCTURAL_TABLE.get(key)
  if inst is None: return None
  # collect source vars in order (leaves of the matched tree)
  srcs = _collect_leaves(m)
  return UOp(Ops.INS, m.dtype, tuple(srcs), arg=inst)

def _collect_leaves(uop, _seen=None):
  """Collect leaf UOps (non-ALU) from a matched tree in left-to-right order."""
  if _seen is None: _seen = set()
  _ALU_OPS = GroupOp.ALU | {Ops.CAST, Ops.BITCAST, Ops.WHERE}
  uid = id(uop)
  if uid in _seen: return []
  _seen.add(uid)
  if uop.op in (Ops.CONST, Ops.VCONST): return []
  if uop.op not in _ALU_OPS: return [uop]
  result = []
  for s in uop.src:
    result.extend(_collect_leaves(s, _seen))
  return result

# ═══════════════════════════════════════════════════════════════
# Build the PatternMatcher
# ═══════════════════════════════════════════════════════════════

def build_isel_patterns(pcode_dict=PCODE, vgpr_only=False) -> PatternMatcher:
  """Parse pcode and build a PatternMatcher for instruction selection."""
  direct_best, structural_best = _parse_pcode_patterns(pcode_dict, vgpr_only=vgpr_only)

  # populate direct table
  _DIRECT_TABLE.clear()
  for (op, src_dtypes, dtype), opcode in direct_best.items():
    _DIRECT_TABLE[(op, src_dtypes, dtype)] = make_inst(opcode)

  # populate structural table
  _STRUCTURAL_TABLE.clear()
  for opcode, norm, pkey in structural_best:
    _STRUCTURAL_TABLE[pkey] = make_inst(opcode)

  patterns: list[tuple] = []

  # structural patterns first (more specific, should match before catch-all direct)
  for opcode, norm, pkey in structural_best:
    pat = uop_to_upat(norm).named('m')
    patterns.append((pat, _isel_structural))

  # direct ALU: catch-all patterns that look up by (op, src_dtypes, dtype)
  patterns.append((UPat(GroupOp.ALU, name='m'), _isel_direct))
  patterns.append((UPat(Ops.CAST, name='m'), _isel_direct))
  patterns.append((UPat(Ops.BITCAST, name='m'), _isel_direct))

  return PatternMatcher(patterns)

@functools.cache
def rdna3_isel() -> PatternMatcher:
  """Build the default RDNA3 instruction selector (VOP-only for LLVM inline asm)."""
  return build_isel_patterns(PCODE, vgpr_only=True)

# ═══════════════════════════════════════════════════════════════
# LLVM inline asm rendering for Ops.INS
# ═══════════════════════════════════════════════════════════════

import re
from tinygrad.dtype import PtrDType

def _ins_mnemonic(inst) -> str:
  """Get the assembly mnemonic from an Inst opcode (strip _E32/_E64 suffix)."""
  return re.sub(r'_e(32|64)$', '', inst.op.name.lower())

def _ldt(dt):
  """LLVM type string for a DType."""
  if dt.vcount > 1: return f"<{dt.vcount} x {_ldt(dt.scalar())}>"
  if isinstance(dt, PtrDType): return _ldt(dt.base) + "*"
  return {dtypes.void: "void", dtypes.bool: "i1", dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
          dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
          dtypes.float16: "half", dtypes.bfloat16: "bfloat", dtypes.float32: "float", dtypes.float64: "double"}[dt]

def render_ins_llvm(ctx, x):
  """Render Ops.INS as LLVM inline assembly call."""
  inst = x.arg
  mnem = _ins_mnemonic(inst)
  n_srcs = len(x.src)
  # build operand string: $0 = dest, $1..$N = sources
  ops = ", ".join(f"${i}" for i in range(n_srcs + 1))
  asm_str = f"{mnem} {ops}"
  # constraints: =v for output, v for each input (VGPR)
  constraints = "=v," + ",".join("v" for _ in range(n_srcs))
  # LLVM types and values
  ret_type = _ldt(x.dtype)
  args = ", ".join(f"{_ldt(s.dtype)} {ctx[s]}" for s in x.src)
  return f"  {ctx[x]} = call {ret_type} asm \"{asm_str}\", \"{constraints}\"({args})"

# ═══════════════════════════════════════════════════════════════
# AMDISELRenderer: LLVM renderer with pcode-based instruction selection
# ═══════════════════════════════════════════════════════════════

from tinygrad.renderer.llvmir import AMDLLVMRenderer

class AMDISELRenderer(AMDLLVMRenderer):
  """AMD renderer that uses pcode-derived instruction selection for ALU ops."""
  def __init__(self, arch: str):
    super().__init__(arch)
    # add ISel as extra_matcher: rewrites ALU UOps → Ops.INS
    self.extra_matcher = self.extra_matcher + rdna3_isel()
    # add Ops.INS rendering to string_rewrite
    self.string_rewrite = PatternMatcher([(UPat(Ops.INS, name='x'), render_ins_llvm)]) + self.string_rewrite
  def __reduce__(self): return self.__class__, (self.arch,)
