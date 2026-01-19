# UOp-based pseudocode compiler for AMD GPU instruction emulation
import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType, AddrSpace

from extra.assembly.amd.pcode_parse import If, For
from extra.assembly.amd.pcode_transform import parse_transform, MEM_BUF as PCODE_MEM_BUF

FLOATS = (dtypes.float16, dtypes.float32, dtypes.float64)
MASK32, MASK64 = 0xffffffff, 0xffffffffffffffff

def _ftz32(bits: int) -> float:
  """Flush f32 denormals to zero (RDNA3 default mode)."""
  bits = bits & MASK32
  return 0.0 if (bits & 0x7f800000) == 0 and (bits & 0x007fffff) != 0 else struct.unpack('<f', struct.pack('<I', bits))[0]

def _cast(x: UOp, dtype: DType) -> UOp:
  return x if x.dtype == dtype else UOp(Ops.BITCAST if dtype.itemsize == x.dtype.itemsize else Ops.CAST, dtype, (x,))

# Input variables
INPUT_VARS = {n: UOp(Ops.DEFINE_VAR, dt, (), (n, 0, (1 << dt.itemsize*8) - 1)) for n, dt in [
  ('S0', dtypes.uint32), ('S1', dtypes.uint32), ('S2', dtypes.uint32), ('D0', dtypes.uint32),
  ('S0_64', dtypes.uint64), ('S1_64', dtypes.uint64), ('S2_64', dtypes.uint64), ('D0_64', dtypes.uint64),
  ('SCC', dtypes.uint32), ('VCC', dtypes.uint64), ('EXEC', dtypes.uint64), ('laneId', dtypes.uint32),
  ('SIMM16', dtypes.int32), ('SIMM32', dtypes.uint32), ('PC', dtypes.uint64),
  ('ADDR', dtypes.uint64), ('SDATA', dtypes.uint64), ('VDATA', dtypes.uint64), ('VDST', dtypes.uint32),
  ('RETURN_DATA', dtypes.uint64), ('DATA', dtypes.uint64), ('DATA2', dtypes.uint64),
  ('OFFSET', dtypes.uint32), ('OFFSET0', dtypes.uint32), ('OFFSET1', dtypes.uint32),
  ('OPSEL', dtypes.uint32), ('OPSEL_HI', dtypes.uint32), ('SRC0', dtypes.uint32), ('M0', dtypes.uint32),
]}
INPUT_VARS['ADDR_BASE'] = INPUT_VARS['ADDR']

MEM_BUF = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(addrspace=AddrSpace.GLOBAL), arg=0)
LDS_BUF = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(addrspace=AddrSpace.LOCAL), arg=0)

class Ctx:
  def __init__(self, mem_buf: UOp = MEM_BUF):
    self.vars, self.decls, self.outputs, self.mem_stores, self.mem_buf = dict(INPUT_VARS), {}, [], [], mem_buf

# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

def _expr(node: UOp, ctx: Ctx, hint: DType = None) -> UOp:
  """Transform parsed UOp expression to resolved UOp."""
  match node:
    case UOp(Ops.CONST, dtypes.void, (type_src,), val):  # Deferred const: infer type from type_src
      return UOp.const(_expr(type_src, ctx, hint).dtype, val)
    case UOp(Ops.CONST, dt, _, val): return UOp.const(dt, val)

    case UOp(Ops.DEFINE_VAR, dt, _, (name, _, _)):
      if name not in ctx.vars: raise ValueError(f"Unknown variable: {name}")
      return _cast(ctx.vars[name], hint or dt if dt != dtypes.void else ctx.vars[name].dtype)

    case UOp(Ops.BITCAST, dt, (inner,)):
      # Typed variable access: S0.f32 -> look up S0 and cast/bitcast to f32
      if inner.op == Ops.DEFINE_VAR and inner.dtype == dtypes.void:
        name = inner.arg[0]
        vn = name + '_64' if dt.itemsize == 8 and name.isupper() else name
        base = ctx.vars.get(vn) if vn in ctx.vars else ctx.vars.get(name)
        if base is None: raise ValueError(f"Unknown variable: {name}")
        if dt.itemsize == 3 and 'int' in dt.name:  # 24-bit int
          masked = UOp(Ops.AND, dtypes.uint32, (base, UOp.const(dtypes.uint32, 0xffffff)))
          return masked if 'uint' in dt.name else UOp(Ops.SUB, dtypes.int32, (UOp(Ops.XOR, dtypes.int32, (masked, UOp.const(dtypes.int32, 0x800000))), UOp.const(dtypes.int32, 0x800000)))
        if dt == dtypes.float16: return UOp(Ops.BITCAST, dtypes.float16, (UOp(Ops.AND, dtypes.uint16, (_cast(base, dtypes.uint16), UOp.const(dtypes.uint16, 0xffff))),))
        if dt in FLOATS: return UOp(Ops.BITCAST, dt, (base,))
        if dt in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64): return _cast(ctx.vars.get(name + '_64', base) if dt == dtypes.int64 else base, dt)
        return _cast(base, dt)
      # Non-variable BITCAST: ensure proper types for float16/bfloat16
      inner_resolved = _expr(inner, ctx, dt)
      if dt in (dtypes.float16, dtypes.bfloat16): return UOp(Ops.BITCAST, dt, (_cast(inner_resolved, dtypes.uint16),))
      if dt in FLOATS: return UOp(Ops.BITCAST, dt, (inner_resolved,))
      return _cast(inner_resolved, dt)

    case UOp(Ops.CUSTOMI, _, (base_expr, hi_expr, lo_expr)):  # Slice or array access
      # VGPR[lane][reg] read
      if base_expr.op == Ops.CUSTOMI and hi_expr is lo_expr:
        inner_base, inner_idx, _ = base_expr.src
        if inner_base.op == Ops.DEFINE_VAR and inner_base.arg[0] == 'VGPR':
          lane, reg = _expr(inner_idx, ctx, dtypes.uint32), _expr(hi_expr, ctx, dtypes.uint32)
          return UOp(Ops.CUSTOM, dtypes.uint32, (UOp(Ops.ADD, dtypes.uint32, (UOp(Ops.MUL, dtypes.uint32, (lane, UOp.const(dtypes.uint32, 256))), reg)),), arg='vgpr_read')
      # SGPR[idx] read
      if base_expr.op == Ops.DEFINE_VAR and base_expr.arg[0] == 'SGPR' and hi_expr is lo_expr:
        return UOp(Ops.CUSTOM, dtypes.uint32, (_expr(hi_expr, ctx, dtypes.uint32),), arg='sgpr_read')
      # Array element access
      if base_expr.op == Ops.DEFINE_VAR and isinstance(base_expr.arg, tuple) and hi_expr is lo_expr:
        name, var_dtype = base_expr.arg[0], ctx.decls.get(base_expr.arg[0])
        if var_dtype is not None and var_dtype.count > 1:
          idx_uop = _expr(hi_expr, ctx).simplify()
          if idx_uop.op == Ops.CONST:
            return ctx.vars.get(f"{name}_{int(idx_uop.arg)}", UOp.const(var_dtype.scalar(), 0))
      base, hi_uop, lo_uop = _expr(base_expr, ctx), _expr(hi_expr, ctx).simplify(), _expr(lo_expr, ctx).simplify()
      # Single bit: base[idx]
      if hi_expr is lo_expr:
        return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, base.dtype, (base, _cast(lo_uop, base.dtype))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
      # Bit slice: base[hi:lo]
      if hi_uop.op == Ops.CONST and lo_uop.op == Ops.CONST:
        hi_val, lo_val = int(hi_uop.arg), int(lo_uop.arg)
        if hi_val < lo_val:  # Reversed slice - bit reverse
          width = lo_val - hi_val + 1
          result_dt = dtypes.uint64 if width == 64 else dtypes.uint32
          result = UOp.const(result_dt, 0)
          for i in range(width):
            bit = UOp(Ops.AND, result_dt, (UOp(Ops.SHR, result_dt, (_cast(base, result_dt), UOp.const(result_dt, i))), UOp.const(result_dt, 1)))
            result = UOp(Ops.OR, result_dt, (result, UOp(Ops.SHL, result_dt, (bit, UOp.const(result_dt, width - 1 - i)))))
          return result
        shifted = UOp(Ops.SHR, base.dtype, (base, UOp.const(base.dtype, lo_val))) if lo_val else base
        return UOp(Ops.AND, dtypes.uint32, (_cast(shifted, dtypes.uint32), UOp.const(dtypes.uint32, (1 << (hi_val - lo_val + 1)) - 1)))
      raise ValueError(f"Non-constant slice bounds: {node}")

    case UOp(Ops.WHERE, dt, (cond, tv, fv)):
      return UOp(Ops.WHERE, dt, (_expr(cond, ctx), _expr(tv, ctx, dt), _expr(fv, ctx, dt)))

    # Memory operations: INDEX from pcode_transform -> LOAD with actual buffer
    case UOp(Ops.INDEX, dt, (buf, addr)):
      actual_buf = ctx.mem_buf if buf is PCODE_MEM_BUF else buf
      idx = UOp(Ops.INDEX, dt.ptr(0, ctx.mem_buf.dtype.addrspace), (actual_buf, _expr(addr, ctx, dtypes.uint64)))
      return UOp(Ops.LOAD, dt, (idx,))

    # Generic recursion: just transform children, preserve op and dtype
    case UOp(op, dt, srcs, arg):
      return UOp(op, dt, tuple(_expr(s, ctx, hint) for s in srcs), arg)
  raise ValueError(f"Cannot transform expression: {node}")

# ═══════════════════════════════════════════════════════════════════════════════
# STATEMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

OUT_VARS = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')

def _get_var_name(u: UOp) -> str|None:
  if u.op == Ops.DEFINE_VAR and isinstance(u.arg, tuple): return u.arg[0]
  if u.op == Ops.BITCAST and u.src[0].op == Ops.DEFINE_VAR: return u.src[0].arg[0]
  return None

def _get_lhs_info(lhs: UOp, ctx: Ctx) -> tuple[str, DType, int|None, int|None, str|None, int|str|None, UOp|tuple|None]:
  """Extract: (var_name, dtype, hi_bit, lo_bit, idx_var, array_idx, dynamic_idx)"""
  # Simple variable: D or D.type
  if lhs.op == Ops.DEFINE_VAR: return lhs.arg[0], ctx.vars.get(lhs.arg[0], UOp.const(dtypes.uint32, 0)).dtype, None, None, None, None, None
  if lhs.op == Ops.BITCAST and (name := _get_var_name(lhs.src[0])): return name, lhs.dtype, None, None, None, None, None
  # Indexed access: base[idx] or base[hi:lo]
  if lhs.op == Ops.CUSTOMI or (lhs.op == Ops.BITCAST and lhs.src[0].op == Ops.CUSTOMI):
    outer_dt = lhs.dtype if lhs.op == Ops.BITCAST else None
    ci = lhs.src[0] if lhs.op == Ops.BITCAST else lhs
    base, hi_u, lo_u = ci.src
    # VGPR[lane][reg]
    if base.op == Ops.CUSTOMI and _get_var_name(base.src[0]) == 'VGPR':
      return 'VGPR', outer_dt or dtypes.uint32, None, None, None, None, (base.src[1], hi_u)
    # SGPR[reg]
    if (name := _get_var_name(base)) == 'SGPR': return 'SGPR', outer_dt or dtypes.uint32, None, None, None, None, hi_u
    if name is None: raise ValueError(f"Cannot parse LHS: {lhs}")
    dt = outer_dt or (base.dtype if base.op == Ops.BITCAST else dtypes.uint32)
    is_single = hi_u is lo_u
    # Constant indices
    if hi_u.op == Ops.CONST and lo_u.op == Ops.CONST:
      hi, lo = int(hi_u.arg), int(lo_u.arg)
      if is_single and (vdt := ctx.decls.get(name)) and vdt.count > 1: return name, vdt.scalar(), None, None, None, hi, None
      return name, dt, hi, lo, None, None, None
    # Variable index
    if is_single and hi_u.op == Ops.DEFINE_VAR:
      idx = hi_u.arg[0]
      if (vdt := ctx.decls.get(name)) and vdt.count > 1: return name, vdt.scalar(), None, None, None, idx, None
      return name, dt, None, None, idx, None, None
    # Dynamic expression index
    if is_single: return name, dt, None, None, None, None, hi_u
  raise ValueError(f"Cannot parse LHS: {lhs}")

def _stmt(stmt, ctx: Ctx):
  match stmt:
    # Declaration: DEFINE_VAR with dtype and name arg (arg is tuple: (name, min, max))
    case UOp(Ops.DEFINE_VAR, dtype, arg=(name, _, _)) if name is not None and dtype != dtypes.void:
      ctx.decls[name] = dtype
      if name == 'S' and dtype.count == 3:
        ctx.vars['S_0'], ctx.vars['S_1'], ctx.vars['S_2'] = ctx.vars['S0'], ctx.vars['S1'], ctx.vars['S2']
      else:
        ctx.vars[name] = UOp.const(dtype, 0)

    # Memory store (from pcode_transform): STORE(INDEX(buf, addr), val)
    case UOp(Ops.STORE, _, (idx, val)) if idx.op == Ops.INDEX:
      buf, addr = idx.src
      actual_buf = ctx.mem_buf if buf is PCODE_MEM_BUF else buf
      dt = idx.dtype  # element type
      idx_expr = UOp(Ops.INDEX, dt.ptr(0, ctx.mem_buf.dtype.addrspace), (actual_buf, _expr(addr, ctx, dtypes.uint64)))
      val_expr = _expr(val, ctx, dt)
      ctx.mem_stores.append(UOp(Ops.STORE, dtypes.void, (idx_expr, val_expr)))
      return

    # GROUP: execute all statements in the group
    case UOp(Ops.GROUP, _, stmts):
      for s in stmts: _stmt(s, ctx)
      return

    # Assignment: ASSIGN(lhs, rhs)
    case UOp(Ops.ASSIGN, _, (lhs, rhs)):
      var, dtype, hi, lo, idx_var, array_idx, dynamic_idx = _get_lhs_info(lhs, ctx)

      # VGPR write
      if var == 'VGPR' and isinstance(dynamic_idx, tuple):
        lane, reg = _expr(dynamic_idx[0], ctx, dtypes.uint32), _expr(dynamic_idx[1], ctx, dtypes.uint32)
        idx = UOp(Ops.ADD, dtypes.uint32, (UOp(Ops.MUL, dtypes.uint32, (lane, UOp.const(dtypes.uint32, 256))), reg))
        ctx.outputs.append(('VGPR_WRITE', UOp(Ops.CUSTOM, dtypes.uint32, (idx, _cast(_expr(rhs, ctx, dtype), dtypes.uint32)), arg='vgpr_write'), dtypes.uint32))
        return

      # SGPR write
      if var == 'SGPR' and dynamic_idx is not None and not isinstance(dynamic_idx, tuple):
        ctx.outputs.append(('SGPR_WRITE', UOp(Ops.CUSTOM, dtypes.uint32, (_expr(dynamic_idx, ctx, dtypes.uint32), _cast(_expr(rhs, ctx, dtype), dtypes.uint32)), arg='sgpr_write'), dtypes.uint32))
        return

      # Dynamic bit index: D0.u32[expr] = value
      if dynamic_idx is not None and not isinstance(dynamic_idx, tuple):
        idx_uop, rhs_uop = _expr(dynamic_idx, ctx, dtypes.uint32), _expr(rhs, ctx, dtypes.uint32)
        op_dt = dtypes.uint64 if dtype.itemsize == 8 else dtypes.uint32
        base = _cast(ctx.vars.get(var, UOp.const(op_dt, 0)), op_dt)
        bit_mask = UOp(Ops.SHL, op_dt, (UOp.const(op_dt, 1), _cast(idx_uop, op_dt)))
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp(Ops.XOR, op_dt, (bit_mask, UOp.const(op_dt, -1))))),
                                     UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (_cast(rhs_uop, op_dt), UOp.const(op_dt, 1))), _cast(idx_uop, op_dt)))))
        ctx.vars[var] = result
        if var in OUT_VARS:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
        return

      # Array element: arr[idx] = value
      if array_idx is not None:
        rhs_uop = _expr(rhs, ctx, dtype)
        if isinstance(array_idx, str):
          idx_uop = ctx.vars.get(array_idx)
          if idx_uop is None or idx_uop.op != Ops.CONST: raise ValueError(f"Non-constant array index: {array_idx}")
          array_idx = int(idx_uop.arg)
        ctx.vars[f"{var}_{array_idx}"] = rhs_uop
        return

      # Variable bit index: var[idx_var] = cond
      if idx_var is not None:
        base, idx, cond = ctx.vars.get(var), ctx.vars.get(idx_var), _expr(rhs, ctx)
        bit_mask = UOp(Ops.SHL, dtype, (UOp.const(dtype, 1), _cast(idx, dtype)))
        result = UOp(Ops.OR, dtype, (UOp(Ops.AND, dtype, (base, UOp(Ops.XOR, dtype, (bit_mask, UOp.const(dtype, -1))))),
                                     UOp(Ops.SHL, dtype, (UOp(Ops.AND, dtype, (_cast(cond, dtype), UOp.const(dtype, 1))), _cast(idx, dtype)))))
        ctx.vars[var] = result
        if var in OUT_VARS: ctx.outputs.append((var, result, dtype))
        return

      # Bit slice: var[hi:lo] = value
      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        op_dt = dtypes._uint256 if hi >= 128 else dtypes._uint128 if hi >= 64 else dtypes.uint64 if hi >= 32 else dtypes.uint32
        base = _cast(ctx.vars.get(var, UOp.const(op_dt, 0)), op_dt)
        rhs_uop = _expr(rhs, ctx, dtype)
        rhs_bits = UOp(Ops.BITCAST, dtypes.uint16 if dtype == dtypes.float16 else dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (rhs_uop,)) if dtype in FLOATS else _cast(rhs_uop, op_dt)
        if dtype == dtypes.float16: rhs_bits = _cast(rhs_bits, op_dt)
        mask, width = (1 << (hi - lo + 1)) - 1, op_dt.itemsize * 8
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp.const(op_dt, ~(mask << lo) & ((1 << width) - 1)))),
                                     UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (rhs_bits, UOp.const(op_dt, mask))), UOp.const(op_dt, lo)))))
        ctx.vars[var] = result
        if var in OUT_VARS:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
        return

      # Simple assignment
      rhs_uop = _expr(rhs, ctx, dtype)
      ctx.vars[var] = rhs_uop
      out_dtype = rhs_uop.dtype if rhs_uop.dtype != dtypes.void else dtype
      if out_dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'): ctx.vars[var + '_64'] = rhs_uop
      if var in OUT_VARS:
        ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
        ctx.outputs.append((var, rhs_uop, out_dtype))

    case UOp(Ops.GROUP, _, stmts):
      for s in stmts: _stmt(s, ctx)

    case If(branches):
      parsed = [((_expr(c, ctx) if c is not None else None), (sc := Ctx(ctx.mem_buf), sc.vars.update(ctx.vars), sc.decls.update(ctx.decls), [_stmt(s, sc) for s in b], sc)[-1]) for c, b in branches]
      assigned = {n for _, sc in parsed for n, _, _ in sc.outputs if n in OUT_VARS} | {n for _, sc in parsed for n, v in sc.vars.items() if n not in ctx.vars or ctx.vars[n] is not v if n not in OUT_VARS and n not in INPUT_VARS}
      for var in assigned:
        is_out, dtype = var in OUT_VARS, next((d for _, sc in parsed for n, _, d in sc.outputs if n == var), ctx.decls.get(var, dtypes.uint32))
        result = ctx.vars.get(var, UOp.const(dtype, 0))
        for cond_uop, sub_ctx in reversed(parsed):
          val = next((u for n, u, _ in sub_ctx.outputs if n == var), None) if is_out else sub_ctx.vars.get(var) if var in sub_ctx.vars and sub_ctx.vars[var] is not ctx.vars.get(var) else None
          if val is not None: result = val if cond_uop is None else UOp(Ops.WHERE, val.dtype, (cond_uop, val, _cast(result, val.dtype)))
        ctx.vars[var] = result
        if is_out: ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var] + [(var, result, dtype)]

    case For(var, start, end, body):
      start_v, end_v = _expr(start, ctx).simplify(), _expr(end, ctx).simplify()
      if start_v.op != Ops.CONST or end_v.op != Ops.CONST: raise ValueError(f"For loop bounds must be constant: {start_v}, {end_v}")
      for i in range(int(end_v.arg), int(start_v.arg) - 1, -1):
        ctx.vars[var] = UOp.const(ctx.decls.get(var, dtypes.uint32), i)
        for s in body: _stmt(s, ctx)

# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _float_to_bits(val: float, dtype: DType) -> int:
  if dtype == dtypes.float32: return struct.unpack('<I', struct.pack('<f', val))[0]
  if dtype == dtypes.float16:
    if math.isnan(val): return 0x7e00
    if math.isinf(val): return 0x7c00 if val > 0 else 0xfc00
    if abs(val) > 65504.0: return 0x7c00 if val > 0 else 0xfc00
    if abs(val) < 6.103515625e-05 and val != 0: return 0x0000 if val > 0 else 0x8000
    return struct.unpack('<H', struct.pack('<e', val))[0]
  if dtype == dtypes.float64: return struct.unpack('<Q', struct.pack('<d', val))[0]
  return int(val)

TWO_OVER_PI_1201 = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6

def _eval_uop(u: UOp) -> int|float|None:
  """Recursively evaluate a UOp tree to a constant value."""
  if u.op == Ops.CONST: return u.arg
  if u.op == Ops.CAST:
    v = _eval_uop(u.src[0])
    if v is None: return None
    # Float to int: truncate toward zero
    if u.dtype in (dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64, dtypes.int16, dtypes.uint16, dtypes.int8, dtypes.uint8):
      if u.src[0].dtype in FLOATS: return int(v)
    # Int to float
    if u.dtype in FLOATS and u.src[0].dtype not in FLOATS: return float(v)
    return v
  if u.op == Ops.BITCAST:
    v = _eval_uop(u.src[0])
    if v is None: return None
    if u.dtype == dtypes.float64 and u.src[0].dtype in (dtypes.uint64, dtypes.int64): return struct.unpack('<d', struct.pack('<Q', int(v) & MASK64))[0]
    if u.dtype == dtypes.float32 and u.src[0].dtype in (dtypes.uint32, dtypes.int32): return _ftz32(int(v))
    if u.dtype in (dtypes.uint64, dtypes.int64) and u.src[0].dtype == dtypes.float64: return struct.unpack('<Q', struct.pack('<d', float(v)))[0]
    if u.dtype in (dtypes.uint32, dtypes.int32) and u.src[0].dtype == dtypes.float32: return struct.unpack('<I', struct.pack('<f', float(v)))[0]
    return v
  if u.op == Ops.MULACC:
    a, b, c = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
    return math.fma(float(a), float(b), float(c)) if None not in (a, b, c) else None
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHR, Ops.SHL):
    l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
    if l is None or r is None: return None
    ops = {Ops.ADD: lambda a, b: a + b, Ops.SUB: lambda a, b: a - b, Ops.MUL: lambda a, b: a * b,
           Ops.AND: lambda a, b: int(a) & int(b), Ops.OR: lambda a, b: int(a) | int(b), Ops.XOR: lambda a, b: int(a) ^ int(b),
           Ops.SHR: lambda a, b: int(a) >> int(b), Ops.SHL: lambda a, b: int(a) << int(b)}
    return ops[u.op](l, r)
  if u.op == Ops.NEG: v = _eval_uop(u.src[0]); return -v if v is not None else None
  if u.op in (Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT, Ops.CMPLE):
    l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
    if l is None or r is None: return None
    return {Ops.CMPEQ: l == r, Ops.CMPNE: l != r, Ops.CMPLT: l < r, Ops.CMPLE: l <= r}[u.op]
  if u.op == Ops.WHERE:
    c, t, f = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
    return t if c else f if None not in (c, t, f) else None
  if u.op == Ops.CUSTOM:
    if u.arg == 'trig_preop_result':
      shift = _eval_uop(u.src[0])
      return float(((TWO_OVER_PI_1201 << int(shift)) >> (1201 - 53)) & 0x1fffffffffffff) if shift is not None else None
    if u.arg in ('s_ff1_i32_b32', 's_ff1_i32_b64'):
      v = _eval_uop(u.src[0])
      if v is None: return None
      mask = MASK64 if 'b64' in u.arg else MASK32
      v = int(v) & mask
      if v == 0: return 64 if 'b64' in u.arg else 32
      n = 0
      while (v & 1) == 0: v >>= 1; n += 1
      return n
    if u.arg == 'vgpr_read': return None  # Needs runtime substitution
  return None

_DTYPE_ACCESSOR = {dtypes.uint8: 'u8', dtypes.int8: 'i8', dtypes.uint16: 'u16', dtypes.int16: 'i16',
                   dtypes.uint32: 'u32', dtypes.int32: 'i32', dtypes.uint64: 'u64', dtypes.int64: 'i64',
                   dtypes.float32: 'u32', dtypes.float64: 'u64'}

def _compile_pseudocode(pseudocode: str, mem_buf: UOp = MEM_BUF, op_name: str | None = None) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp], list[UOp]]:
  ctx = Ctx(mem_buf=mem_buf)
  try:
    stmts = parse_transform(pseudocode, op_name)
  except AssertionError as e:
    print("issue parsing")
    print(pseudocode)
    print(e)
    raise
  for stmt in stmts: _stmt(stmt, ctx)
  return UOp(Ops.SINK, dtypes.void, tuple(u for _, u, _ in ctx.outputs) or ()), [(n, d) for n, _, d in ctx.outputs], INPUT_VARS, ctx.mem_stores

def _make_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp], mem_stores: list[UOp]):
  if mem_stores: sink = UOp(Ops.SINK, dtypes.void, sink.src + tuple(mem_stores))
  topo = sink.toposort()
  is_lds, is_mem = any(u.op == Ops.DEFINE_LOCAL for u in topo), bool(mem_stores) or any(u.op == Ops.LOAD for u in topo)

  def _extract_results(s, MEM=None):
    for u in s.src:
      if u.op == Ops.STORE:
        addr, val, dt = int(u.src[0].src[1].arg), u.src[1].arg, u.src[0].dtype.base
        if dt == dtypes.float32: val = struct.unpack('<I', struct.pack('<f', val))[0]
        elif dt == dtypes.float64: val = struct.unpack('<Q', struct.pack('<d', val))[0]
        setattr(MEM[addr], _DTYPE_ACCESSOR.get(dt, 'u32'), int(val))
    result = {}
    for i, (name, dtype) in enumerate(output_info):
      if i >= len(s.src): continue
      val = s.src[i].arg if s.src[i].op == Ops.CONST else _eval_uop(s.src[i])
      if val is None: continue
      result[name] = _float_to_bits(val, dtype) if dtype in FLOATS else int(val) & ((1 << (dtype.itemsize * 8)) - 1)
    return result

  def _do_loads(s, MEM):
    loads = {}
    for u in s.toposort():
      if u.op == Ops.LOAD:
        addr, dt = int(u.src[0].src[1].arg), u.src[0].dtype.base
        loads[u] = UOp.const(dt, getattr(MEM[addr], _DTYPE_ACCESSOR.get(dt, 'u32')))
    return s.substitute(loads).simplify() if loads else s

  if is_lds:
    def fn(MEM, addr, data0=0, data1=0, offset0=0, offset1=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['DATA']: UOp.const(dtypes.uint64, data0),
               input_vars['DATA2']: UOp.const(dtypes.uint64, data1), input_vars['OFFSET']: UOp.const(dtypes.uint32, offset0),
               input_vars['OFFSET0']: UOp.const(dtypes.uint32, offset0), input_vars['OFFSET1']: UOp.const(dtypes.uint32, offset1),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      return _extract_results(_do_loads(sink.substitute(dvars).simplify(), MEM), MEM)
    return fn
  elif is_mem:
    def fn(MEM, addr, vdata=0, vdst=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['SDATA']: UOp.const(dtypes.uint64, 0),
               input_vars['VDATA']: UOp.const(dtypes.uint64, vdata), input_vars['VDST']: UOp.const(dtypes.uint64, vdst),
               input_vars['DATA']: UOp.const(dtypes.uint64, vdata), input_vars['DATA2']: UOp.const(dtypes.uint64, 0),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      return _extract_results(_do_loads(sink.substitute(dvars).simplify(), MEM), MEM)
    return fn
  else:
    def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None, opsel=0, opsel_hi=0):
      simm16 = (literal if -32768 <= literal <= 32767 else (literal - 65536 if literal < 65536 else 0)) if literal else 0
      dvars = {
        input_vars['S0']: UOp.const(dtypes.uint32, s0 & MASK32), input_vars['S1']: UOp.const(dtypes.uint32, s1 & MASK32),
        input_vars['S2']: UOp.const(dtypes.uint32, s2 & MASK32), input_vars['D0']: UOp.const(dtypes.uint32, d0 & MASK32),
        input_vars['S0_64']: UOp.const(dtypes.uint64, s0), input_vars['S1_64']: UOp.const(dtypes.uint64, s1),
        input_vars['S2_64']: UOp.const(dtypes.uint64, s2), input_vars['D0_64']: UOp.const(dtypes.uint64, d0),
        input_vars['SCC']: UOp.const(dtypes.uint32, scc), input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
        input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask), input_vars['laneId']: UOp.const(dtypes.uint32, laneId),
        input_vars['SIMM16']: UOp.const(dtypes.int32, simm16), input_vars['SIMM32']: UOp.const(dtypes.uint32, literal or 0),
        input_vars['PC']: UOp.const(dtypes.uint64, pc or 0), input_vars['OPSEL']: UOp.const(dtypes.uint32, opsel),
        input_vars['OPSEL_HI']: UOp.const(dtypes.uint32, opsel_hi), input_vars['SRC0']: UOp.const(dtypes.uint32, src0_idx),
      }
      s1_sub = sink.substitute(dvars).simplify()
      if VGPR is not None:
        vgpr_subs = {}
        for u in s1_sub.toposort():
          if u.op == Ops.CUSTOM and u.arg == 'vgpr_read':
            idx = _eval_uop(u.src[0])
            if idx is not None:
              lane, reg = int(idx) // 256, int(idx) % 256
              vgpr_subs[u] = UOp.const(dtypes.uint32, VGPR[lane][reg] if lane < len(VGPR) and reg < len(VGPR[lane]) else 0)
        if vgpr_subs: s1_sub = s1_sub.substitute(vgpr_subs).simplify()
      return _extract_results(s1_sub)
    return fn

@functools.cache
def compile_uop(op_name: str, pseudocode: str):
  mem_buf = LDS_BUF if op_name.startswith('DS_') else MEM_BUF
  sink, output_info, input_vars, mem_stores = _compile_pseudocode(pseudocode, mem_buf, op_name)
  return _make_fn(sink, output_info, input_vars, mem_stores)
