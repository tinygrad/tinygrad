# RDNA3 Renderer - uses assembly DSL and RDNARegAlloc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace, Invalid
from tinygrad.renderer import Renderer
from tinygrad.helpers import get_single_element
from tinygrad.renderer.rdna_regalloc import RDNARegAlloc
from tinygrad.renderer.rdna_uops import rdna_matcher
from tinygrad.renderer.cstyle import create_non_native_float_pats, cast_float_to_bf16
from tinygrad.codegen.late.devectorizer import no_vectorized_alu
from tinygrad.codegen.opt import tc
from extra.assembly.rdna3.lib import Inst
from extra.assembly.rdna3.asm import waitcnt
from extra.assembly.rdna3.autogen import (
  v, s, VGPR, SGPR, VCC_LO, EXEC_LO, NULL, OFF, SrcEnum,
  # VOP1
  v_mov_b32_e32, v_cvt_f32_i32_e32, v_cvt_i32_f32_e32, v_cvt_f32_u32_e32, v_cvt_u32_f32_e32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_rcp_f32_e32, v_rsq_f32_e32, v_sqrt_f32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_trunc_f32_e32, v_sin_f32_e32,
  # Additional conversion instructions
  v_cvt_f64_f32_e32, v_cvt_f32_f64_e32, v_cvt_f64_i32_e32, v_cvt_f64_u32_e32,
  v_cvt_i32_f64_e32, v_cvt_u32_f64_e32, v_ldexp_f64,
  # VOP2
  v_add_f32_e32, v_sub_f32_e32, v_mul_f32_e32, v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  v_add_nc_u32_e32, v_sub_nc_u32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_ashrrev_i32_e32,
  v_max_f32_e32, v_max_i32_e32, v_max_u32_e32,
  # VOP3
  v_fma_f32, v_fma_f64, v_mad_u64_u32, v_mad_i64_i32, v_lshlrev_b64, v_add3_u32,
  v_mul_lo_u32, v_mul_hi_u32, v_bfe_u32, v_bfe_i32, v_add_co_u32, v_add_co_ci_u32_e32, v_cndmask_b32_e64,
  v_add_f64, v_mul_f64, v_sub_co_u32, v_sub_co_ci_u32_e32,
  v_cmp_lt_f32_e32, v_cmp_eq_f32_e32, v_cmp_neq_f32_e32, v_cmp_gt_f32_e32,
  v_cmp_lt_i32_e32, v_cmp_eq_i32_e32, v_cmp_ne_i32_e32, v_cmp_gt_i32_e32,
  v_cmp_lt_u32_e32, v_cmp_eq_u32_e32, v_cmp_ne_u32_e32, v_cmp_gt_u32_e32,
  # SOPP/SOP
  s_endpgm, s_waitcnt, s_barrier, s_branch, s_cbranch_vccnz, s_cbranch_execz, s_sendmsg,
  s_mov_b32, s_mov_b64, s_and_saveexec_b32, s_or_b32,
  # SMEM
  s_load_b32, s_load_b64, s_load_b128,
  # FLAT/GLOBAL
  global_load_b32, global_load_b64, global_load_b128, global_load_u16, global_load_u8, global_load_i8,
  global_store_b32, global_store_b64, global_store_b128, global_store_b16, global_store_b8,
  # DS (local memory)
  ds_load_b32, ds_load_b64, ds_load_b128, ds_store_b32, ds_store_b64, ds_store_b128,
  # WMMA
  v_wmma_f32_16x16x16_f16,
)

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = True
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
  max_upcast_size = 16
  rdna_bf16_cast = PatternMatcher([(UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16)])
  extra_matcher = rdna_matcher + create_non_native_float_pats((dtypes.bfloat16,)) + rdna_bf16_cast
  tensor_cores = tc.amd_rdna3
  # Declare hardware-supported ops (enables decomposition patterns like fast_idiv for constant division)
  code_for_op = {
    # Transcendental ops
    Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None, Ops.SQRT: lambda: None, Ops.RECIPROCAL: lambda: None,
    # Bitwise ops
    Ops.AND: lambda: None, Ops.OR: lambda: None, Ops.XOR: lambda: None, Ops.SHL: lambda: None, Ops.SHR: lambda: None,
    # Arithmetic ops
    Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.NEG: lambda: None,
    Ops.IDIV: lambda: None, Ops.MOD: lambda: None, Ops.TRUNC: lambda: None,
    # Comparison ops
    Ops.CMPLT: lambda: None, Ops.CMPEQ: lambda: None, Ops.CMPNE: lambda: None, Ops.WHERE: lambda: None,
    # Max (used in various patterns)
    Ops.MAX: lambda: None,
  }

  def __init__(self, arch: str = "gfx1100"):
    self.arch = arch

  def __reduce__(self): return self.__class__, (self.arch,)

  def render(self, uops: list[UOp]) -> str:
    ra = RDNARegAlloc(uops)  # Register allocator with liveness analysis
    r: dict[UOp, VGPR | SGPR | int] = {}  # UOp -> register mapping
    code: list[Inst] = []  # Generated instructions
    bufs: list[UOp] = []
    vars_: list[UOp] = []  # Symbolic variables
    kernarg_offset: dict[UOp, int] = {}
    current_offset = 0
    lds_size = 0
    labels: dict[str, int] = {}  # Label -> instruction index
    pending_waits: set[UOp] = set()  # Track loads that need waits before use
    # Allocate dedicated SGPRs for exec mask saving to avoid conflicts with kernel arguments
    # These will be allocated on first use, which happens after all DEFINE_GLOBAL/VAR ops
    exec_save_load: list[SGPR | None] = [None]  # Wrapped in list for nonlocal mutation
    exec_save_if: list[SGPR | None] = [None]
    def get_exec_save_load() -> SGPR:
      if exec_save_load[0] is None: exec_save_load[0] = ra.alloc_sgpr(None)
      return s[exec_save_load[0]] if isinstance(exec_save_load[0], int) else exec_save_load[0]
    def get_exec_save_if() -> SGPR:
      if exec_save_if[0] is None: exec_save_if[0] = ra.alloc_sgpr(None)
      return s[exec_save_if[0]] if isinstance(exec_save_if[0], int) else exec_save_if[0]

    def maybe_wait(srcs):
      """Emit waitcnt if any source (or transitive source) is pending from an async load."""
      def needs_wait(src, visited=None):
        if visited is None: visited = set()
        if src in visited: return False
        visited.add(src)
        if src in pending_waits: return True
        # Check transitive sources (e.g., GEP -> LOAD)
        for s in src.src:
          if needs_wait(s, visited): return True
        return False
      for src in srcs:
        if needs_wait(src):
          code.append(s_waitcnt(waitcnt(vmcnt=0, lgkmcnt=0)))
          pending_waits.clear()
          return

    # Fixed registers
    kernarg_ptr = s[0:2]  # s[0:1] holds kernarg pointer
    group_id = (s[2], s[3], s[4])  # s[2:4] holds group ID xyz
    local_id = (v[0], v[1], v[2])  # v[0:2] holds local ID xyz

    def get_reg(u: UOp) -> VGPR | SGPR | int:
      """Get register for a UOp, handling constants and aliases."""
      import struct, math
      if u in r:
        val = r[u]
        # INDEX ops store (reg, cond) tuples - extract just the register
        if isinstance(val, tuple): return val[0]
        return val
      if u.op is Ops.CONST:
        val = u.arg
        if val is Invalid: return 0  # Invalid index - return safe address (will be masked anyway)
        if isinstance(val, bool): return 1 if val else 0  # Handle bool before int (bool is subclass of int)
        if isinstance(val, float):
          if val in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0): return val
          # Convert inf/nan to hex representation
          if math.isinf(val) or math.isnan(val):
            val = struct.unpack("I", struct.pack("f", val))[0]
        elif isinstance(val, int) and -16 <= val <= 64: return val
        # Load literal constant into register
        # For 64-bit types, need to load both low and high 32 bits
        if u.dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          reg = ra.alloc_vgpr_range(u, 2)
          lo = val & 0xFFFFFFFF
          hi = (val >> 32) & 0xFFFFFFFF
          code.append(v_mov_b32_e32(v[reg.idx], lo))
          code.append(v_mov_b32_e32(v[reg.idx + 1], hi))
          r[u] = reg
          return reg
        reg = ra.alloc_vgpr(u)
        code.append(v_mov_b32_e32(reg, val))
        r[u] = reg
        return reg
      raise ValueError(f"No register for {u}")

    def emit_cmp(op: Ops, dtype: DType, dst: VGPR, a, b):
      """Emit comparison instruction based on dtype."""
      # VOPC encoding: src0 can be constant, vsrc1 must be VGPR
      # For non-symmetric comparisons (LT), we swap and use the opposite (GT)
      cmp_map = {
        (Ops.CMPLT, dtypes.float32): v_cmp_lt_f32_e32, (Ops.CMPLT, dtypes.int32): v_cmp_lt_i32_e32, (Ops.CMPLT, dtypes.uint32): v_cmp_lt_u32_e32,
        (Ops.CMPEQ, dtypes.float32): v_cmp_eq_f32_e32, (Ops.CMPEQ, dtypes.int32): v_cmp_eq_i32_e32, (Ops.CMPEQ, dtypes.uint32): v_cmp_eq_u32_e32,
        (Ops.CMPNE, dtypes.float32): v_cmp_neq_f32_e32, (Ops.CMPNE, dtypes.int32): v_cmp_ne_i32_e32, (Ops.CMPNE, dtypes.uint32): v_cmp_ne_u32_e32,
      }
      # GT versions for swapping CMPLT: a < b ⇔ b > a
      cmp_gt_map = {
        (Ops.CMPLT, dtypes.float32): v_cmp_gt_f32_e32, (Ops.CMPLT, dtypes.int32): v_cmp_gt_i32_e32, (Ops.CMPLT, dtypes.uint32): v_cmp_gt_u32_e32,
      }
      base_dtype = dtypes.float32 if dtypes.is_float(dtype) else dtypes.int32 if dtype in (dtypes.int8, dtypes.int16, dtypes.int32) else dtypes.uint32
      def is_const(x): return isinstance(x, (int, float))
      # For CMPLT with constant in vsrc1 position, swap and use GT
      if op is Ops.CMPLT and is_const(b) and not is_const(a):
        cmp_fn = cmp_gt_map.get((op, base_dtype))
        if cmp_fn:
          code.append(cmp_fn(b, a))  # b > a ⇔ a < b
          code.append(v_cndmask_b32_e64(dst, 0, 1, VCC_LO))
          return
      # For symmetric comparisons, just swap
      cmp_fn = cmp_map.get((op, base_dtype))
      if cmp_fn:
        if op in (Ops.CMPEQ, Ops.CMPNE) and is_const(b) and not is_const(a):
          a, b = b, a
        code.append(cmp_fn(a, b))  # VOPC implicitly writes to VCC
        code.append(v_cndmask_b32_e64(dst, 0, 1, VCC_LO))  # Use VOP3: src2 is the VCC condition

    def emit_alu(u: UOp, dst: VGPR):
      """Emit ALU instruction."""
      op, dtype = u.op, u.dtype
      srcs = [get_reg(s) for s in u.src]
      a, b = (srcs[0], srcs[1]) if len(srcs) >= 2 else (srcs[0], 0)

      # For VOP2: src0 can be constant/literal, but vsrc1 must be VGPR
      # Swap operands for commutative ops when b is constant and a is not
      def is_const(x): return isinstance(x, (int, float))
      if op in (Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX) and is_const(b) and not is_const(a):
        a, b = b, a

      if op is Ops.ADD:
        if dtypes.is_float(dtype):
          if dtype == dtypes.float64:
            code.append(v_add_f64(dst, a, b))
          else:
            code.append(v_add_f32_e32(dst, a, b))
        elif dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          # 64-bit integer add requires carry chain
          # a and b are VGPRs pointing to register pairs (low, high)
          code.append(v_add_co_u32(v[dst.idx], VCC_LO, v[a.idx], v[b.idx]))  # low + carry out
          code.append(v_add_co_ci_u32_e32(v[dst.idx + 1], v[a.idx + 1], v[b.idx + 1]))  # high + carry in
        else:
          code.append(v_add_nc_u32_e32(dst, a, b))
      elif op is Ops.SUB:
        if dtypes.is_float(dtype):
          if dtype == dtypes.float64:
            # v_sub_f64 doesn't exist - use v_add_f64 with negated b
            code.append(v_mul_f64(dst, -1.0, b))  # negate b
            code.append(v_add_f64(dst, a, dst))   # a + (-b)
          else:
            code.append(v_sub_f32_e32(dst, a, b))
        elif dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          # 64-bit integer sub requires borrow chain
          code.append(v_sub_co_u32(v[dst.idx], VCC_LO, v[a.idx], v[b.idx]))  # low - borrow out
          code.append(v_sub_co_ci_u32_e32(v[dst.idx + 1], v[a.idx + 1], v[b.idx + 1]))  # high - borrow in
        else:
          code.append(v_sub_nc_u32_e32(dst, a, b))
      elif op is Ops.MUL:
        if dtypes.is_float(dtype):
          code.append(v_mul_f32_e32(dst, a, b))
        elif dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          # 64-bit multiply: need to handle signed cast with large constant specially
          # Check for fast_idiv pattern: signed 32-bit cast * large constant (high bit set)
          a_uop, b_uop = u.src[0], u.src[1]
          a_is_signed_cast = a_uop.op is Ops.CAST and a_uop.src[0].dtype == dtypes.int32
          b_is_const_hibit = b_uop.op is Ops.CONST and isinstance(b_uop.arg, int) and (b_uop.arg & 0x80000000) != 0
          a_reg = a if isinstance(a, (int, float)) else v[a.idx]
          b_reg = b if isinstance(b, (int, float)) else v[b.idx]
          if dtype in (dtypes.int64, dtypes.long) and a_is_signed_cast and b_is_const_hibit:
            # Special case for fast_idiv: do unsigned multiply then correct for sign
            # When a < 0: unsigned(a) = a + 2^32, so unsigned_result = a*b + 2^32*b
            # High 32 bits are off by b, so we subtract b when a < 0
            b_lo = b_uop.arg & 0xFFFFFFFF
            scratch = ra.alloc_vgpr(u)
            # Get the source register of the CAST (the original signed int32)
            a_src_reg = get_reg(a_uop.src[0])
            code.append(v_mul_lo_u32(v[dst.idx], a_reg, b_lo))      # dst_lo = a * b (low 32)
            code.append(v_mul_hi_u32(v[dst.idx + 1], a_reg, b_lo))  # dst_hi = a * b (high 32)
            code.append(v_cmp_gt_i32_e32(0, a_src_reg))  # vcc_lo = (0 > a), i.e., (a < 0)
            code.append(v_cndmask_b32_e64(scratch, 0, b_lo, VCC_LO))  # scratch = a < 0 ? b : 0
            code.append(v_sub_nc_u32_e32(v[dst.idx + 1], v[dst.idx + 1], scratch))  # dst_hi -= scratch
          elif dtype in (dtypes.int64, dtypes.long):
            code.append(v_mad_i64_i32(dst, NULL, a_reg, b_reg, 0))
          else:
            code.append(v_mad_u64_u32(dst, NULL, a_reg, b_reg, 0))
        else:
          code.append(v_mul_lo_u32(dst, a, b))
      elif op is Ops.AND: code.append(v_and_b32_e32(dst, a, b))
      elif op is Ops.OR: code.append(v_or_b32_e32(dst, a, b))
      elif op is Ops.XOR: code.append(v_xor_b32_e32(dst, a, b))
      elif op is Ops.SHL:
        if dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          code.append(v_lshlrev_b64(dst, b, a))  # 64-bit shift left
        else:
          code.append(v_lshlrev_b32_e32(dst, b, a))
      elif op is Ops.SHR:
        src_dtype = u.src[0].dtype
        # Handle 64-bit shift right: for shift >= 32, result = high_reg >> (shift - 32)
        if src_dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          shift_amt = b if isinstance(b, int) else None
          if shift_amt is not None and shift_amt >= 32:
            # Source is in register pair v[idx:idx+1], need high register (idx+1)
            src_uop = u.src[0]
            src_reg = r[src_uop]  # Get register from the r mapping
            src_idx = src_reg.idx if isinstance(src_reg, VGPR) else src_reg
            high_reg = v[src_idx + 1]
            adj_shift = shift_amt - 32
            if src_dtype in (dtypes.uint64, dtypes.ulong):
              code.append(v_lshrrev_b32_e32(dst, adj_shift, high_reg))
            else:
              code.append(v_ashrrev_i32_e32(dst, adj_shift, high_reg))
          else:
            # shift < 32 or dynamic shift - needs more complex handling
            code.append(v_lshrrev_b32_e32(dst, b, a) if src_dtype in (dtypes.uint64, dtypes.ulong) else v_ashrrev_i32_e32(dst, b, a))
        else:
          code.append(v_lshrrev_b32_e32(dst, b, a) if dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8) else v_ashrrev_i32_e32(dst, b, a))
      elif op is Ops.MAX:
        if dtypes.is_float(dtype): code.append(v_max_f32_e32(dst, a, b))
        elif dtype in (dtypes.int32, dtypes.int16, dtypes.int8): code.append(v_max_i32_e32(dst, a, b))
        else: code.append(v_max_u32_e32(dst, a, b))
      elif op is Ops.MULACC:
        c = srcs[2] if len(srcs) > 2 else 0
        if dtype == dtypes.float64:
          code.append(v_fma_f64(dst, a, b, c))
        else:
          code.append(v_fma_f32(dst, a, b, c))
      elif op is Ops.RECIPROCAL: code.append(v_rcp_f32_e32(dst, a))
      elif op is Ops.SQRT: code.append(v_sqrt_f32_e32(dst, a))
      elif op is Ops.EXP2: code.append(v_exp_f32_e32(dst, a))
      elif op is Ops.LOG2: code.append(v_log_f32_e32(dst, a))
      elif op is Ops.TRUNC: code.append(v_trunc_f32_e32(dst, a))
      elif op is Ops.SIN: code.append(v_sin_f32_e32(dst, a))
      elif op is Ops.NEG:
        if dtypes.is_float(dtype): code.append(v_mul_f32_e32(dst, -1.0, a))
        else: code.append(v_sub_nc_u32_e32(dst, 0, a))
      elif op in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE):
        emit_cmp(op, u.src[0].dtype, dst, a, b)
      elif op is Ops.WHERE:
        cond, true_val, false_val = srcs[0], srcs[1], srcs[2]
        code.append(v_cmp_ne_i32_e32(0, cond))  # VOPC: src0=constant, vsrc1=VGPR; 0 != cond ⇔ cond != 0
        code.append(v_cndmask_b32_e64(dst, false_val, true_val, VCC_LO))
      elif op is Ops.IDIV:
        # Integer division using floating-point approximation
        # quotient = trunc(float(a) * rcp(float(b)))
        # For signed: handle signs, do unsigned div, restore sign
        is_signed = dtype in (dtypes.int32, dtypes.int16, dtypes.int8)
        if is_signed:
          # Compute absolute values and track signs
          tmp_abs_a = ra.alloc_vgpr(u)  # |a|
          tmp_abs_b = ra.alloc_vgpr(u)  # |b|
          tmp_abs_a_orig = ra.alloc_vgpr(u)  # copy of |a| for correction
          tmp_abs_b_orig = ra.alloc_vgpr(u)  # copy of |b| for correction
          tmp_sign = ra.alloc_vgpr(u)  # sign bit
          tmp_neg = ra.alloc_vgpr(u)   # temp for negation
          tmp_q = ra.alloc_vgpr(u)     # quotient
          tmp_rem = ra.alloc_vgpr(u)   # temp for correction
          # |a|: abs(a) = a >= 0 ? a : -a
          code.append(v_sub_nc_u32_e32(tmp_neg, 0, a))  # -a
          code.append(v_cmp_gt_i32_e32(0, a))  # 0 > a means a < 0
          code.append(v_cndmask_b32_e64(tmp_abs_a, a, tmp_neg, VCC_LO))  # |a|
          code.append(v_mov_b32_e32(tmp_abs_a_orig, tmp_abs_a))  # save |a|
          # |b|: abs(b) = b >= 0 ? b : -b
          code.append(v_sub_nc_u32_e32(tmp_neg, 0, b))  # -b
          code.append(v_cmp_gt_i32_e32(0, b))  # 0 > b means b < 0
          code.append(v_cndmask_b32_e64(tmp_abs_b, b, tmp_neg, VCC_LO))  # |b|
          code.append(v_mov_b32_e32(tmp_abs_b_orig, tmp_abs_b))  # save |b|
          # sign = (a < 0) XOR (b < 0) -> top bit indicates result is negative
          code.append(v_xor_b32_e32(tmp_sign, a, b))
          # Do unsigned division: |a| / |b|
          code.append(v_cvt_f32_u32_e32(tmp_abs_a, tmp_abs_a))  # float(|a|)
          code.append(v_cvt_f32_u32_e32(tmp_abs_b, tmp_abs_b))  # float(|b|)
          code.append(v_rcp_f32_e32(tmp_abs_b, tmp_abs_b))  # 1/|b|
          code.append(v_mul_f32_e32(tmp_abs_a, tmp_abs_a, tmp_abs_b))  # |a|/|b|
          code.append(v_trunc_f32_e32(tmp_abs_a, tmp_abs_a))  # trunc - may be 1 too low
          code.append(v_cvt_u32_f32_e32(tmp_q, tmp_abs_a))  # quotient estimate
          # Correct: if (q+1)*|b| <= |a|, then q should be q+1
          code.append(v_add_nc_u32_e32(tmp_abs_a, 1, tmp_q))  # q+1
          code.append(v_mul_lo_u32(tmp_rem, tmp_abs_a, tmp_abs_b_orig))  # (q+1)*|b|
          code.append(v_cmp_gt_u32_e32(tmp_rem, tmp_abs_a_orig))  # (q+1)*|b| > |a| means q is correct
          code.append(v_cndmask_b32_e64(dst, tmp_abs_a, tmp_q, VCC_LO))  # dst = vcc ? q : q+1
          # Negate result if signs differ (top bit of tmp_sign is set)
          code.append(v_sub_nc_u32_e32(tmp_neg, 0, dst))  # -quotient
          code.append(v_cmp_gt_i32_e32(0, tmp_sign))  # 0 > tmp_sign means top bit is 1
          code.append(v_cndmask_b32_e64(dst, dst, tmp_neg, VCC_LO))
        else:
          # Unsigned division using floating-point with correction
          # The rcp instruction has limited precision, so we need to correct
          tmp_a = ra.alloc_vgpr(u)
          tmp_b = ra.alloc_vgpr(u)
          tmp_q = ra.alloc_vgpr(u)
          tmp_rem = ra.alloc_vgpr(u)
          code.append(v_cvt_f32_u32_e32(tmp_a, a))  # float(a)
          code.append(v_cvt_f32_u32_e32(tmp_b, b))  # float(b)
          code.append(v_rcp_f32_e32(tmp_b, tmp_b))  # 1/b (approximate)
          code.append(v_mul_f32_e32(tmp_a, tmp_a, tmp_b))  # a/b
          code.append(v_trunc_f32_e32(tmp_a, tmp_a))  # trunc(a/b) - may be 1 too low
          code.append(v_cvt_u32_f32_e32(tmp_q, tmp_a))  # quotient estimate
          # Correct: if (q+1)*b <= a, then q should be q+1
          code.append(v_add_nc_u32_e32(tmp_a, 1, tmp_q))  # q+1
          code.append(v_mul_lo_u32(tmp_rem, tmp_a, b))  # (q+1)*b
          code.append(v_cmp_gt_u32_e32(tmp_rem, a))  # (q+1)*b > a means q is correct
          code.append(v_cndmask_b32_e64(dst, tmp_a, tmp_q, VCC_LO))  # dst = vcc ? q : q+1
      elif op is Ops.MOD:
        # Modulo: a % b = a - (a // b) * b
        is_signed = dtype in (dtypes.int32, dtypes.int16, dtypes.int8)
        tmp1 = ra.alloc_vgpr(u)
        tmp2 = ra.alloc_vgpr(u)
        if is_signed:
          # Compute quotient first (same as IDIV signed)
          tmp_sign = ra.alloc_vgpr(u)
          tmp_abs_a = ra.alloc_vgpr(u)
          tmp_abs_b = ra.alloc_vgpr(u)
          # |a|
          code.append(v_sub_nc_u32_e32(tmp_abs_a, 0, a))  # -a
          code.append(v_cmp_gt_i32_e32(0, a))
          code.append(v_cndmask_b32_e64(tmp_abs_a, a, tmp_abs_a, VCC_LO))  # |a|
          # |b|
          code.append(v_sub_nc_u32_e32(tmp_abs_b, 0, b))  # -b
          code.append(v_cmp_gt_i32_e32(0, b))
          code.append(v_cndmask_b32_e64(tmp_abs_b, b, tmp_abs_b, VCC_LO))  # |b|
          # Unsigned division of |a| / |b|
          code.append(v_cvt_f32_u32_e32(tmp1, tmp_abs_a))
          code.append(v_cvt_f32_u32_e32(tmp2, tmp_abs_b))
          code.append(v_rcp_f32_e32(tmp2, tmp2))
          code.append(v_mul_f32_e32(tmp1, tmp1, tmp2))
          code.append(v_trunc_f32_e32(tmp1, tmp1))
          code.append(v_cvt_u32_f32_e32(tmp1, tmp1))  # quotient magnitude
          # mod = |a| - quotient * |b|
          code.append(v_mul_lo_u32(tmp2, tmp1, tmp_abs_b))
          code.append(v_sub_nc_u32_e32(dst, tmp_abs_a, tmp2))  # |a| % |b|
          # Result sign follows a's sign
          code.append(v_sub_nc_u32_e32(tmp1, 0, dst))  # -result
          code.append(v_cmp_gt_i32_e32(0, a))
          code.append(v_cndmask_b32_e64(dst, dst, tmp1, VCC_LO))
        else:
          # Unsigned: a % b = a - (a // b) * b
          # Save original 'a' and 'b' since we'll overwrite temps
          tmp_a_orig = ra.alloc_vgpr(u)  # save original a
          tmp_b_orig = ra.alloc_vgpr(u)  # save original b
          tmp_q = ra.alloc_vgpr(u)       # quotient
          tmp_rem = ra.alloc_vgpr(u)     # temp for correction
          code.append(v_mov_b32_e32(tmp_a_orig, a))  # save a
          code.append(v_mov_b32_e32(tmp_b_orig, b))  # save b
          code.append(v_cvt_f32_u32_e32(tmp1, a))    # float(a)
          code.append(v_cvt_f32_u32_e32(tmp2, b))    # float(b)
          code.append(v_rcp_f32_e32(tmp2, tmp2))     # 1/b
          code.append(v_mul_f32_e32(tmp1, tmp1, tmp2))  # a/b
          code.append(v_trunc_f32_e32(tmp1, tmp1))   # trunc(a/b)
          code.append(v_cvt_u32_f32_e32(tmp_q, tmp1))  # quotient estimate
          # Correct quotient: if (q+1)*b <= a, then q should be q+1
          code.append(v_add_nc_u32_e32(tmp1, 1, tmp_q))  # q+1
          code.append(v_mul_lo_u32(tmp_rem, tmp1, tmp_b_orig))  # (q+1)*b
          code.append(v_cmp_gt_u32_e32(tmp_rem, tmp_a_orig))  # (q+1)*b > a
          code.append(v_cndmask_b32_e64(tmp_q, tmp1, tmp_q, VCC_LO))  # corrected q
          # mod = a - quotient * b
          code.append(v_mul_lo_u32(tmp2, tmp_q, tmp_b_orig))  # quotient * b
          code.append(v_sub_nc_u32_e32(dst, tmp_a_orig, tmp2))  # a - quotient * b
      else:
        code.append(v_mov_b32_e32(dst, a))  # Fallback: just move

    # Process UOps
    for i, u in enumerate(uops):
      ra.free_dead_regs(i)  # Free registers that are no longer needed

      if u.op is Ops.SINK: continue

      elif u.op is Ops.DEFINE_GLOBAL:
        bufs.append(u)
        kernarg_offset[u] = current_offset
        current_offset += 8  # 64-bit pointer
        r[u] = ra.alloc_sgpr_pair(u)

      elif u.op is Ops.DEFINE_VAR:
        vars_.append(u)
        kernarg_offset[u] = current_offset
        current_offset += 4
        r[u] = ra.alloc_sgpr(u) or ra.alloc_vgpr(u)

      elif u.op is Ops.DEFINE_LOCAL:
        # Size can be from arg (tuple/int) or from dtype.size
        if isinstance(u.arg, tuple):
          size = u.arg[1]
        elif isinstance(u.arg, int) and u.arg > 0:
          size = u.arg
        else:
          size = u.dtype.size if hasattr(u.dtype, 'size') else 0
        u_lds_size = u.dtype.itemsize * size
        lds_size = max(lds_size, u_lds_size)
        r[u] = 0  # LDS base is always 0 (offsets are relative)

      elif u.op is Ops.DEFINE_REG:
        # Register-space buffer for WMMA/tensor core outputs
        # dtype is a pointer to register space with size = element count
        num_regs = u.dtype.size if hasattr(u.dtype, 'size') and u.dtype.size > 0 else 16
        r[u] = ra.alloc_vgpr_range(u, num_regs)

      elif u.op is Ops.SPECIAL:
        # SPECIAL arg is a string like 'lidx0', 'gidx1', etc.
        # With .amdhsa_system_vgpr_workitem_id 2, workitem IDs are PACKED in v0:
        #   bits 0-9:   workitem_id_x
        #   bits 10-19: workitem_id_y
        #   bits 20-29: workitem_id_z
        # Group IDs are in s2, s3, s4 as separate values
        axis = u.arg
        idx = int(axis[-1])
        dst = ra.alloc_vgpr(u)
        r[u] = dst
        if axis.startswith('l'):  # Local ID - extract from packed v0
          if idx == 0:
            code.append(v_and_b32_e32(dst, 0x3ff, v[0]))  # bits 0-9
          else:
            code.append(v_bfe_u32(dst, v[0], idx * 10, 10))  # bits 10-19 or 20-29
        else:  # Group ID - copy from SGPR to VGPR
          code.append(v_mov_b32_e32(dst, s[2 + idx]))

      elif u.op is Ops.CONST:
        pass  # Handled lazily in get_reg

      elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR,
                    Ops.MAX, Ops.MULACC, Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2,
                    Ops.TRUNC, Ops.NEG, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.WHERE, Ops.SIN,
                    Ops.IDIV, Ops.MOD):
        maybe_wait(u.src)  # Wait for any pending loads used by this operation
        dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(u.dtype) else ra.alloc_vgpr(u)
        r[u] = dst
        emit_alu(u, dst)

      elif u.op is Ops.CAST:
        maybe_wait(u.src)  # Wait for source value
        src_reg = get_reg(u.src[0])
        src_dtype, dst_dtype = u.src[0].dtype, u.dtype
        if src_dtype == dst_dtype:
          r[u] = src_reg
        else:
          SMALL_SIGNED = (dtypes.int8, dtypes.int16)
          SMALL_UNSIGNED = (dtypes.uint8, dtypes.uint16)
          SMALL_INTS = SMALL_SIGNED + SMALL_UNSIGNED
          INT64_TYPES = (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong)
          dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(dst_dtype) else ra.alloc_vgpr(u)
          r[u] = dst
          # float32 <-> int32/uint32
          if dst_dtype == dtypes.float32 and src_dtype == dtypes.int32:
            code.append(v_cvt_f32_i32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.uint32:
            code.append(v_cvt_f32_u32_e32(dst, src_reg))
          elif dst_dtype == dtypes.int32 and src_dtype == dtypes.float32:
            code.append(v_cvt_i32_f32_e32(dst, src_reg))
          elif dst_dtype == dtypes.uint32 and src_dtype == dtypes.float32:
            code.append(v_cvt_u32_f32_e32(dst, src_reg))
          # float32 <-> small ints (via 32-bit convert)
          elif dst_dtype == dtypes.float32 and src_dtype in SMALL_SIGNED:
            # Sign-extend to 32-bit then convert
            tmp = ra.alloc_vgpr(u)
            bits = 8 if src_dtype.itemsize == 1 else 16
            code.append(v_bfe_i32(tmp, src_reg, 0, bits))
            code.append(v_cvt_f32_i32_e32(dst, tmp))
          elif dst_dtype == dtypes.float32 and src_dtype in SMALL_UNSIGNED:
            code.append(v_cvt_f32_u32_e32(dst, src_reg))  # Zero-extension is implicit
          elif dst_dtype in SMALL_SIGNED and src_dtype == dtypes.float32:
            code.append(v_cvt_i32_f32_e32(dst, src_reg))  # Truncation is implicit on store
          elif dst_dtype in SMALL_UNSIGNED and src_dtype == dtypes.float32:
            code.append(v_cvt_u32_f32_e32(dst, src_reg))
          # float16 <-> float32
          elif dst_dtype == dtypes.float16 and src_dtype == dtypes.float32:
            code.append(v_cvt_f16_f32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.float16:
            code.append(v_cvt_f32_f16_e32(dst, src_reg))
          # float16 <-> int (via float32)
          elif dst_dtype == dtypes.float16 and dtypes.is_int(src_dtype) and src_dtype not in INT64_TYPES:
            tmp = ra.alloc_vgpr(u)
            if src_dtype in SMALL_SIGNED:
              bits = 8 if src_dtype.itemsize == 1 else 16
              code.append(v_bfe_i32(tmp, src_reg, 0, bits))
              code.append(v_cvt_f32_i32_e32(tmp, tmp))
            elif src_dtype in SMALL_UNSIGNED:
              code.append(v_cvt_f32_u32_e32(tmp, src_reg))
            elif src_dtype == dtypes.int32:
              code.append(v_cvt_f32_i32_e32(tmp, src_reg))
            else:  # uint32
              code.append(v_cvt_f32_u32_e32(tmp, src_reg))
            code.append(v_cvt_f16_f32_e32(dst, tmp))
          elif dtypes.is_int(dst_dtype) and dst_dtype not in INT64_TYPES and src_dtype == dtypes.float16:
            tmp = ra.alloc_vgpr(u)
            code.append(v_cvt_f32_f16_e32(tmp, src_reg))
            if dst_dtype in (dtypes.int8, dtypes.int16, dtypes.int32):
              code.append(v_cvt_i32_f32_e32(dst, tmp))
            else:
              code.append(v_cvt_u32_f32_e32(dst, tmp))
          # bfloat16 <-> float32
          elif dst_dtype == dtypes.bfloat16 and src_dtype == dtypes.float32:
            code.append(v_lshrrev_b32_e32(dst, 16, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.bfloat16:
            code.append(v_lshlrev_b32_e32(dst, 16, src_reg))
          # bfloat16 <-> int (via float32)
          elif dst_dtype == dtypes.bfloat16 and dtypes.is_int(src_dtype) and src_dtype not in INT64_TYPES:
            tmp = ra.alloc_vgpr(u)
            if src_dtype in SMALL_SIGNED:
              bits = 8 if src_dtype.itemsize == 1 else 16
              code.append(v_bfe_i32(tmp, src_reg, 0, bits))
              code.append(v_cvt_f32_i32_e32(tmp, tmp))
            elif src_dtype in SMALL_UNSIGNED:
              code.append(v_cvt_f32_u32_e32(tmp, src_reg))
            elif src_dtype == dtypes.int32:
              code.append(v_cvt_f32_i32_e32(tmp, src_reg))
            else:
              code.append(v_cvt_f32_u32_e32(tmp, src_reg))
            code.append(v_lshrrev_b32_e32(dst, 16, tmp))
          elif dtypes.is_int(dst_dtype) and dst_dtype not in INT64_TYPES and src_dtype == dtypes.bfloat16:
            tmp = ra.alloc_vgpr(u)
            code.append(v_lshlrev_b32_e32(tmp, 16, src_reg))  # bf16 -> f32
            if dst_dtype in (dtypes.int8, dtypes.int16, dtypes.int32):
              code.append(v_cvt_i32_f32_e32(dst, tmp))
            else:
              code.append(v_cvt_u32_f32_e32(dst, tmp))
          # float64 <-> float32
          elif dst_dtype == dtypes.float64 and src_dtype == dtypes.float32:
            code.append(v_cvt_f64_f32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.float64:
            code.append(v_cvt_f32_f64_e32(dst, src_reg))
          # float64 <-> int32/uint32
          elif dst_dtype == dtypes.float64 and src_dtype == dtypes.int32:
            code.append(v_cvt_f64_i32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float64 and src_dtype == dtypes.uint32:
            code.append(v_cvt_f64_u32_e32(dst, src_reg))
          elif dst_dtype == dtypes.int32 and src_dtype == dtypes.float64:
            code.append(v_cvt_i32_f64_e32(dst, src_reg))
          elif dst_dtype == dtypes.uint32 and src_dtype == dtypes.float64:
            code.append(v_cvt_u32_f64_e32(dst, src_reg))
          # float64 <-> small ints (via float64 <-> int32)
          elif dst_dtype == dtypes.float64 and src_dtype in SMALL_SIGNED:
            tmp = ra.alloc_vgpr(u)
            bits = 8 if src_dtype.itemsize == 1 else 16
            code.append(v_bfe_i32(tmp, src_reg, 0, bits))
            code.append(v_cvt_f64_i32_e32(dst, tmp))
          elif dst_dtype == dtypes.float64 and src_dtype in SMALL_UNSIGNED:
            code.append(v_cvt_f64_u32_e32(dst, src_reg))
          elif dst_dtype in SMALL_SIGNED and src_dtype == dtypes.float64:
            code.append(v_cvt_i32_f64_e32(dst, src_reg))
          elif dst_dtype in SMALL_UNSIGNED and src_dtype == dtypes.float64:
            code.append(v_cvt_u32_f64_e32(dst, src_reg))
          # float64 <-> float16/bfloat16 (via float32)
          elif dst_dtype == dtypes.float64 and src_dtype == dtypes.float16:
            tmp = ra.alloc_vgpr(u)
            code.append(v_cvt_f32_f16_e32(tmp, src_reg))
            code.append(v_cvt_f64_f32_e32(dst, tmp))
          elif dst_dtype == dtypes.float16 and src_dtype == dtypes.float64:
            tmp = ra.alloc_vgpr(u)
            code.append(v_cvt_f32_f64_e32(tmp, src_reg))
            code.append(v_cvt_f16_f32_e32(dst, tmp))
          elif dst_dtype == dtypes.float64 and src_dtype == dtypes.bfloat16:
            tmp = ra.alloc_vgpr(u)
            code.append(v_lshlrev_b32_e32(tmp, 16, src_reg))
            code.append(v_cvt_f64_f32_e32(dst, tmp))
          elif dst_dtype == dtypes.bfloat16 and src_dtype == dtypes.float64:
            tmp = ra.alloc_vgpr(u)
            code.append(v_cvt_f32_f64_e32(tmp, src_reg))
            code.append(v_lshrrev_b32_e32(dst, 16, tmp))
          # int64 -> smaller types (just take low 32 bits)
          elif dst_dtype not in INT64_TYPES and src_dtype in INT64_TYPES:
            code.append(v_mov_b32_e32(dst, src_reg))  # src_reg is already the low 32 bits
          # smaller types -> int64 (extend to 64 bits)
          elif dst_dtype in INT64_TYPES and src_dtype not in INT64_TYPES:
            if src_dtype in SMALL_SIGNED:
              bits = 8 if src_dtype.itemsize == 1 else 16
              code.append(v_bfe_i32(v[dst.idx], src_reg, 0, bits))
              code.append(v_ashrrev_i32_e32(v[dst.idx + 1], 31, v[dst.idx]))  # Sign-extend high word
            elif src_dtype == dtypes.int32:
              code.append(v_mov_b32_e32(v[dst.idx], src_reg))
              code.append(v_ashrrev_i32_e32(v[dst.idx + 1], 31, src_reg))  # Sign-extend high word
            else:  # unsigned types
              code.append(v_mov_b32_e32(v[dst.idx], src_reg))
              code.append(v_mov_b32_e32(v[dst.idx + 1], 0))  # Zero-extend high word
          # int64 <-> float (complex - convert via f64)
          elif dst_dtype in INT64_TYPES and src_dtype == dtypes.float64:
            # Convert f64 to int32, then extend
            code.append(v_cvt_i32_f64_e32(v[dst.idx], src_reg) if dst_dtype in (dtypes.int64, dtypes.long) else v_cvt_u32_f64_e32(v[dst.idx], src_reg))
            if dst_dtype in (dtypes.int64, dtypes.long):
              code.append(v_ashrrev_i32_e32(v[dst.idx + 1], 31, v[dst.idx]))
            else:
              code.append(v_mov_b32_e32(v[dst.idx + 1], 0))
          elif src_dtype in INT64_TYPES and dst_dtype == dtypes.float64:
            # Convert low 32 bits to f64, add high 32 bits * 2^32
            tmp = ra.alloc_vgpr_pair(u)
            is_signed = src_dtype in (dtypes.int64, dtypes.long)
            code.append(v_cvt_f64_u32_e32(dst, src_reg))  # Low word (always unsigned)
            if is_signed:
              code.append(v_cvt_f64_i32_e32(tmp, v[src_reg.idx + 1]))  # High word (signed)
            else:
              code.append(v_cvt_f64_u32_e32(tmp, v[src_reg.idx + 1]))  # High word (unsigned)
            code.append(v_ldexp_f64(tmp, tmp, 32))  # tmp *= 2^32
            code.append(v_add_f64(dst, dst, tmp))  # result = low + high * 2^32
          elif src_dtype in INT64_TYPES and dst_dtype == dtypes.float32:
            # Convert via float64
            tmp = ra.alloc_vgpr_pair(u)
            tmp2 = ra.alloc_vgpr_pair(u)
            is_signed = src_dtype in (dtypes.int64, dtypes.long)
            code.append(v_cvt_f64_u32_e32(tmp, src_reg))
            if is_signed:
              code.append(v_cvt_f64_i32_e32(tmp2, v[src_reg.idx + 1]))
            else:
              code.append(v_cvt_f64_u32_e32(tmp2, v[src_reg.idx + 1]))
            code.append(v_ldexp_f64(tmp2, tmp2, 32))
            code.append(v_add_f64(tmp, tmp, tmp2))
            code.append(v_cvt_f32_f64_e32(dst, tmp))
          elif dst_dtype in INT64_TYPES and src_dtype == dtypes.float32:
            # Convert via float64
            tmp = ra.alloc_vgpr_pair(u)
            code.append(v_cvt_f64_f32_e32(tmp, src_reg))
            code.append(v_cvt_i32_f64_e32(v[dst.idx], tmp) if dst_dtype in (dtypes.int64, dtypes.long) else v_cvt_u32_f64_e32(v[dst.idx], tmp))
            if dst_dtype in (dtypes.int64, dtypes.long):
              code.append(v_ashrrev_i32_e32(v[dst.idx + 1], 31, v[dst.idx]))
            else:
              code.append(v_mov_b32_e32(v[dst.idx + 1], 0))
          # small int <-> int32 (sign/zero extension)
          elif dst_dtype == dtypes.int32 and src_dtype in SMALL_SIGNED:
            bits = 8 if src_dtype.itemsize == 1 else 16
            code.append(v_bfe_i32(dst, src_reg, 0, bits))
          elif dst_dtype == dtypes.int32 and src_dtype in SMALL_UNSIGNED:
            code.append(v_mov_b32_e32(dst, src_reg))  # Zero-extension implicit
          elif dst_dtype == dtypes.uint32 and src_dtype in SMALL_INTS:
            code.append(v_mov_b32_e32(dst, src_reg))  # Zero-extension implicit
          elif dst_dtype in SMALL_INTS and src_dtype in (dtypes.int32, dtypes.uint32):
            code.append(v_mov_b32_e32(dst, src_reg))  # Truncation is implicit
          # small int <-> small int
          elif dst_dtype in SMALL_INTS and src_dtype in SMALL_INTS:
            if dst_dtype in SMALL_SIGNED and src_dtype in SMALL_SIGNED and dst_dtype.itemsize > src_dtype.itemsize:
              # Sign extend
              bits = 8 if src_dtype.itemsize == 1 else 16
              code.append(v_bfe_i32(dst, src_reg, 0, bits))
            else:
              code.append(v_mov_b32_e32(dst, src_reg))
          # bool conversions
          elif dst_dtype == dtypes.bool:
            if src_dtype in INT64_TYPES:
              tmp = ra.alloc_vgpr(u)
              code.append(v_or_b32_e32(tmp, src_reg, v[src_reg.idx + 1]))
              code.append(v_cmp_ne_u32_e32(0, tmp))
            elif src_dtype == dtypes.float64:
              code.append(v_cmp_neq_f32_e32(0, src_reg))  # Just check low word for now
            else:
              code.append(v_cmp_ne_i32_e32(0, src_reg))
            code.append(v_cndmask_b32_e64(dst, 0, 1, VCC_LO))
          elif src_dtype == dtypes.bool:
            if dst_dtype == dtypes.float32:
              code.append(v_cmp_ne_i32_e32(0, src_reg))
              code.append(v_cndmask_b32_e64(dst, 0, 0x3f800000, VCC_LO))  # 1.0f
            elif dst_dtype == dtypes.float16:
              code.append(v_cmp_ne_i32_e32(0, src_reg))
              code.append(v_cndmask_b32_e64(dst, 0, 0x3c00, VCC_LO))  # 1.0 in f16
            elif dst_dtype == dtypes.float64:
              code.append(v_cmp_ne_i32_e32(0, src_reg))
              code.append(v_cndmask_b32_e64(v[dst.idx], 0, 0, VCC_LO))
              code.append(v_cndmask_b32_e64(v[dst.idx + 1], 0, 0x3ff00000, VCC_LO))  # 1.0 in f64
            elif dst_dtype in INT64_TYPES:
              code.append(v_mov_b32_e32(v[dst.idx], src_reg))
              code.append(v_mov_b32_e32(v[dst.idx + 1], 0))
            else:
              code.append(v_mov_b32_e32(dst, src_reg))
          else:
            raise NotImplementedError(f"CAST from {src_dtype} to {dst_dtype} not implemented")

      elif u.op is Ops.BITCAST:
        r[u] = get_reg(u.src[0])  # Bitcast is just a reinterpretation

      elif u.op is Ops.INDEX:
        buf, idx = u.src[0], u.src[1]
        cond = u.src[2] if len(u.src) > 2 else None  # Optional condition (3rd source)
        if isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory: just the offset
          r[u] = (get_reg(idx), cond)  # Store as tuple with condition
        elif isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.REG:
          # Register space: just the offset (will be used to index into VGPR range)
          r[u] = (get_reg(idx), cond)
        else:
          # Global memory: INDEX stores just the byte offset (idx already scaled by rdna_uops)
          # LOAD/STORE will use saddr with the buffer's SGPR pair as base
          idx_reg = get_reg(idx)
          if isinstance(idx_reg, (int, float)):
            # Constant offset - load into VGPR
            dst = ra.alloc_vgpr(u)
            code.append(v_mov_b32_e32(dst, idx_reg))
            r[u] = (dst, cond)
          else:
            r[u] = (idx_reg, cond)

      elif u.op is Ops.LOAD:
        idx_uop = u.src[0]
        default_uop = u.src[1] if len(u.src) > 1 else None  # Optional default value
        # For INDEX, r[idx_uop] is a tuple (addr, cond)
        idx_result = r.get(idx_uop) if idx_uop in r else get_reg(idx_uop)
        if isinstance(idx_result, tuple):
          addr, cond_uop = idx_result
        else:
          addr, cond_uop = idx_result, None
        dtype = u.dtype
        itemsize = dtype.itemsize if hasattr(dtype, 'itemsize') else 4
        buf_uop = idx_uop.src[0] if idx_uop.op is Ops.INDEX else idx_uop

        if isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory load - allocate VGPRs based on size
          if itemsize == 16:
            dst = ra.alloc_vgpr_range(u, 4)
            code.append(ds_load_b128(vdst=dst, addr=addr))
          elif itemsize == 8:
            dst = ra.alloc_vgpr_pair(u)
            code.append(ds_load_b64(vdst=dst, addr=addr))
          else:
            dst = ra.alloc_vgpr(u)
            code.append(ds_load_b32(vdst=dst, addr=addr))
        elif isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.REG:
          # Register-space load: return register from the buffer's range
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if isinstance(addr, (int, float)):
            # addr is the element index, not byte offset
            reg_offset = int(addr)
            dst = v[buf_reg.idx + reg_offset]
          else:
            # Variable offset - use first register as fallback (TODO: proper indirect)
            dst = v[buf_reg.idx]
          r[u] = dst
          continue  # Skip the rest, no actual load needed
        else:
          # Global memory load: use buffer SGPR pair as saddr
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if itemsize == 16:
            dst = ra.alloc_vgpr_range(u, 4)  # b128 needs 4 VGPRs
          elif itemsize == 8:
            dst = ra.alloc_vgpr_pair(u)  # b64 needs 2 VGPRs
          else:
            dst = ra.alloc_vgpr(u)  # b32 and smaller

          # Handle conditional load (gated load with optional default value)
          if cond_uop is not None:
            # First, initialize dst with default value (or 0 if no default)
            if default_uop is not None:
              default_val = get_reg(default_uop)
              if isinstance(default_val, (int, float)):
                for j in range(itemsize // 4):
                  code.append(v_mov_b32_e32(v[dst.idx + j] if hasattr(dst, 'idx') else dst, default_val))
              else:
                # Copy from default vector
                for j in range(itemsize // 4):
                  code.append(v_mov_b32_e32(v[dst.idx + j] if hasattr(dst, 'idx') else dst, v[default_val.idx + j] if hasattr(default_val, 'idx') else default_val))
            else:
              # No default value - initialize to 0 for safety
              for j in range(itemsize // 4):
                code.append(v_mov_b32_e32(v[dst.idx + j] if hasattr(dst, 'idx') else dst, 0))

            # Set up exec mask based on condition (use dynamically allocated SGPR)
            cond_reg = get_reg(cond_uop)
            code.append(v_cmp_ne_i32_e32(0, cond_reg))  # VCC = (cond != 0)
            # Clamp address to 0 for masked lanes to prevent invalid memory accesses
            # Even with exec masking, garbage addresses can cause protection faults
            # IMPORTANT: Use a temp register to avoid corrupting addr which may be used by STORE later
            if isinstance(addr, VGPR):
              clamped_addr = ra.alloc_vgpr(u)
              code.append(v_cndmask_b32_e64(clamped_addr, 0, addr, VCC_LO))  # clamped = cond ? addr : 0
              addr = clamped_addr  # Use clamped address for this load only
            code.append(s_and_saveexec_b32(get_exec_save_load(), VCC_LO))  # Save exec, mask with condition
            # Now do the load (only executed by lanes where condition is true)

          if itemsize == 1:
            code.append(global_load_u8(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 2:
            code.append(global_load_u16(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 4:
            code.append(global_load_b32(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 8:
            code.append(global_load_b64(vdst=dst, addr=addr, saddr=buf_reg))
          else:
            code.append(global_load_b128(vdst=dst, addr=addr, saddr=buf_reg))

          # Restore exec mask if we masked it
          if cond_uop is not None:
            code.append(s_mov_b32(EXEC_LO, get_exec_save_load()))

        r[u] = dst
        pending_waits.add(u)  # Track that this load result needs wait before use

      elif u.op is Ops.STORE:
        maybe_wait(u.src)  # Wait for value to store
        idx_uop, val_uop = u.src[0], u.src[1]
        # For INDEX, r[idx_uop] is a tuple (addr, cond)
        idx_result = r.get(idx_uop) if idx_uop in r else get_reg(idx_uop)
        if isinstance(idx_result, tuple):
          addr, cond_uop = idx_result
        else:
          addr, cond_uop = idx_result, None
        val = get_reg(val_uop)
        dtype = val_uop.dtype
        itemsize = dtype.itemsize if hasattr(dtype, 'itemsize') else 4
        # STORE data operand must be a VGPR, not an inline constant
        if isinstance(val, (int, float)):
          if itemsize == 8:
            # 64-bit constant needs 2 VGPRs
            tmp = ra.alloc_vgpr_pair(val_uop)
            code.append(v_mov_b32_e32(v[tmp.idx], val & 0xffffffff if isinstance(val, int) else val))
            code.append(v_mov_b32_e32(v[tmp.idx + 1], (val >> 32) & 0xffffffff if isinstance(val, int) else 0))
            val = tmp
          else:
            tmp = ra.alloc_vgpr(val_uop)
            code.append(v_mov_b32_e32(tmp, val))
            val = tmp
        buf_uop = idx_uop.src[0] if idx_uop.op is Ops.INDEX else idx_uop

        if isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory store - use appropriate instruction based on size
          if itemsize == 16:
            code.append(ds_store_b128(addr=addr, data0=val))
          elif itemsize == 8:
            code.append(ds_store_b64(addr=addr, data0=val))
          else:
            code.append(ds_store_b32(addr=addr, data0=val))
        elif isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.REG:
          # Register-space store: copy to register range
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if isinstance(addr, (int, float)):
            # addr is the element index, not byte offset
            reg_offset = int(addr)
            code.append(v_mov_b32_e32(v[buf_reg.idx + reg_offset], val))
          else:
            # Variable offset - use first register as fallback (TODO: proper indirect)
            code.append(v_mov_b32_e32(v[buf_reg.idx], val))
        else:
          # Global memory store: use buffer SGPR pair as saddr
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result

          # Handle conditional store (mask exec for lanes where condition is false)
          if cond_uop is not None:
            cond_reg = get_reg(cond_uop)
            code.append(v_cmp_ne_i32_e32(0, cond_reg))  # VCC = (cond != 0)
            # Clamp address to 0 for masked lanes to prevent invalid memory accesses
            # IMPORTANT: Use a temp register to avoid corrupting addr which may be used elsewhere
            if isinstance(addr, VGPR):
              clamped_addr = ra.alloc_vgpr(val_uop)
              code.append(v_cndmask_b32_e64(clamped_addr, 0, addr, VCC_LO))  # clamped = cond ? addr : 0
              addr = clamped_addr  # Use clamped address for this store only
            code.append(s_and_saveexec_b32(get_exec_save_load(), VCC_LO))  # Save exec, mask with condition

          if itemsize == 1:
            code.append(global_store_b8(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 2:
            code.append(global_store_b16(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 4:
            code.append(global_store_b32(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 8:
            code.append(global_store_b64(addr=addr, data=val, saddr=buf_reg))
          else:
            code.append(global_store_b128(addr=addr, data=val, saddr=buf_reg))

          # Restore exec mask if we masked it
          if cond_uop is not None:
            code.append(s_mov_b32(EXEC_LO, get_exec_save_load()))

      elif u.op is Ops.RANGE:
        loop_var = ra.alloc_vgpr(u)
        r[u] = loop_var
        code.append(v_mov_b32_e32(loop_var, -1))  # Start at -1
        code.append(f"s_branch .L_END_{i}")  # Jump to loop end check
        code.append(f".L_BODY_{i}:")  # Loop body label

      elif u.op is Ops.END:
        if len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
          range_uop = u.src[1]
          range_idx = uops.index(range_uop)
          loop_var = r[range_uop]
          bound = get_reg(range_uop.src[0])
          code.append(f".L_END_{range_idx}:")  # Loop end label
          code.append(v_add_nc_u32_e32(loop_var, 1, loop_var))  # VOP2: src0=constant, vsrc1=VGPR
          # VOPC: src0 can be constant/SGPR, vsrc1 must be VGPR
          # We need loop_var < bound. Use bound > loop_var since loop_var is VGPR (vsrc1)
          code.append(v_cmp_gt_i32_e32(bound, loop_var))  # bound > loop_var ⇔ loop_var < bound
          code.append(f"s_cbranch_vccnz .L_BODY_{range_idx}")  # Branch back to loop body

      elif u.op is Ops.BARRIER:
        code.append(s_barrier())

      elif u.op is Ops.AFTER:
        # AFTER ensures previous operations complete, then returns the buffer
        # src[0] is the buffer, src[1] is the operation that must complete
        buf_uop = u.src[0]
        r[u] = get_reg(buf_uop)

      elif u.op is Ops.VECTORIZE:
        # Allocate contiguous registers for vector - pack small types
        count = len(u.src)
        scalar_dtype = u.dtype.scalar()
        if scalar_dtype.itemsize == 2:  # float16, int16, etc. - 2 elements per VGPR
          num_regs = (count + 1) // 2
          dst_range = ra.alloc_vgpr_range(u, num_regs)
          for j, src in enumerate(u.src):
            src_reg = get_reg(src)
            reg_idx = dst_range.idx + j // 2
            if j % 2 == 0:  # Low 16 bits - mask to clear upper bits (src may have garbage there)
              code.append(v_and_b32_e32(v[reg_idx], 0xFFFF, src_reg))
            else:  # High 16 bits - shift to upper 16 and OR
              tmp = ra.alloc_vgpr(u)
              code.append(v_lshlrev_b32_e32(tmp, 16, src_reg))
              code.append(v_or_b32_e32(v[reg_idx], v[reg_idx], tmp))
          r[u] = dst_range
        elif scalar_dtype.itemsize == 1:  # int8, uint8 - 4 elements per VGPR
          num_regs = (count + 3) // 4
          dst_range = ra.alloc_vgpr_range(u, num_regs)
          for j, src in enumerate(u.src):
            src_reg = get_reg(src)
            reg_idx = dst_range.idx + j // 4
            byte_idx = j % 4
            if byte_idx == 0:  # Low byte - mask to clear upper bits
              code.append(v_and_b32_e32(v[reg_idx], 0xFF, src_reg))
            else:  # Shift and OR into position
              tmp = ra.alloc_vgpr(u)
              code.append(v_and_b32_e32(tmp, 0xFF, src_reg))  # Mask first
              code.append(v_lshlrev_b32_e32(tmp, byte_idx * 8, tmp))
              code.append(v_or_b32_e32(v[reg_idx], v[reg_idx], tmp))
          r[u] = dst_range
        else:  # 32-bit or larger - one or more VGPRs per element
          dst_range = ra.alloc_vgpr_range(u, count)
          for j, src in enumerate(u.src):
            src_reg = get_reg(src)
            code.append(v_mov_b32_e32(v[dst_range.idx + j], src_reg))
          r[u] = dst_range

      elif u.op is Ops.GEP:
        maybe_wait(u.src)  # Wait for source load to complete before extraction
        vec_reg = get_reg(u.src[0])
        idx = u.arg[0] if isinstance(u.arg, tuple) else u.arg
        src_dtype = u.src[0].dtype
        if isinstance(vec_reg, VGPR):
          # Handle packed data types (multiple elements per VGPR)
          if src_dtype.scalar().itemsize == 2:  # float16, bfloat16, int16, uint16
            # Two elements per 32-bit VGPR: element i is in VGPR[i//2], bits [15:0] if i%2==0, [31:16] if i%2==1
            reg_idx = vec_reg.idx + idx // 2
            if idx % 2 == 1:  # Need high 16 bits - shift and extract
              dst = ra.alloc_vgpr(u)
              code.append(v_lshrrev_b32_e32(dst, 16, v[reg_idx]))
              r[u] = dst
            else:  # Low 16 bits - can use directly (cvt instructions use low bits)
              r[u] = v[reg_idx]
          elif src_dtype.scalar().itemsize == 1:  # int8, uint8
            # Four elements per 32-bit VGPR
            reg_idx = vec_reg.idx + idx // 4
            byte_idx = idx % 4
            if byte_idx == 0:
              r[u] = v[reg_idx]  # Low byte, can use directly
            else:
              dst = ra.alloc_vgpr(u)
              code.append(v_lshrrev_b32_e32(dst, byte_idx * 8, v[reg_idx]))
              r[u] = dst
          else:  # 32-bit or 64-bit types - one or more VGPRs per element
            r[u] = v[vec_reg.idx + idx]
        else:
          r[u] = vec_reg

      elif u.op is Ops.GROUP:
        # GROUP is handled at a higher level; just skip (NOOP)
        pass

      elif u.op is Ops.IF:
        # Save exec and mask with condition
        cond = get_reg(u.src[0])
        code.append(v_cmp_ne_i32_e32(0, cond))  # condition != 0
        code.append(s_and_saveexec_b32(get_exec_save_if(), VCC_LO))  # Save exec, AND with condition
        code.append(f"s_cbranch_execz .L_ENDIF_{i}")  # Skip if all lanes masked

      elif u.op is Ops.ENDIF:
        # Restore exec mask
        if_uop = u.src[0]
        if_idx = uops.index(if_uop)
        code.append(f".L_ENDIF_{if_idx}:")
        code.append(s_mov_b32(EXEC_LO, get_exec_save_if()))  # exec_lo = saved

    # Emit kernel prologue (load kernargs)
    prologue: list[Inst] = []
    for buf in bufs:
      offset = kernarg_offset[buf]
      dst = r[buf]
      prologue.append(s_load_b64(dst, kernarg_ptr, offset))
    for var in vars_:
      offset = kernarg_offset[var]
      dst = r[var]
      prologue.append(s_load_b32(dst, kernarg_ptr, offset))
    prologue.append(s_waitcnt(waitcnt(lgkmcnt=0)))

    # Emit epilogue - MSG_DEALLOC_VGPRS = 3
    epilogue: list[Inst] = [
      s_waitcnt(waitcnt(vmcnt=0, lgkmcnt=0)),
      s_sendmsg(simm16=3),  # sendmsg(MSG_DEALLOC_VGPRS)
      s_endpgm(),
    ]

    # Combine and convert to text
    all_code = prologue + code + epilogue
    asm_lines = [item if isinstance(item, str) else item.disasm() for item in all_code]

    # Generate kernel header
    v_cnt, s_cnt = ra.max_vgpr, ra.max_sgpr
    header = f""".text
.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"
.global kernel
.type kernel,@function
.p2align 8
kernel:
"""
    body = "\n".join(asm_lines)
    footer = self._render_metadata("kernel", bufs, vars_, v_cnt, s_cnt, lds_size)
    return header + body + "\n" + footer

  def _render_metadata(self, name: str, bufs: list[UOp], vars_: list[UOp], v_cnt: int, s_cnt: int, lds_size: int) -> str:
    """Generate AMDHSA kernel descriptor and metadata."""
    import yaml
    kernargs = []
    offset = 0
    for buf in bufs:
      kernargs.append({".name": f"buf{len(kernargs)}", ".offset": offset, ".size": 8, ".value_kind": "global_buffer"})
      offset += 8
    for i, var in enumerate(vars_):
      kernargs.append({".name": f"var{i}", ".offset": offset, ".size": 4, ".value_kind": "by_value"})
      offset += 4

    metadata = {
      "amdhsa.kernels": [{
        ".name": name,
        ".symbol": f"{name}.kd",
        ".kernarg_segment_size": offset,
        ".group_segment_fixed_size": lds_size,
        ".private_segment_fixed_size": 0,
        ".kernarg_segment_align": 8,
        ".wavefront_size": 32,
        ".sgpr_count": s_cnt + 6,
        ".vgpr_count": v_cnt,
        ".max_flat_workgroup_size": 1024,
        ".args": kernargs,
      }],
      "amdhsa.version": [1, 2],
    }
    metadata_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False)

    return f"""
.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size {lds_size}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size {offset}
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_next_free_vgpr {v_cnt}
  .amdhsa_next_free_sgpr {s_cnt + 6}
  .amdhsa_wavefront_size32 1
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_system_vgpr_workitem_id 2
.end_amdhsa_kernel

.amdgpu_metadata
{metadata_yaml}.end_amdgpu_metadata
"""
