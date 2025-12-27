# RDNA3 Renderer - uses assembly DSL and RDNARegAlloc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace, Invalid
from tinygrad.renderer import Renderer
from tinygrad.renderer.rdna_regalloc import RDNARegAlloc
from tinygrad.renderer.rdna_uops import rdna_matcher
from tinygrad.renderer.cstyle import create_non_native_float_pats, cast_float_to_bf16
from tinygrad.codegen.opt import tc
from extra.assembly.rdna3.lib import Inst
from extra.assembly.rdna3.asm import waitcnt
from extra.assembly.rdna3.autogen import (
  v, s, VGPR, SGPR, VCC_LO, EXEC_LO, NULL,
  # VOP1
  v_mov_b32_e32, v_cvt_f32_i32_e32, v_cvt_i32_f32_e32, v_cvt_f32_u32_e32, v_cvt_u32_f32_e32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_rcp_f32_e32, v_rcp_f64_e32, v_sqrt_f32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_trunc_f32_e32, v_sin_f32_e32, v_fract_f32_e32,
  v_cvt_f64_f32_e32, v_cvt_f32_f64_e32, v_cvt_f64_i32_e32, v_cvt_f64_u32_e32,
  v_cvt_i32_f64_e32, v_cvt_u32_f64_e32, v_trunc_f64_e32, v_floor_f64_e32,
  # VOP2
  v_add_f32_e32, v_sub_f32_e32, v_mul_f32_e32, v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  v_add_nc_u32_e32, v_sub_nc_u32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_ashrrev_i32_e32,
  v_max_f32_e32, v_max_i32_e32, v_max_u32_e32,
  # VOP3
  v_fma_f32, v_fma_f64, v_mad_u64_u32, v_mad_i64_i32, v_lshlrev_b64, v_lshrrev_b64, v_ashrrev_i64,
  v_mul_lo_u32, v_mul_hi_u32, v_bfe_u32, v_bfe_i32,
  v_add_co_u32, v_add_co_ci_u32_e32, v_cndmask_b32_e64, v_add_f64, v_mul_f64, v_sub_co_u32, v_sub_co_ci_u32_e32,
  v_cmp_lt_f32_e32, v_cmp_eq_f32_e32, v_cmp_neq_f32_e32, v_cmp_gt_f32_e32,
  v_cmp_lt_f64_e32, v_cmp_eq_f64_e32, v_cmp_neq_f64_e32, v_cmp_gt_f64_e32,
  v_cmp_lt_i32_e32, v_cmp_eq_i32_e32, v_cmp_ne_i32_e32, v_cmp_gt_i32_e32,
  v_cmp_lt_u32_e32, v_cmp_eq_u32_e32, v_cmp_ne_u32_e32, v_cmp_gt_u32_e32,
  # SOPP/SOP
  s_endpgm, s_waitcnt, s_barrier, s_sendmsg, s_mov_b32, s_and_saveexec_b32,
  # SMEM
  s_load_b32, s_load_b64,
  # FLAT/GLOBAL
  global_load_b32, global_load_b64, global_load_b128, global_load_u16, global_load_u8,
  global_store_b32, global_store_b64, global_store_b128, global_store_b16, global_store_b8,
  # DS (local memory)
  ds_load_b32, ds_load_b64, ds_load_b128, ds_store_b32, ds_store_b64, ds_store_b128,
  # WMMA (wave matrix multiply-accumulate)
  v_wmma_f32_16x16x16_f16, v_wmma_f32_16x16x16_bf16, v_wmma_f16_16x16x16_f16, v_wmma_bf16_16x16x16_bf16,
)

# Helper for VOP2: src0 can be constant/literal, vsrc1 must be VGPR - swap for commutative ops
def _sw(ctx, a, b):
  ar, br = ctx.get_reg(a), ctx.get_reg(b)
  return (br, ar) if isinstance(br, (int, float)) and not isinstance(ar, (int, float)) else (ar, br)

# Helper for 64-bit bitwise operations: apply op to both low and high 32-bit parts
def _bitwise64(ctx, a, b, op):
  ar, br = ctx.get_reg(a), ctx.get_reg(b)
  # Handle immediate constants: extract low and high 32-bit parts
  if isinstance(br, int):
    b_lo, b_hi = br & 0xFFFFFFFF, (br >> 32) & 0xFFFFFFFF
    return [op(ctx.dst, ar, b_lo), op(v[ctx.dst.idx+1], v[ar.idx+1] if isinstance(ar, VGPR) else 0, b_hi)]
  if isinstance(ar, int):
    a_lo, a_hi = ar & 0xFFFFFFFF, (ar >> 32) & 0xFFFFFFFF
    return [op(ctx.dst, a_lo, br), op(v[ctx.dst.idx+1], a_hi, v[br.idx+1])]
  # Both are VGPRs
  return [op(ctx.dst, ar, br), op(v[ctx.dst.idx+1], v[ar.idx+1], v[br.idx+1])]

# Helper for Newton-Raphson refined reciprocal: y' = y * (2 - x*y) for full precision
# Hardware rcp has ~1 ulp error which causes off-by-one issues with truncated division
def _refined_rcp_f32(ctx, x, a):
  ar = ctx.get_reg(a)
  dst = ctx.dst
  scratch = v[ctx.ra.get_scratch_vgpr()]
  # Newton-Raphson: y' = y * (2 - x*y) = y - y*(x*y - 1)
  # scratch = rcp(x)                    # y ≈ 1/x
  # dst = fma(x, y, -1) = x*y - 1       # error from 1
  # dst = y * dst = y * (x*y - 1)       # correction term
  # dst = y - dst = y * (2 - x*y)       # refined result
  return [v_rcp_f32_e32(scratch, ar), v_fma_f32(dst, ar, scratch, -1.0),
          v_mul_f32_e32(dst, scratch, dst), v_sub_f32_e32(dst, scratch, dst)]

# Module-level PatternMatcher for simple ALU and CAST operations
render_ops = PatternMatcher([
  # CAST: float32 <-> int32/uint32
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.int32),), name="x"), lambda ctx,x,a: [v_cvt_f32_i32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.uint32),), name="x"), lambda ctx,x,a: [v_cvt_f32_u32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.int32, (UPat.var("a", dtypes.float32),), name="x"), lambda ctx,x,a: [v_cvt_i32_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.uint32, (UPat.var("a", dtypes.float32),), name="x"), lambda ctx,x,a: [v_cvt_u32_f32_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: float32 <-> small int
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", (dtypes.uint8, dtypes.uint16)),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_u32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.int8),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 8), v_cvt_f32_i32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.int16),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 16), v_cvt_f32_i32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, (dtypes.uint8, dtypes.uint16), (UPat.var("a", dtypes.float32),), name="x"),
   lambda ctx,x,a: [v_cvt_u32_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, (dtypes.int8, dtypes.int16), (UPat.var("a", dtypes.float32),), name="x"),
   lambda ctx,x,a: [v_cvt_i32_f32_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: float16 <-> float32
  (UPat(Ops.CAST, dtypes.float16, (UPat.var("a", dtypes.float32),), name="x"), lambda ctx,x,a: [v_cvt_f16_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.float16),), name="x"), lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: float16 -> ints (via f32)
  (UPat(Ops.CAST, dtypes.int32, (UPat.var("a", dtypes.float16),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a)), v_cvt_i32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, dtypes.uint32, (UPat.var("a", dtypes.float16),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a)), v_cvt_u32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, (dtypes.int8, dtypes.int16), (UPat.var("a", dtypes.float16),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a)), v_cvt_i32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, (dtypes.uint8, dtypes.uint16), (UPat.var("a", dtypes.float16),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a)), v_cvt_u32_f32_e32(ctx.dst, ctx.dst)]),
  # CAST: ints -> float16 (via f32)
  (UPat(Ops.CAST, dtypes.float16, (UPat.var("a", dtypes.ints),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_i32_e32(ctx.dst, ctx.get_reg(a)), v_cvt_f16_f32_e32(ctx.dst, ctx.dst)]),
  # CAST: bfloat16 <-> float32 (shift)
  (UPat(Ops.CAST, dtypes.bfloat16, (UPat.var("a", dtypes.float32),), name="x"), lambda ctx,x,a: [v_lshrrev_b32_e32(ctx.dst, 16, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.bfloat16),), name="x"), lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a))]),
  # CAST: bfloat16 -> ints (via f32)
  (UPat(Ops.CAST, dtypes.int32, (UPat.var("a", dtypes.bfloat16),), name="x"),
   lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a)), v_cvt_i32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, dtypes.uint32, (UPat.var("a", dtypes.bfloat16),), name="x"),
   lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a)), v_cvt_u32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, (dtypes.int8, dtypes.int16), (UPat.var("a", dtypes.bfloat16),), name="x"),
   lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a)), v_cvt_i32_f32_e32(ctx.dst, ctx.dst)]),
  (UPat(Ops.CAST, (dtypes.uint8, dtypes.uint16), (UPat.var("a", dtypes.bfloat16),), name="x"),
   lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a)), v_cvt_u32_f32_e32(ctx.dst, ctx.dst)]),
  # CAST: ints -> bfloat16 (via f32)
  (UPat(Ops.CAST, dtypes.bfloat16, (UPat.var("a", dtypes.ints),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_i32_e32(ctx.dst, ctx.get_reg(a)), v_lshrrev_b32_e32(ctx.dst, 16, ctx.dst)]),
  # CAST: float64 <-> float32
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.float32),), name="x"), lambda ctx,x,a: [v_cvt_f64_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.float64),), name="x"), lambda ctx,x,a: [v_cvt_f32_f64_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: float64 <-> int32/uint32
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.int32),), name="x"), lambda ctx,x,a: [v_cvt_f64_i32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.uint32),), name="x"), lambda ctx,x,a: [v_cvt_f64_u32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.int32, (UPat.var("a", dtypes.float64),), name="x"), lambda ctx,x,a: [v_cvt_i32_f64_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.uint32, (UPat.var("a", dtypes.float64),), name="x"), lambda ctx,x,a: [v_cvt_u32_f64_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: float64 <-> small int
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", (dtypes.uint8, dtypes.uint16)),), name="x"),
   lambda ctx,x,a: [v_cvt_f64_u32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.int8),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(v[ctx.dst.idx], ctx.get_reg(a), 0, 8), v_cvt_f64_i32_e32(ctx.dst, v[ctx.dst.idx])]),
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.int16),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(v[ctx.dst.idx], ctx.get_reg(a), 0, 16), v_cvt_f64_i32_e32(ctx.dst, v[ctx.dst.idx])]),
  (UPat(Ops.CAST, (dtypes.uint8, dtypes.uint16), (UPat.var("a", dtypes.float64),), name="x"),
   lambda ctx,x,a: [v_cvt_u32_f64_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, (dtypes.int8, dtypes.int16), (UPat.var("a", dtypes.float64),), name="x"),
   lambda ctx,x,a: [v_cvt_i32_f64_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: int64 -> smaller types (just take low 32 bits)
  (UPat(Ops.CAST, dtypes.ints, (UPat.var("a", (dtypes.int64, dtypes.uint64)),), name="x"), lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: int64/uint64 -> float32 (via float64: low + high*2^32, 2^32=0x41F0000000000000)
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.int64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(6), [v_cvt_f64_u32_e32(v[s:s+2], ctx.get_reg(a)),
                    v_cvt_f64_i32_e32(v[s+2:s+4], v[ctx.get_reg(a).idx+1]),
                    v_mov_b32_e32(v[s+4], 0), v_mov_b32_e32(v[s+5], 0x41F00000),
                    v_mul_f64(v[s+2:s+4], v[s+4:s+6], v[s+2:s+4]),
                    v_add_f64(v[s:s+2], v[s:s+2], v[s+2:s+4]), v_cvt_f32_f64_e32(ctx.dst, v[s:s+2])])[1]),
  (UPat(Ops.CAST, dtypes.float32, (UPat.var("a", dtypes.uint64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(6), [v_cvt_f64_u32_e32(v[s:s+2], ctx.get_reg(a)),
                    v_cvt_f64_u32_e32(v[s+2:s+4], v[ctx.get_reg(a).idx+1]),
                    v_mov_b32_e32(v[s+4], 0), v_mov_b32_e32(v[s+5], 0x41F00000),
                    v_mul_f64(v[s+2:s+4], v[s+4:s+6], v[s+2:s+4]),
                    v_add_f64(v[s:s+2], v[s:s+2], v[s+2:s+4]), v_cvt_f32_f64_e32(ctx.dst, v[s:s+2])])[1]),
  # CAST: int64/uint64 -> float64 (low + high*2^32, 2^32=0x41F0000000000000)
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.int64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(4), [v_cvt_f64_u32_e32(ctx.dst, ctx.get_reg(a)),
                    v_cvt_f64_i32_e32(v[s:s+2], v[ctx.get_reg(a).idx+1]),
                    v_mov_b32_e32(v[s+2], 0), v_mov_b32_e32(v[s+3], 0x41F00000),
                    v_mul_f64(v[s:s+2], v[s+2:s+4], v[s:s+2]), v_add_f64(ctx.dst, ctx.dst, v[s:s+2])])[1]),
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.uint64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(4), [v_cvt_f64_u32_e32(ctx.dst, ctx.get_reg(a)),
                    v_cvt_f64_u32_e32(v[s:s+2], v[ctx.get_reg(a).idx+1]),
                    v_mov_b32_e32(v[s+2], 0), v_mov_b32_e32(v[s+3], 0x41F00000),
                    v_mul_f64(v[s:s+2], v[s+2:s+4], v[s:s+2]), v_add_f64(ctx.dst, ctx.dst, v[s:s+2])])[1]),
  # CAST: float64 -> int64 (trunc, extract high*2^32 and low parts)
  # s[0:2]=trunc, s[2:4]=high_float, s[4:6]=trunc_copy for final subtract
  (UPat(Ops.CAST, dtypes.int64, (UPat.var("a", dtypes.float64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(6), ar:=ctx.get_reg(a), [
     v_trunc_f64_e32(v[s:s+2], ar),  # Truncate to integer
     v_mov_b32_e32(v[s+4], v[s]), v_mov_b32_e32(v[s+5], v[s+1]),  # Save trunc copy
     v_mov_b32_e32(v[s+2], 0), v_mov_b32_e32(v[s+3], 0x3DF00000),  # 2^-32
     v_mul_f64(v[s+2:s+4], v[s:s+2], v[s+2:s+4]),  # high = trunc / 2^32
     v_floor_f64_e32(v[s+2:s+4], v[s+2:s+4]),  # floor(high)
     v_cvt_i32_f64_e32(v[ctx.dst.idx+1], v[s+2:s+4]),  # high 32 bits
     v_mov_b32_e32(v[s], 0), v_mov_b32_e32(v[s+1], 0x41F00000),  # 2^32
     v_mul_f64(v[s+2:s+4], v[s+2:s+4], v[s:s+2]),  # high * 2^32
     v_mul_f64(v[s+2:s+4], -1.0, v[s+2:s+4]),  # negate high*2^32
     v_add_f64(v[s:s+2], v[s+4:s+6], v[s+2:s+4]),  # trunc - high*2^32 = low
     v_cvt_u32_f64_e32(ctx.dst, v[s:s+2])])[2]),  # low 32 bits
  (UPat(Ops.CAST, dtypes.uint64, (UPat.var("a", dtypes.float64),), name="x"),
   lambda ctx,x,a: (s:=ctx.ra.get_scratch_vgpr(6), ar:=ctx.get_reg(a), [
     v_trunc_f64_e32(v[s:s+2], ar),  # Truncate to integer
     v_mov_b32_e32(v[s+4], v[s]), v_mov_b32_e32(v[s+5], v[s+1]),  # Save trunc copy
     v_mov_b32_e32(v[s+2], 0), v_mov_b32_e32(v[s+3], 0x3DF00000),  # 2^-32
     v_mul_f64(v[s+2:s+4], v[s:s+2], v[s+2:s+4]),  # high = trunc / 2^32
     v_floor_f64_e32(v[s+2:s+4], v[s+2:s+4]),  # floor(high)
     v_cvt_u32_f64_e32(v[ctx.dst.idx+1], v[s+2:s+4]),  # high 32 bits
     v_mov_b32_e32(v[s], 0), v_mov_b32_e32(v[s+1], 0x41F00000),  # 2^32
     v_mul_f64(v[s+2:s+4], v[s+2:s+4], v[s:s+2]),  # high * 2^32
     v_mul_f64(v[s+2:s+4], -1.0, v[s+2:s+4]),  # negate high*2^32
     v_add_f64(v[s:s+2], v[s+4:s+6], v[s+2:s+4]),  # trunc - high*2^32 = low
     v_cvt_u32_f64_e32(ctx.dst, v[s:s+2])])[2]),  # low 32 bits
  # CAST: int8 -> larger types (sign extend)
  (UPat(Ops.CAST, (dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32), (UPat.var("a", dtypes.int8),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 8)]),
  # CAST: int16 -> 32-bit types (sign extend)
  (UPat(Ops.CAST, (dtypes.int32, dtypes.uint32), (UPat.var("a", dtypes.int16),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 16)]),
  # CAST: small signed int -> int64 (sign extend to 32-bit, then sign extend to 64-bit)
  (UPat(Ops.CAST, dtypes.int64, (UPat.var("a", dtypes.int8),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 8), v_ashrrev_i32_e32(v[ctx.dst.idx+1], 31, ctx.dst)]),
  (UPat(Ops.CAST, dtypes.int64, (UPat.var("a", dtypes.int16),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 16), v_ashrrev_i32_e32(v[ctx.dst.idx+1], 31, ctx.dst)]),
  # CAST: small signed int -> uint64 (sign extend to 32-bit, then zero extend high bits for unsigned reinterpret)
  (UPat(Ops.CAST, dtypes.uint64, (UPat.var("a", dtypes.int8),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 8), v_ashrrev_i32_e32(v[ctx.dst.idx+1], 31, ctx.dst)]),
  (UPat(Ops.CAST, dtypes.uint64, (UPat.var("a", dtypes.int16),), name="x"),
   lambda ctx,x,a: [v_bfe_i32(ctx.dst, ctx.get_reg(a), 0, 16), v_ashrrev_i32_e32(v[ctx.dst.idx+1], 31, ctx.dst)]),
  # CAST: int32 -> int64 (sign extend: copy sign bit to high 32 bits)
  (UPat(Ops.CAST, (dtypes.int64, dtypes.uint64), (UPat.var("a", dtypes.int32),), name="x"),
   lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a)), v_ashrrev_i32_e32(v[ctx.dst.idx+1], 31, ctx.get_reg(a))]),
  # CAST: uint32 -> int64/uint64 (zero extend: set high 32 bits to 0)
  (UPat(Ops.CAST, (dtypes.int64, dtypes.uint64), (UPat.var("a", dtypes.uint32),), name="x"),
   lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a)), v_mov_b32_e32(v[ctx.dst.idx+1], 0)]),
  # CAST: small unsigned int -> int64/uint64 (zero extend)
  (UPat(Ops.CAST, (dtypes.int64, dtypes.uint64), (UPat.var("a", (dtypes.uint8, dtypes.uint16)),), name="x"),
   lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a)), v_mov_b32_e32(v[ctx.dst.idx+1], 0)]),
  # CAST: small int <-> int32/uint32, small int <-> small int (just mov for unsigned or same-size)
  (UPat(Ops.CAST, dtypes.ints, (UPat.var("a", dtypes.ints),), name="x"), lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a))]),
  # CAST: bool <-> int/float
  (UPat(Ops.CAST, (dtypes.int64, dtypes.uint64), (UPat.var("a", dtypes.bool),), name="x"),
   lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a)), v_mov_b32_e32(v[ctx.dst.idx+1], 0)]),
  (UPat(Ops.CAST, dtypes.ints, (UPat.var("a", dtypes.bool),), name="x"), lambda ctx,x,a: [v_mov_b32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.CAST, dtypes.bool, (UPat.var("a", dtypes.ints),), name="x"),
   lambda ctx,x,a: [v_cmp_ne_i32_e32(0, ctx.get_reg(a)), v_cndmask_b32_e64(ctx.dst, 0, 1, VCC_LO)]),
  (UPat(Ops.CAST, dtypes.bool, (UPat.var("a", dtypes.float32),), name="x"),
   lambda ctx,x,a: [v_cmp_neq_f32_e32(0.0, ctx.get_reg(a)), v_cndmask_b32_e64(ctx.dst, 0, 1, VCC_LO)]),
  (UPat(Ops.CAST, dtypes.bool, (UPat.var("a", dtypes.float16),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f16_e32(ctx.dst, ctx.get_reg(a)), v_cmp_neq_f32_e32(0.0, ctx.dst), v_cndmask_b32_e64(ctx.dst, 0, 1, VCC_LO)]),
  (UPat(Ops.CAST, dtypes.bool, (UPat.var("a", dtypes.float64),), name="x"),
   lambda ctx,x,a: [v_cvt_f32_f64_e32(ctx.dst, ctx.get_reg(a)), v_cmp_neq_f32_e32(0.0, ctx.dst), v_cndmask_b32_e64(ctx.dst, 0, 1, VCC_LO)]),
  (UPat(Ops.CAST, dtypes.bool, (UPat.var("a", dtypes.bfloat16),), name="x"),
   lambda ctx,x,a: [v_lshlrev_b32_e32(ctx.dst, 16, ctx.get_reg(a)), v_cmp_neq_f32_e32(0.0, ctx.dst), v_cndmask_b32_e64(ctx.dst, 0, 1, VCC_LO)]),
  (UPat(Ops.CAST, dtypes.float64, (UPat.var("a", dtypes.bool),), name="x"),
   lambda ctx,x,a: [v_cvt_f64_u32_e32(ctx.dst, ctx.get_reg(a))]),  # bool -> float64
  (UPat(Ops.CAST, dtypes.floats, (UPat.var("a", dtypes.bool),), name="x"), lambda ctx,x,a: [v_cvt_f32_u32_e32(ctx.dst, ctx.get_reg(a))]),
  # ADD: float64, floats, int64, default to i32
  (UPat(Ops.ADD, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_add_f64(ctx.dst, ctx.get_reg(a), ctx.get_reg(b))]),
  (UPat(Ops.ADD, dtype=dtypes.floats, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_add_f32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.ADD, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_add_co_u32(v[ctx.dst.idx], VCC_LO, v[ctx.get_reg(a).idx], v[ctx.get_reg(b).idx]),
                    v_add_co_ci_u32_e32(v[ctx.dst.idx+1], v[ctx.get_reg(a).idx+1], v[ctx.get_reg(b).idx+1])]),
  (UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_add_nc_u32_e32(ctx.dst, *_sw(ctx,a,b))]),
  # SUB: float64, floats, int64, default to i32
  (UPat(Ops.SUB, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_mul_f64(ctx.dst, -1.0, ctx.get_reg(b)), v_add_f64(ctx.dst, ctx.get_reg(a), ctx.dst)]),
  (UPat(Ops.SUB, dtype=dtypes.floats, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_sub_f32_e32(ctx.dst, ctx.get_reg(a), ctx.get_reg(b))]),
  (UPat(Ops.SUB, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_sub_co_u32(v[ctx.dst.idx], VCC_LO, v[ctx.get_reg(a).idx], v[ctx.get_reg(b).idx]),
                    v_sub_co_ci_u32_e32(v[ctx.dst.idx+1], v[ctx.get_reg(a).idx+1], v[ctx.get_reg(b).idx+1])]),
  (UPat(Ops.SUB, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_sub_nc_u32_e32(ctx.dst, ctx.get_reg(a), ctx.get_reg(b))]),
  # MUL: floats, ints (int64 complex - handled in emit_alu)
  (UPat(Ops.MUL, dtype=dtypes.floats, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_mul_f32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.MUL, dtype=(dtypes.int32, dtypes.uint32, dtypes.int16, dtypes.uint16, dtypes.int8, dtypes.uint8),
   src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_mul_lo_u32(ctx.dst, *_sw(ctx,a,b))]),
  # Bitwise: 64-bit (need to operate on both low and high 32-bit parts)
  (UPat(Ops.AND, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: _bitwise64(ctx, a, b, v_and_b32_e32)),
  (UPat(Ops.OR, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: _bitwise64(ctx, a, b, v_or_b32_e32)),
  (UPat(Ops.XOR, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: _bitwise64(ctx, a, b, v_xor_b32_e32)),
  # Bitwise: 32-bit and smaller ints, and bool
  (UPat(Ops.AND, dtype=(dtypes.int32, dtypes.uint32, dtypes.int16, dtypes.uint16, dtypes.int8, dtypes.uint8, dtypes.bool),
   src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_and_b32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.OR, dtype=(dtypes.int32, dtypes.uint32, dtypes.int16, dtypes.uint16, dtypes.int8, dtypes.uint8, dtypes.bool),
   src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_or_b32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.XOR, dtype=(dtypes.int32, dtypes.uint32, dtypes.int16, dtypes.uint16, dtypes.int8, dtypes.uint8, dtypes.bool),
   src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_xor_b32_e32(ctx.dst, *_sw(ctx,a,b))]),
  # SHL: int64, default to i32
  (UPat(Ops.SHL, dtype=(dtypes.int64, dtypes.uint64), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_lshlrev_b64(ctx.dst, ctx.get_reg(b), ctx.get_reg(a))]),
  (UPat(Ops.SHL, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_lshlrev_b32_e32(ctx.dst, ctx.get_reg(b), ctx.get_reg(a))]),
  # SHR: int64 signed, uint64 unsigned, default to 32-bit
  (UPat(Ops.SHR, dtype=dtypes.int64, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_ashrrev_i64(ctx.dst, ctx.get_reg(b), ctx.get_reg(a))]),
  (UPat(Ops.SHR, dtype=dtypes.uint64, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda ctx,x,a,b: [v_lshrrev_b64(ctx.dst, ctx.get_reg(b), ctx.get_reg(a))]),
  # MAX: floats, signed ints, unsigned ints, bool (bool uses OR since max(True, False) = True)
  (UPat(Ops.MAX, dtype=dtypes.floats, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_max_f32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.MAX, dtype=dtypes.sints, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_max_i32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.MAX, dtype=dtypes.uints, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_max_u32_e32(ctx.dst, *_sw(ctx,a,b))]),
  (UPat(Ops.MAX, dtype=dtypes.bool, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b: [v_or_b32_e32(ctx.dst, *_sw(ctx,a,b))]),
  # MULACC (FMA): float64, floats
  (UPat(Ops.MULACC, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b"), UPat.var("d")), name="x"),
   lambda ctx,x,a,b,d: [v_fma_f64(ctx.dst, ctx.get_reg(a), ctx.get_reg(b), ctx.get_reg(d))]),
  (UPat(Ops.MULACC, dtype=dtypes.floats, src=(UPat.var("a"), UPat.var("b"), UPat.var("d")), name="x"),
   lambda ctx,x,a,b,d: [v_fma_f32(ctx.dst, ctx.get_reg(a), ctx.get_reg(b), ctx.get_reg(d))]),
  # Transcendental: float64 first (for precision), then float32 with Newton-Raphson refinement
  (UPat(Ops.RECIPROCAL, dtype=dtypes.float64, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_rcp_f64_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.RECIPROCAL, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), _refined_rcp_f32),
  (UPat(Ops.SQRT, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_sqrt_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.EXP2, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_exp_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.LOG2, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_log_f32_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.TRUNC, dtype=dtypes.float64, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_trunc_f64_e32(ctx.dst, ctx.get_reg(a))]),
  (UPat(Ops.TRUNC, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_trunc_f32_e32(ctx.dst, ctx.get_reg(a))]),
  # NEG: floats vs ints
  (UPat(Ops.NEG, dtype=dtypes.floats, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_mul_f32_e32(ctx.dst, -1.0, ctx.get_reg(a))]),
  (UPat(Ops.NEG, dtype=dtypes.ints, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: [v_sub_nc_u32_e32(ctx.dst, 0, ctx.get_reg(a))]),
])


# Context class for PatternMatcher - provides access to registers and code emission
class RenderContext:
  def __init__(self, ra: RDNARegAlloc, r: dict, code: list, get_reg_fn):
    self.ra = ra
    self.r = r
    self.code = code
    self.get_reg = get_reg_fn
    self.dst: VGPR | None = None  # Set before each ALU op

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
  # NOTE: SIN removed - use software implementations for precision with large values
  code_for_op = {
    # Transcendental ops (SIN uses software for precision with large values, RECIPROCAL is needed by software SIN)
    Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SQRT: lambda: None, Ops.TRUNC: lambda: None, Ops.RECIPROCAL: lambda: None,
    # Bitwise ops
    Ops.AND: lambda: None, Ops.OR: lambda: None, Ops.XOR: lambda: None, Ops.SHL: lambda: None, Ops.SHR: lambda: None,
    # Arithmetic ops (IDIV/MOD handled directly for proper 64-bit support)
    Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.NEG: lambda: None,
    Ops.IDIV: lambda: None, Ops.MOD: lambda: None,
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
    r: dict[UOp, VGPR | SGPR | int | tuple] = {}  # UOp -> register mapping (RANGE stores (loop_var, bound) tuple)
    code: list[Inst | str] = []  # Generated instructions (str for labels like .L_BODY_N)
    bufs: list[UOp] = []
    vars_: list[UOp] = []  # Symbolic variables
    kernarg_offset: dict[UOp, int] = {}
    current_offset = 0
    lds_size = 0
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
        for src_ in src.src:
          if needs_wait(src_, visited): return True
        return False
      for src in srcs:
        if needs_wait(src):
          code.append(s_waitcnt(waitcnt(vmcnt=0, lgkmcnt=0)))
          pending_waits.clear()
          return

    # Fixed registers
    kernarg_ptr = s[0:2]  # s[0:1] holds kernarg pointer

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
        # For 64-bit types, always load into register pair (can't use inline constants for 64-bit ADD)
        if u.dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
          reg = ra.alloc_vgpr_range(u, 2)
          lo = int(val) & 0xFFFFFFFF
          hi = (int(val) >> 32) & 0xFFFFFFFF
          code.append(v_mov_b32_e32(v[reg.idx], lo))
          code.append(v_mov_b32_e32(v[reg.idx + 1], hi))
          r[u] = reg
          return reg
        # Float64 constants: load as two 32-bit parts (IEEE 754 double)
        if u.dtype == dtypes.float64:
          reg = ra.alloc_vgpr_range(u, 2)
          bits64 = struct.unpack("Q", struct.pack("d", float(val)))[0]
          lo = bits64 & 0xFFFFFFFF
          hi = (bits64 >> 32) & 0xFFFFFFFF
          code.append(v_mov_b32_e32(v[reg.idx], lo))
          code.append(v_mov_b32_e32(v[reg.idx + 1], hi))
          r[u] = reg
          return reg
        if isinstance(val, float):
          if val in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0): return val
          # Convert inf/nan to hex representation
          if math.isinf(val) or math.isnan(val):
            val = struct.unpack("I", struct.pack("f", val))[0]
        elif isinstance(val, int) and -16 <= val <= 64: return val
        # Load literal constant into register
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
        (Ops.CMPLT, dtypes.float64): v_cmp_lt_f64_e32, (Ops.CMPEQ, dtypes.float64): v_cmp_eq_f64_e32, (Ops.CMPNE, dtypes.float64): v_cmp_neq_f64_e32,
      }
      # GT versions for swapping CMPLT: a < b ⇔ b > a
      cmp_gt_map = {
        (Ops.CMPLT, dtypes.float32): v_cmp_gt_f32_e32, (Ops.CMPLT, dtypes.int32): v_cmp_gt_i32_e32, (Ops.CMPLT, dtypes.uint32): v_cmp_gt_u32_e32,
        (Ops.CMPLT, dtypes.float64): v_cmp_gt_f64_e32,
      }
      base_dtype = dtypes.float64 if dtype == dtypes.float64 else dtypes.float32 if dtypes.is_float(dtype) else \
                   dtypes.int32 if dtype in (dtypes.int8, dtypes.int16, dtypes.int32) else dtypes.uint32
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
      """Emit ALU instruction for complex ops not handled by render_ops PatternMatcher."""
      op, dtype = u.op, u.dtype
      srcs = [get_reg(s) for s in u.src]
      a, b = (srcs[0], srcs[1]) if len(srcs) >= 2 else (srcs[0], 0)

      if op is Ops.MUL and dtype in (dtypes.int64, dtypes.uint64):
        # 64-bit multiply: handle fast_idiv pattern (signed 32-bit cast * large constant)
        a_uop, b_uop = u.src[0], u.src[1]
        a_is_signed_cast = a_uop.op is Ops.CAST and a_uop.src[0].dtype == dtypes.int32
        b_is_const_hibit = b_uop.op is Ops.CONST and isinstance(b_uop.arg, int) and (b_uop.arg & 0x80000000) != 0
        a_reg, b_reg = (a if isinstance(a, (int, float)) else v[a.idx]), (b if isinstance(b, (int, float)) else v[b.idx])
        if dtype == dtypes.int64 and a_is_signed_cast and b_is_const_hibit:
          # fast_idiv: unsigned multiply then correct for sign (when a < 0, high bits off by b)
          b_lo, scratch, a_src_reg = b_uop.arg & 0xFFFFFFFF, ra.alloc_vgpr(u), get_reg(a_uop.src[0])
          code.extend([v_mul_lo_u32(v[dst.idx], a_reg, b_lo), v_mul_hi_u32(v[dst.idx + 1], a_reg, b_lo),
                       v_cmp_gt_i32_e32(0, a_src_reg), v_cndmask_b32_e64(scratch, 0, b_lo, VCC_LO),
                       v_sub_nc_u32_e32(v[dst.idx + 1], v[dst.idx + 1], scratch)])
        elif dtype == dtypes.int64: code.append(v_mad_i64_i32(dst, NULL, a_reg, b_reg, 0))
        else: code.append(v_mad_u64_u32(dst, NULL, a_reg, b_reg, 0))
      elif op is Ops.SHR:
        src_dtype = u.src[0].dtype
        is_unsigned = src_dtype in (dtypes.uint64, dtypes.uint32, dtypes.uint16, dtypes.uint8)
        if src_dtype in (dtypes.int64, dtypes.uint64) and isinstance(b, int) and b >= 32:
          # 64-bit shift >= 32: result = high_reg >> (shift - 32)
          src_reg = r[u.src[0]]
          high_reg = v[(src_reg.idx if isinstance(src_reg, VGPR) else src_reg) + 1]
          code.append(v_lshrrev_b32_e32(dst, b - 32, high_reg) if is_unsigned else v_ashrrev_i32_e32(dst, b - 32, high_reg))
        else:
          code.append(v_lshrrev_b32_e32(dst, b, a) if is_unsigned else v_ashrrev_i32_e32(dst, b, a))
      elif op in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE):
        emit_cmp(op, u.src[0].dtype, dst, a, b)
      elif op is Ops.WHERE:
        cond, true_val, false_val = srcs[0], srcs[1], srcs[2]
        code.append(v_cmp_ne_i32_e32(0, cond))
        if dtype == dtypes.float64:
          # For float64: select both low and high 32-bit parts
          code.append(v_cndmask_b32_e64(v[dst.idx], v[false_val.idx], v[true_val.idx], VCC_LO))
          code.append(v_cndmask_b32_e64(v[dst.idx+1], v[false_val.idx+1], v[true_val.idx+1], VCC_LO))
        else:
          code.append(v_cndmask_b32_e64(dst, false_val, true_val, VCC_LO))
      elif op is Ops.IDIV:
        # Integer division using floating-point approximation
        # quotient = trunc(float(a) * rcp(float(b)))
        if dtype in (dtypes.int64, dtypes.uint64):
          # 64-bit division via float64 (53 bits precision, sufficient for most cases)
          # For values > 53 bits, may have off-by-one errors but acceptable for Payne-Hanek
          s = ra.get_scratch_vgpr(8)
          a_reg, b_reg = (a if isinstance(a, VGPR) else v[a.idx] if hasattr(a, 'idx') else a,
                          b if isinstance(b, VGPR) else v[b.idx] if hasattr(b, 'idx') else b)
          # Convert a to float64: low + high*2^32
          code.append(v_cvt_f64_u32_e32(v[s:s+2], a_reg))  # low part
          a_hi = v[a_reg.idx+1] if isinstance(a_reg, VGPR) else 0
          code.append(v_cvt_f64_u32_e32(v[s+2:s+4], a_hi))  # high part
          code.append(v_mov_b32_e32(v[s+4], 0))
          code.append(v_mov_b32_e32(v[s+5], 0x41F00000))  # 2^32 in float64
          code.append(v_mul_f64(v[s+2:s+4], v[s+4:s+6], v[s+2:s+4]))  # high * 2^32
          code.append(v_add_f64(v[s:s+2], v[s:s+2], v[s+2:s+4]))  # a as float64
          # Convert b to float64: low + high*2^32
          code.append(v_cvt_f64_u32_e32(v[s+2:s+4], b_reg))  # low part
          b_hi = v[b_reg.idx+1] if isinstance(b_reg, VGPR) else 0
          code.append(v_cvt_f64_u32_e32(v[s+4:s+6], b_hi))  # high part
          code.append(v_mov_b32_e32(v[s+6], 0))
          code.append(v_mov_b32_e32(v[s+7], 0x41F00000))  # 2^32 in float64
          code.append(v_mul_f64(v[s+4:s+6], v[s+6:s+8], v[s+4:s+6]))  # high * 2^32
          code.append(v_add_f64(v[s+2:s+4], v[s+2:s+4], v[s+4:s+6]))  # b as float64
          # Compute a/b via reciprocal
          code.append(v_rcp_f64_e32(v[s+4:s+6], v[s+2:s+4]))  # 1/b
          code.append(v_mul_f64(v[s:s+2], v[s:s+2], v[s+4:s+6]))  # a/b
          code.append(v_trunc_f64_e32(v[s:s+2], v[s:s+2]))  # floor(a/b)
          # Convert back to uint64: for most cases, result fits in low 32 bits, high is 0
          code.append(v_cvt_u32_f64_e32(v[dst.idx], v[s:s+2]))  # low part
          # For high part: (result - low) / 2^32
          code.append(v_cvt_f64_u32_e32(v[s+2:s+4], v[dst.idx]))  # low as float64
          code.append(v_mul_f64(v[s+2:s+4], -1.0, v[s+2:s+4]))  # -low
          code.append(v_add_f64(v[s:s+2], v[s:s+2], v[s+2:s+4]))  # result - low
          code.append(v_mov_b32_e32(v[s+4], 0))
          code.append(v_mov_b32_e32(v[s+5], 0x3DF00000))  # 2^-32 in float64
          code.append(v_mul_f64(v[s:s+2], v[s:s+2], v[s+4:s+6]))  # (result - low) * 2^-32
          code.append(v_cvt_u32_f64_e32(v[dst.idx+1], v[s:s+2]))  # high part
        # For signed: handle signs, do unsigned div, restore sign
        elif dtype in (dtypes.int32, dtypes.int16, dtypes.int8):
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

      if u.op is Ops.DEFINE_GLOBAL:
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
                    Ops.TRUNC, Ops.NEG, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.WHERE,
                    Ops.IDIV, Ops.MOD):
        maybe_wait(u.src)  # Wait for any pending loads used by this operation
        dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(u.dtype) else ra.alloc_vgpr(u)
        r[u] = dst
        ctx = RenderContext(ra, r, code, get_reg)
        ctx.dst = dst
        if (insts := render_ops.rewrite(u, ctx=ctx)) is not None: code.extend(insts)
        else: emit_alu(u, dst)

      elif u.op is Ops.CAST:
        maybe_wait(u.src)
        src_dtype, dst_dtype = u.src[0].dtype, u.dtype
        if src_dtype == dst_dtype:
          r[u] = get_reg(u.src[0])
        else:
          dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(dst_dtype) else ra.alloc_vgpr(u)
          r[u] = dst
          ctx = RenderContext(ra, r, code, get_reg)
          ctx.dst = dst
          insts = render_ops.rewrite(u, ctx=ctx)
          assert insts is not None, f"unhandled cast: {src_dtype} -> {dst_dtype}"
          code.extend(insts)

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
                  dst_reg = v[dst.idx + j] if hasattr(dst, 'idx') else dst
                  src_reg = v[default_val.idx + j] if hasattr(default_val, 'idx') else default_val
                  code.append(v_mov_b32_e32(dst_reg, src_reg))
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
            if itemsize == 8:
              # 64-bit: move both low and high 32-bit parts
              code.append(v_mov_b32_e32(v[buf_reg.idx + reg_offset * 2], v[val.idx] if isinstance(val, VGPR) else val))
              code.append(v_mov_b32_e32(v[buf_reg.idx + reg_offset * 2 + 1], v[val.idx + 1] if isinstance(val, VGPR) else 0))
            else:
              code.append(v_mov_b32_e32(v[buf_reg.idx + reg_offset], val))
          else:
            # Variable offset - use first register as fallback (TODO: proper indirect)
            if itemsize == 8:
              code.append(v_mov_b32_e32(v[buf_reg.idx], v[val.idx] if isinstance(val, VGPR) else val))
              code.append(v_mov_b32_e32(v[buf_reg.idx + 1], v[val.idx + 1] if isinstance(val, VGPR) else 0))
            else:
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
        # IMPORTANT: Materialize the bound BEFORE the loop starts, not lazily at END time
        # Otherwise, if bound > 64, it gets allocated inside the loop body and the first
        # comparison uses uninitialized register (v38=0 from earlier init)
        bound = get_reg(u.src[0])
        r[u] = (loop_var, bound)  # Store both loop var and bound
        code.append(v_mov_b32_e32(loop_var, -1))  # Start at -1
        code.append(f"s_branch .L_END_{i}")  # Jump to loop end check
        code.append(f".L_BODY_{i}:")  # Loop body label

      elif u.op is Ops.END:
        if len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
          range_uop = u.src[1]
          range_idx = uops.index(range_uop)
          loop_var, bound = r[range_uop]  # Get both loop var and pre-materialized bound
          code.append(f".L_END_{range_idx}:")  # Loop end label
          code.append(v_add_nc_u32_e32(loop_var, 1, loop_var))  # VOP2: src0=constant, vsrc1=VGPR
          # VOPC: src0 can be constant/SGPR, vsrc1 must be VGPR
          # We need loop_var < bound. Use bound > loop_var since loop_var is VGPR (vsrc1)
          code.append(v_cmp_gt_i32_e32(bound, loop_var))  # bound > loop_var ⇔ loop_var < bound
          code.append(f"s_cbranch_vccnz .L_BODY_{range_idx}")  # Branch back to loop body

      elif u.op is Ops.BARRIER:
        code.append(s_barrier())

      elif u.op is Ops.WMMA:
        maybe_wait(u.src)
        # WMMA: wave matrix multiply-accumulate
        # arg: (name, dims, dtype_in, dtype_out, device, threads, upcast_axes, reduce_axes)
        # src: (A_vec, B_vec, C_acc)
        dtype_in, dtype_out = u.arg[2], u.dtype.scalar()
        a_reg, b_reg, c_reg = get_reg(u.src[0]), get_reg(u.src[1]), get_reg(u.src[2])
        # Output is 8 floats (or 8 halves) = 8 VGPRs
        dst = ra.alloc_vgpr_range(u, 8)
        # Select the right WMMA instruction based on input/output types
        if dtype_in == dtypes.half and dtype_out == dtypes.float:
          code.append(v_wmma_f32_16x16x16_f16(dst, a_reg, b_reg, c_reg))
        elif dtype_in == dtypes.bfloat16 and dtype_out == dtypes.float:
          code.append(v_wmma_f32_16x16x16_bf16(dst, a_reg, b_reg, c_reg))
        elif dtype_in == dtypes.half and dtype_out == dtypes.half:
          code.append(v_wmma_f16_16x16x16_f16(dst, a_reg, b_reg, c_reg))
        elif dtype_in == dtypes.bfloat16 and dtype_out == dtypes.bfloat16:
          code.append(v_wmma_bf16_16x16x16_bf16(dst, a_reg, b_reg, c_reg))
        else:
          raise NotImplementedError(f"WMMA not implemented for {dtype_in} -> {dtype_out}")
        r[u] = dst

      elif u.op is Ops.AFTER:
        # AFTER ensures previous operations complete, then returns the buffer
        # src[0] is the buffer, src[1] is the operation that must complete
        buf_uop = u.src[0]
        r[u] = get_reg(buf_uop)

      elif u.op is Ops.VECTORIZE:
        maybe_wait(u.src)  # Wait for any pending loads before vectorizing
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
