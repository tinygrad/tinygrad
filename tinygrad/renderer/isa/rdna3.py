from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.helpers import Target
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

pre_isel_matcher = PatternMatcher([
])

"""
def imm(dt:DType, v:int) -> UOp: return UOp.const(dt, truncate[dt](v)).rtag()
def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  def _disp(v:int) -> UOp: return imm(dtypes.int32 if abs(v) > dtypes.int8.max else dtypes.int8, v)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), _disp(0))
  base, idx = x.src
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (base, _cast(idx.src[0]), _disp(idx.src[1].arg * disp_scale))
  if idx.op is Ops.CONST: return (base, UOp(Ops.NOOP), _disp(idx.arg * disp_scale))
  return (base, _cast(idx), _disp(0))
"""

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2 and dt not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4 and dt not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8 and dt not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]: # (offset, base, val)
    pass
def handle_memspace(x:UOp, lins, gins):
    if x.src[0].addrspace == AddrSpace.LOCAL: return lins
    if x.src[0].addrspace == AddrSpace.GLOBAL: return gins
    return None

isel_matcher = PatternMatcher([
    # SALU Ops

    # --- VALU Ops ---
    # NOTE: VALU-only baseline. SALU (s_*) is an optimization for wave-uniform values; there's no uniformity
    #       analysis yet, so emitting per-lane VALU is always correct. Don't gate on dtype.count -- that's
    #       vector *width* within a lane (packed math, v_pk_*), not the SALU/VALU axis.
    # TODO: when to use 64 vs 32 bit embedding? e32/e64
    # - when we need to pass extra info?
    # - check spec
    # TODO: NOOP is an identity/copy that must forward its source value -- lower to a register copy, not v_nop.
    #       v_nop produces no result and would drop the value. Needs copy() infra first.
    # (UPat(Ops.NOOP, name="x"), lambda x: ...),

    # unary alu -- name the *result* so .ins() inherits the result dtype and default src=(operand,)
    (UPat(dtype=dtypes.float32).sqrt().named("x"), lambda x: x.ins(RDNA3Ops.v_sqrt_f32_e32)),
    (UPat(dtype=dtypes.float16).sqrt().named("x"), lambda x: x.ins(RDNA3Ops.v_sqrt_f16_e32)),
    (UPat(dtype=dtypes.float32).sin().named("x"), lambda x: x.ins(RDNA3Ops.v_sin_f32_e32)),
    (UPat(dtype=dtypes.float16).sin().named("x"), lambda x: x.ins(RDNA3Ops.v_sin_f16_e32)),
    (UPat(dtype=dtypes.float64).reciprocal().named("x"), lambda x: x.ins(RDNA3Ops.v_rcp_f64_e32)),
    (UPat(dtype=dtypes.float32).reciprocal().named("x"), lambda x: x.ins(RDNA3Ops.v_rcp_f32_e32)),
    (UPat(dtype=dtypes.float16).reciprocal().named("x"), lambda x: x.ins(RDNA3Ops.v_rcp_f16_e32)),
    # Do these decomposed cos expressions get detected in rewrite?
    (UPat(dtype=dtypes.float16).cos().named("x"), lambda x: x.ins(RDNA3Ops.v_cos_f16_e32)),
    (UPat(dtype=dtypes.float32).cos().named("x"), lambda x: x.ins(RDNA3Ops.v_cos_f32_e32)),
    (UPat(dtype=dtypes.float16).log2().named("x"), lambda x: x.ins(RDNA3Ops.v_log_f16_e32)),
    (UPat(dtype=dtypes.float32).log2().named("x"), lambda x: x.ins(RDNA3Ops.v_log_f32_e32)),
    (UPat(dtype=dtypes.float16).exp2().named("x"), lambda x: x.ins(RDNA3Ops.v_exp_f16_e32)),
    (UPat(dtype=dtypes.float32).exp2().named("x"), lambda x: x.ins(RDNA3Ops.v_exp_f32_e32)),

    # binary alu
    (UPat(Ops.MAX, dtype=dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f16_e32)),
    (UPat(Ops.MAX, dtype=dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f32_e32)),
    (UPat(Ops.MAX, dtype=dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f64)),
    (UPat(Ops.MAX, dtype=dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_max_i16)),
    (UPat(Ops.MAX, dtype=dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_i32_e32)),
    (UPat(Ops.MAX, dtype=dtypes.uint32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_u32_e32)),
    (UPat(Ops.SUB, dtype=dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_f16_e32)),
    (UPat(Ops.SUB, dtype=dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_f32_e32)),
    # (UPat(Ops.SUB, dtype=dtypes.uint32, name="x"), lambda x: x.ins(None if x.dtype.count > 1 else RDNA3Ops.s_sub_u32)), # v_sub_co_u32??
    (UPat(Ops.SUB, dtype=dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_nc_i16)),
    (UPat(Ops.SUB, dtype=dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_nc_i32)),
    ((UPat(dtype=dtypes.int16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_i16)),
    ((UPat(dtype=dtypes.int32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_i32)),
    ((UPat(dtype=dtypes.uint16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_u16)),
    ((UPat(dtype=dtypes.uint32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_u32_e32)),
    ((UPat(dtype=dtypes.float16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f16_e32)),
    ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f32_e32)),
    ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f64)),

    # TODO: figure out muls, hi, i24??
    ((UPat(dtype=dtypes.float16) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f16_e32)),
    ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f32_e32)),
    ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f64)),

    # fused multiply add
    # TODO: check foldable? is that to reduce to fmac? one variable used twice ig
    # ((UPat(Ops.MUL, dtype=dtypes.float32, name="a") + UPat.var("b")).named("c"), lambda ctx,a,b,c: a.ins(RDNA3Ops.v_fma_f32, src=(*a.src, b))),

    # comparisons
    # Notes:
    # - VALU CMPNE for ints does exist: v_cmp_ne_{i,u}32_e32 (the s_cmp scalar form is s_cmp_lg_*).
    # - operand dtype goes in src=(...); the compare node's own dtype is bool (the result), so filtering on
    #   the second positional UPat arg (== result dtype) never matches an int/float operand.
    # - name the *result* so the INS replaces the compare and inherits its dtype; default src is already (a,b).
    # - result dtype is left as bool for now -- TODO: VALU compares write a VCC/EXEC mask, sort out that protocol.
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_i32_e32)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.uint32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_u32_e32)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f16_e32)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f32_e32)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f64_e32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.int32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_ne_i32_e32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.uint32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_ne_u32_e32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f16_e32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f32_e32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f64_e32)),
    ((UPat(dtype=dtypes.int32) < UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_lt_i32_e32)),
    ((UPat(dtype=dtypes.uint32) < UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_lt_u32_e32)),
    ((UPat(dtype=dtypes.float16) < UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_lt_f16_e32)),
    ((UPat(dtype=dtypes.float32) < UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_lt_f32_e32)),
    ((UPat(dtype=dtypes.float64) < UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_lt_f64_e32)),
    ((UPat(dtype=dtypes.int32) > UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_gt_i32_e32)),
    ((UPat(dtype=dtypes.uint32) > UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_gt_u32_e32)),
    ((UPat(dtype=dtypes.int32) <= UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_le_i32_e32)),
    ((UPat(dtype=dtypes.uint32) <= UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_le_u32_e32)),
    ((UPat(dtype=dtypes.int32) >= UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_ge_i32_e32)),
    ((UPat(dtype=dtypes.uint32) >= UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_cmp_ge_u32_e32)),

    # casts
    # - whats the difference between:
    # UPat(...).cast(..., name="x") and UPat(.., name="y").cast(..., name="x")
    (UPat(dtype=dtypes.int32).cast(dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_i16_e32)),
    (UPat(dtype=dtypes.int32).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_f32_e32)),
    (UPat(dtype=dtypes.int32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_f64_e32)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.uint16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_u16_e32)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_f32_e32)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_f64_e32)),
    (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_f32_e32)),
    (UPat(dtype=dtypes.float16).cast(dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_i16_e32)),
    (UPat(dtype=dtypes.float16).cast(dtypes.uint16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_u16_e32)),
    (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_f16_e32)),
    (UPat(dtype=dtypes.float32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_f64_e32)),
    (UPat(dtype=dtypes.float32).cast(dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_i32_e32)),
    (UPat(dtype=dtypes.float32).cast(dtypes.uint32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_u32_e32)),


    # Memory ops
    # (UPat(Ops.LOAD, dtypes.int32, src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(RDNA3Ops.ds_load_)),
    (UPat(Ops.LOAD, dt_32bit, src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(handle_memspace(x, RDNA3Ops.ds_load_b32, RDNA3Ops.global_load_b32), src=())),
])

# **** RDNA3 Instruction encoding ****
def encode(x:UOp):
    pass

encodings = {
}

class RDNA3Renderer(ISARenderer):
    isel_matcher = isel_matcher
    code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.CMPEQ, Ops.CMPNE, Ops.RECIPROCAL)}

    def __init__(self, target:Target):
        super().__init__(target)
        from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
        self.compiler = AMDLLVMCompiler(arch="rdna3")

    def asm_str(self, uops:list[UOp], function_name:str) -> str:
        pass

    def render(self, uops:list[UOp]) -> str:
        for u in uops:
            if u is not Ops.INS: continue
        pass
