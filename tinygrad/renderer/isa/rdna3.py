from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.helpers import Target
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

pre_isel_matcher = PatternMatcher([
])

isel_matcher = PatternMatcher([
    # SALU Ops

    # --- VALU Ops ---
    # TODO: when to use 64 vs 32 bit embedding? e32/e64
    # - when we need to pass extra info? 
    # - check spec
    # unary alu
    (UPat(Ops.NOOP, name="x"), lambda x: x.ins(RDNA3Ops.v_nop_e32)),

    (UPat(name="x", dtype=dtypes.float32).sqrt(), lambda x: x.ins(RDNA3Ops.v_sqrt_f32_e32)),
    (UPat(name="x", dtype=dtypes.float16).sqrt(), lambda x: x.ins(RDNA3Ops.v_sqrt_f16_e32)),
    (UPat(name="x", dtype=dtypes.float32).sin(), lambda x: x.ins(RDNA3Ops.v_sin_f32_e32)),
    (UPat(name="x", dtype=dtypes.float16).sin(), lambda x: x.ins(RDNA3Ops.v_sin_f16_e32)),
    # Do these decomposed cos expressions get detected in rewrite?
    (UPat(name="x", dtype=dtypes.float16).cos(), lambda x: x.ins(RDNA3Ops.v_cos_f16_e32)),
    (UPat(name="x", dtype=dtypes.float32).cos(), lambda x: x.ins(RDNA3Ops.v_cos_f32_e32)),
    (UPat(name="x", dtype=dtypes.float16).log2(), lambda x: x.ins(RDNA3Ops.v_log_f16_e32)),
    (UPat(name="x", dtype=dtypes.float32).log2(), lambda x: x.ins(RDNA3Ops.v_log_f32_e32)),
    (UPat(name="x", dtype=dtypes.float16).exp2(), lambda x: x.ins(RDNA3Ops.v_exp_f16_e32)),
    (UPat(name="x", dtype=dtypes.float32).exp2(), lambda x: x.ins(RDNA3Ops.v_exp_f32_e32)),

    # binary alu
    (UPat(Ops.MAX, dtype=dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_max_f16)),
    (UPat(Ops.MAX, dtype=dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_max_f32)),
    (UPat(Ops.MAX, dtype=dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_max_f64 if x.dtype.count > 1 else None)),
    (UPat(Ops.MAX, dtype=dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_max_i16 if x.dtype.count > 1 else None)),
    (UPat(Ops.MAX, dtype=dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_i32_e32 if x.dtype.count > 1 else RDNA3Ops.s_max_i32)),
    (UPat(Ops.MAX, dtype=dtypes.uint32, name="x"), lambda x: x.ins(RDNA3Ops.v_max_u32_e32 if x.dtype.count > 1 else RDNA3Ops.s_max_u32)),
    (UPat(Ops.SUB, dtype=dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_sub_f16)),
    (UPat(Ops.SUB, dtype=dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_sub_f32)),
    (UPat(Ops.SUB, dtype=dtypes.uint32, name="x"), lambda x: x.ins(None if x.dtype.count > 1 else RDNA3Ops.s_sub_u32)), # v_sub_co_u32??
    (UPat(Ops.SUB, dtype=dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_nc_i16 if x.dtype.count > 1 else None)),
    (UPat(Ops.SUB, dtype=dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_sub_nc_i32 if x.dtype.count > 1 else RDNA3Ops.s_sub_i32)),
    ((UPat(dtype=dtypes.int16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_i16 if x.dtype.count > 1 else None)),
    ((UPat(dtype=dtypes.int32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_i32 if x.dtype.count > 1 else RDNA3Ops.s_add_i32)),
    ((UPat(dtype=dtypes.uint16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_u16 if x.dtype.count > 1 else None)),
    ((UPat(dtype=dtypes.uint32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_nc_u32_e32 if x.dtype.count > 1 else RDNA3Ops.s_add_u32)),
    ((UPat(dtype=dtypes.float16) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_add_f16)),
    ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_add_f32)),
    ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f64 if x.dtype.count > 1 else None)),

    # TODO: figure out muls, hi, i24??
    ((UPat(dtype=dtypes.float16) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_mul_f16)),
    ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_mul_f32)),
    ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_mul_f64 if x.dtype.count > 1 else None)),

    # fused multiply add
    # TODO: check foldable? is that to reduce to fmac? one variable used twice ig
    # ((UPat(Ops.MUL, dtype=dtypes.float32, name="a") + UPat.var("b")).named("c"), lambda ctx,a,b,c: a.ins(RDNA3Ops.v_fma_f32, src=(*a.src, b))),

    # comparisons
    # Notes: 
    # - No VALU CMPNE for ints?, does that need to be decomposed?
    # - Do I need to automatically downscale scalar f64?
    # - figure out EXEC mask protocol, when to emit cmpx ins?
    (UPat(Ops.CMPEQ, dtypes.int32, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(RDNA3Ops.s_cmp_eq_i32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_eq_i32_e32, src=(a, b))),
    (UPat(Ops.CMPEQ, dtypes.uint32, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(RDNA3Ops.s_cmp_eq_u32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_eq_u32_e32, src=(a, b))),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_cmp_eq_f16)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cmp_eq_f32)),
    (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_eq_f64_e32 if x.dtype.count > 1 else None)),
    (UPat(Ops.CMPNE, dtypes.int32, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(RDNA3Ops.s_cmp_lg_i32, src=(a, b)) if a.dtype.count == 1 else None),
    (UPat(Ops.CMPNE, dtypes.uint32, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(RDNA3Ops.s_cmp_lg_u32, src=(a, b)) if a.dtype.count == 1 else None),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_cmp_lg_f16)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cmp_lg_f32)),
    (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), lambda x: x.ins(RDNA3Ops.v_cmp_lg_f64_e32 if x.dtype.count > 1 else None)),
    (UPat.var("a", dtypes.int32) < UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_lt_i32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_lt_i32_e32, src=(a,b))),
    (UPat.var("a", dtypes.uint32) < UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_lt_u32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_lt_u32_e32, src=(a, b))),
    (UPat.var("a", dtypes.float16) < UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_lt_f16 if a.dtype.count == 1 else RDNA3Ops.v_cmp_lt_f16_e32, src=(a, b))),
    (UPat.var("a", dtypes.float32) < UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_lt_f32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_lt_f32_e32, src=(a, b))),
    (UPat.var("a", dtypes.int32) > UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_gt_i32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_gt_i32_e32, src=(a,b))),
    (UPat.var("a", dtypes.uint32) > UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_gt_u32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_gt_u32_e32, src=(a, b))),
    (UPat.var("a", dtypes.int32) <= UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_le_i32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_le_i32_e32, src=(a,b))),
    (UPat.var("a", dtypes.uint32) <= UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_le_u32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_le_u32_e32, src=(a, b))),
    (UPat.var("a", dtypes.int32) >= UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_ge_i32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_ge_i32_e32, src=(a,b))),
    (UPat.var("a", dtypes.uint32) >= UPat.var("b"), lambda a, b: a.ins(RDNA3Ops.s_cmp_ge_u32 if a.dtype.count == 1 else RDNA3Ops.v_cmp_ge_u32_e32, src=(a, b))),

    # casts
    # - whats the difference between:
    # UPat(...).cast(..., name="x") and UPat(.., name="y").cast(..., name="x")
    (UPat(dtype=dtypes.int32).cast(dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_i16_e32)),
    (UPat(dtype=dtypes.int32).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_i32_f32)),
    (UPat(dtype=dtypes.int32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_i32_f64_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.uint16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_u16_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_u32_f32)),
    (UPat(dtype=dtypes.uint32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_u32_f64_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_f32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_f16_f32)),
    (UPat(dtype=dtypes.float16).cast(dtypes.int16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_i16_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.float16).cast(dtypes.uint16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f16_u16_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_f16_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_f32_f16)),
    (UPat(dtype=dtypes.float32).cast(dtypes.float64, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_f64_e32 if x.dtype.count > 1 else None)),
    (UPat(dtype=dtypes.float32).cast(dtypes.int32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_i32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_f32_i32)),
    (UPat(dtype=dtypes.float32).cast(dtypes.uint32, name="x"), lambda x: x.ins(RDNA3Ops.v_cvt_f32_u32_e32 if x.dtype.count > 1 else RDNA3Ops.s_cvt_f32_u32)),
])

# **** RDNA3 Instruction encoding ****
def encode(x:UOp):
    pass

encodings = {
}

class RDNA3Renderer(ISARenderer):
    isel_matcher = isel_matcher
    code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.CMPEQ, Ops.CMPNE)}

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
