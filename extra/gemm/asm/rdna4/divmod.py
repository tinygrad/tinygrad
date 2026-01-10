from extra.assembly.amd.autogen.rdna4.ins import *

# --------------------------------------------------------------------
# Integer division helpers (RDNA4): float reciprocal estimate + fixups.
# Return a flat list of ops; splice with *helper(...)
# --------------------------------------------------------------------

def u32_div_floor_f32_refine(q, n, d, prod, r) -> list[Inst]:
  # q = floor(n / d), unsigned, d != 0
  return [
    v_cvt_f32_u32_e32(q, d),
    v_rcp_iflag_f32_e32(q, q),
    v_cvt_f32_u32_e32(prod, n),
    v_mul_f32_e32(q, q, prod),
    v_cvt_u32_f32_e32(q, q),

    v_mul_lo_u32(prod, q, d),
    v_sub_nc_u32_e32(r, n, prod),

    v_cmp_eq_u32_e64(RawImm(106), r, d),
    s_mov_b32(EXEC_LO, VCC_LO),
    v_add_nc_u32_e32(q, 1, q),
    s_mov_b32(EXEC_LO, -1),

    v_cmp_gt_u32_e64(RawImm(106), r, d),
    s_mov_b32(EXEC_LO, VCC_LO),
    v_sub_nc_u32_e64(q, q, 1),
    s_mov_b32(EXEC_LO, -1),
  ]


def u32_divmod_floor_f32_refine(q, r, n, d, prod) -> list[Inst]:
  return [
    *u32_div_floor_f32_refine(q, n, d, prod, r),
    v_mul_lo_u32(prod, q, d),
    v_sub_nc_u32_e32(r, n, prod),
  ]


def u32_div_ceil_f32(q, n, d, prod, r) -> list[Inst]:
  # q = ceil(n / d), unsigned, d != 0
  return [
    v_cvt_f32_u32_e32(q, d),
    v_rcp_iflag_f32_e32(q, q),
    v_cvt_f32_u32_e32(prod, n),
    v_mul_f32_e32(q, q, prod),
    v_cvt_u32_f32_e32(q, q),

    v_mul_lo_u32(prod, q, d),
    v_sub_nc_u32_e32(r, n, prod),

    v_cmp_ne_u32_e64(RawImm(106), r, 0),
    v_add_co_ci_u32(q, VCC_LO, q, 0, VCC_LO),
  ]
