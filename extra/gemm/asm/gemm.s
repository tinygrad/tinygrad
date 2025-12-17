.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.section	.text.
.global	gemm
.p2align	8
.type	gemm,@function

gemm:
  s_mov_b32 s49, s4
  s_and_b32 s1, s1, 0xffff
  s_load_dword s25, s[0:1], 0xe0
  s_load_dword s26, s[0:1], 0xf0
  s_load_dword s27, s[0:1], 0x100
  s_load_dword s28, s[0:1], 0xa0
  s_load_dword s29, s[0:1], 0xc0
  s_load_dword s30, s[0:1], 0x80
  s_load_dword s20, s[0:1], 0x40
  s_load_dword s21, s[0:1], 0x50
  s_load_dwordx2 s[4:5], s[0:1], 0x20
  s_load_dwordx2 s[8:9], s[0:1], 0x30
  s_load_dwordx2 s[12:13], s[0:1], 0x10
  s_load_dwordx2 s[16:17], s[0:1], 0x0
  s_load_dword s48, s[0:1], 0x110
  s_load_dword s50, s[0:1], 0x120
  s_load_dwordx2 s[36:37], s[0:1], 0x130
  s_load_dword s57, s[0:1], 0x140
  v_lshrrev_b32_e32 v1, 10, v0
  v_lshrrev_b32_e32 v2, 10, v1
  v_and_b32_e32 v2, 0x3ff, v2
  v_and_b32_e32 v1, 0x3ff, v1
  v_and_b32_e32 v0, 0x3ff, v0
  v_lshrrev_b32_e32 v3, 6, v0
  v_and_b32_e32 v0, 63, v0
  s_mov_b32 s22, s2
  s_mov_b32 s23, s3
  v_readfirstlane_b32 s24, v3
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s18, -16
  s_mov_b32 s14, -16
  s_mov_b32 s10, -16
  s_mov_b32 s6, -16
  s_mov_b32 s38, -16
  s_mov_b32 s19, 0x20000
  s_mov_b32 s15, 0x20000
  s_mov_b32 s11, 0x20000
  s_mov_b32 s7, 0x20000
  s_mov_b32 s39, 0x20000
  s_and_b32 s17, s17, 0xffff
  s_and_b32 s13, s13, 0xffff
  s_and_b32 s9, s9, 0xffff
  s_and_b32 s5, s5, 0xffff
  s_and_b32 s37, s37, 0xffff
  s_or_b32 s17, s17, 0x40000
  s_or_b32 s13, s13, 0x40000
  s_or_b32 s9, s9, 0x40000
  s_or_b32 s5, s5, 0x40000
  s_or_b32 s37, s37, 0x40000
  s_mov_b32 s35, 0x7060302
  v_mov_b32_e32 v9, 0xffff0000
  v_mov_b32_e32 v10, 0x7fff0000
  v_mov_b32_e32 v11, 0x7fff
  s_mul_i32 s31, s28, s25
  s_mov_b32 s6, s31
  s_mov_b32 s40, 0x80
  v_lshrrev_b32_e32 v4, 5, v0
  v_lshlrev_b32_e32 v4, 2, v4
  v_mul_lo_u32 v19, v4, s28
  v_and_b32_e32 v4, 31, v0
  v_lshlrev_b32_e32 v4, 2, v4
  v_add_u32_e32 v19, v19, v4
  s_mul_i32 s31, 8, s28
  v_add_u32_e64 v20, v19, s31
  v_add_u32_e64 v21, v20, s31
  v_add_u32_e64 v22, v21, s31
  v_add_u32_e64 v23, v22, s31
  v_add_u32_e64 v24, v23, s31
  v_add_u32_e64 v25, v24, s31
  v_add_u32_e64 v26, v25, s31
  v_add_u32_e64 v27, v26, s31
  v_add_u32_e64 v28, v27, s31
  v_add_u32_e64 v29, v28, s31
  v_add_u32_e64 v30, v29, s31
  s_mul_i32 s31, s23, 0x60
  s_add_u32 s31, s31, s24
  s_mul_i32 s32, s31, s28
  v_add_u32_e64 v19, v19, s32
  v_add_u32_e64 v20, v20, s32
  v_add_u32_e64 v21, v21, s32
  v_add_u32_e64 v22, v22, s32
  v_add_u32_e64 v23, v23, s32
  v_add_u32_e64 v24, v24, s32
  v_add_u32_e64 v25, v25, s32
  v_add_u32_e64 v26, v26, s32
  v_add_u32_e64 v27, v27, s32
  v_add_u32_e64 v28, v28, s32
  v_add_u32_e64 v29, v29, s32
  v_add_u32_e64 v30, v30, s32
  v_lshrrev_b32_e32 v4, 4, v0
  v_lshlrev_b32_e32 v5, 2, v4
  v_and_b32_e32 v4, 15, v0
  v_lshrrev_b32_e32 v6, 2, v4
  v_lshlrev_b32_e32 v6, 5, v6
  v_add_u32_e32 v5, v6, v5
  v_and_b32_e32 v4, 3, v0
  v_mul_u32_u24_e32 v6, 0x308, v4
  v_add_u32_e32 v5, v6, v5
  v_lshlrev_b32_e32 v31, 2, v5
  s_mul_i32 s31, s24, 0xc20
  s_add_u32 s42, 0, s31
  s_add_u32 s43, 0x3080, s42
  s_add_u32 s44, 0x3080, s43
  s_mul_i32 s31, s29, s26
  s_mov_b32 s10, s31
  s_mov_b32 s41, 0x80
  v_lshrrev_b32_e32 v4, 5, v0
  v_lshlrev_b32_e32 v4, 2, v4
  v_mul_lo_u32 v32, v4, s29
  v_and_b32_e32 v4, 31, v0
  v_lshlrev_b32_e32 v4, 2, v4
  v_add_u32_e32 v32, v32, v4
  s_mul_i32 s31, 8, s29
  v_add_u32_e64 v33, v32, s31
  v_add_u32_e64 v34, v33, s31
  v_add_u32_e64 v35, v34, s31
  v_add_u32_e64 v36, v35, s31
  v_add_u32_e64 v37, v36, s31
  v_add_u32_e64 v38, v37, s31
  v_add_u32_e64 v39, v38, s31
  s_mul_i32 s31, s22, 64
  s_add_u32 s31, s31, s24
  s_mul_i32 s32, s31, s29
  v_add_u32_e64 v32, v32, s32
  v_add_u32_e64 v33, v33, s32
  v_add_u32_e64 v34, v34, s32
  v_add_u32_e64 v35, v35, s32
  v_add_u32_e64 v36, v36, s32
  v_add_u32_e64 v37, v37, s32
  v_add_u32_e64 v38, v38, s32
  v_add_u32_e64 v39, v39, s32
  s_cmp_le_u32 s48, 1
  s_cbranch_scc1 label_012C
  s_lshr_b32 s32, s27, 6
  v_cvt_f32_u32_e32 v4, s48
  s_sub_i32 s31, 0, s48
  v_rcp_iflag_f32_e32 v4, v4
  s_nop 0
  v_mul_f32_e32 v4, 0x4f7ffffe, v4
  v_cvt_u32_f32_e32 v4, v4
  v_mul_lo_u32 v5, s31, v4
  v_mul_hi_u32 v5, v4, v5
  v_add_u32_e32 v4, v4, v5
  v_mul_hi_u32 v4, s32, v4
  v_mul_lo_u32 v5, v4, s48
  v_sub_u32_e32 v7, s32, v5
  v_add_u32_e32 v6, 1, v4
  v_cmp_le_u32_e32 vcc, s48, v7
  v_subrev_u32_e32 v5, s48, v7
  s_nop 0
  v_cndmask_b32_e32 v4, v4, v6, vcc
  v_cndmask_b32_e32 v7, v7, v5, vcc
  v_add_u32_e32 v5, 1, v4
  v_cmp_le_u32_e32 vcc, s48, v7
  s_nop 1
  v_cndmask_b32_e32 v7, v4, v5, vcc
  s_nop 3
  v_readfirstlane_b32 s32, v7
  s_nop 3
  s_mul_i32 s32, s32, 64
  s_mul_i32 s31, s49, s32
  s_sub_i32 s52, s27, s31
  s_sub_i32 s31, s48, 1
  s_cmp_eq_i32 s49, s31
  s_cselect_b32 s27, s52, s32
  s_mul_i32 s31, s32, 2
  s_mul_i32 s31, s31, s49
  v_add_u32_e64 v19, v19, s31
  v_add_u32_e64 v20, v20, s31
  v_add_u32_e64 v21, v21, s31
  v_add_u32_e64 v22, v22, s31
  v_add_u32_e64 v23, v23, s31
  v_add_u32_e64 v24, v24, s31
  v_add_u32_e64 v25, v25, s31
  v_add_u32_e64 v26, v26, s31
  v_add_u32_e64 v27, v27, s31
  v_add_u32_e64 v28, v28, s31
  v_add_u32_e64 v29, v29, s31
  v_add_u32_e64 v30, v30, s31
  s_mul_i32 s31, s32, 2
  s_mul_i32 s31, s31, s49
  v_add_u32_e64 v32, v32, s31
  v_add_u32_e64 v33, v33, s31
  v_add_u32_e64 v34, v34, s31
  v_add_u32_e64 v35, v35, s31
  v_add_u32_e64 v36, v36, s31
  v_add_u32_e64 v37, v37, s31
  v_add_u32_e64 v38, v38, s31
  v_add_u32_e64 v39, v39, s31
label_012C:
  v_lshrrev_b32_e32 v4, 4, v0
  v_lshlrev_b32_e32 v5, 2, v4
  v_and_b32_e32 v4, 15, v0
  v_lshrrev_b32_e32 v6, 2, v4
  v_lshlrev_b32_e32 v6, 5, v6
  v_add_u32_e32 v5, v6, v5
  v_and_b32_e32 v4, 3, v0
  v_mul_u32_u24_e32 v6, 0x208, v4
  v_add_u32_e32 v5, v6, v5
  v_lshlrev_b32_e32 v40, 2, v5
  s_mul_i32 s31, s24, 0x200
  v_add_u32_e32 v40, s31, v40
  s_mul_i32 s31, s24, 0x820
  s_add_u32 s45, 0x9180, s31
  s_add_u32 s46, 0x2080, s45
  s_add_u32 s47, 0x2080, s46
  s_mul_i32 s31, s30, s25
  s_mov_b32 s18, s31
  s_cmp_lt_u32 s50, 1
  s_cbranch_scc0 label_0166
  v_and_b32_e64 v12, v0, 15
  v_mul_lo_u32 v12, v12, s30
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_i32_i24_e32 v4, 16, v4
  v_add_u32_e32 v12, v4, v12
  s_mul_i32 s31, s23, 0x60
  s_mul_i32 s31, s31, s30
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, s22, 64
  s_mul_i32 s31, s31, 4
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, 64, s24
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, 16, s30
  v_add_u32_e32 v13, s31, v12
  v_add_u32_e32 v14, s31, v13
  v_add_u32_e32 v15, s31, v14
  v_add_u32_e32 v16, s31, v15
  v_add_u32_e32 v17, s31, v16
  s_mul_i32 s31, s23, 0x60
  s_add_i32 s31, s31, s24
  s_mul_i32 s31, s31, s30
  s_mul_i32 s32, s22, 64
  s_mul_i32 s32, s32, 4
  s_add_i32 s31, s31, s32
  v_lshlrev_b32_e32 v18, 2, v0
  v_add_u32_e32 v18, s31, v18
  s_branch label_018B
label_0166:
  v_and_b32_e64 v12, v0, 15
  v_mul_lo_u32 v12, v12, s30
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_i32_i24_e32 v4, 8, v4
  v_add_u32_e32 v12, v4, v12
  s_mul_i32 s31, s23, 0x60
  s_mul_i32 s31, s31, s30
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, s22, 64
  s_mul_i32 s31, s31, 2
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, 32, s24
  v_add_u32_e32 v12, s31, v12
  s_mul_i32 s31, 16, s30
  v_add_u32_e32 v13, s31, v12
  v_add_u32_e32 v14, s31, v13
  v_add_u32_e32 v15, s31, v14
  v_add_u32_e32 v16, s31, v15
  v_add_u32_e32 v17, s31, v16
  s_mul_i32 s31, s23, 0x60
  s_add_i32 s31, s31, s24
  s_mul_i32 s31, s31, s30
  s_mul_i32 s32, s22, 64
  s_mul_i32 s32, s32, 2
  s_add_i32 s31, s31, s32
  v_lshrrev_b32_e32 v4, 5, v0
  s_mul_i32 s32, s30, 4
  v_mul_lo_u32 v4, v4, s32
  v_and_b32_e32 v5, 31, v0
  v_lshlrev_b32_e32 v5, 2, v5
  v_add_u32_e32 v18, v4, v5
  v_add_u32_e32 v18, s31, v18
label_018B:
  s_cmp_eq_u32 s57, 1
  s_cbranch_scc0 label_01C6
  s_cmp_eq_i32 s49, 0
  s_cbranch_scc0 label_01C6
  s_mul_i32 s31, 2, s26
  s_mov_b32 s38, s31
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_i32_i24_e32 v80, 8, v4
  s_mul_i32 s31, 32, s24
  v_add_u32_e32 v80, s31, v80
  s_mul_i32 s32, s22, 64
  s_mul_i32 s32, s32, 2
  v_add_u32_e32 v80, s32, v80
  v_mov_b32_e32 v82, 0
  v_mov_b32_e32 v83, 0
  buffer_load_dwordx2 v[82:83], v80, s[36:39], 0 offen
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v4, 0xffff0000
  v_and_b32_e32 v4, v82, v4
  v_mov_b32_e32 v5, 0xffff
  v_and_b32_e32 v5, v82, v5
  v_mov_b32_e32 v85, v4
  v_lshlrev_b32_e32 v84, 16, v5
  v_mov_b32_e32 v4, 0xffff0000
  v_and_b32_e32 v4, v83, v4
  v_mov_b32_e32 v5, 0xffff
  v_and_b32_e32 v5, v83, v5
  v_mov_b32_e32 v87, v4
  v_lshlrev_b32_e32 v86, 16, v5
  v_mov_b32_e32 v44, v84
  v_mov_b32_e32 v45, v85
  v_mov_b32_e32 v46, v86
  v_mov_b32_e32 v47, v87
  v_mov_b32_e32 v48, v84
  v_mov_b32_e32 v49, v85
  v_mov_b32_e32 v50, v86
  v_mov_b32_e32 v51, v87
  v_mov_b32_e32 v52, v84
  v_mov_b32_e32 v53, v85
  v_mov_b32_e32 v54, v86
  v_mov_b32_e32 v55, v87
  v_mov_b32_e32 v56, v84
  v_mov_b32_e32 v57, v85
  v_mov_b32_e32 v58, v86
  v_mov_b32_e32 v59, v87
  v_mov_b32_e32 v60, v84
  v_mov_b32_e32 v61, v85
  v_mov_b32_e32 v62, v86
  v_mov_b32_e32 v63, v87
  v_mov_b32_e32 v64, v84
  v_mov_b32_e32 v65, v85
  v_mov_b32_e32 v66, v86
  v_mov_b32_e32 v67, v87
  s_branch label_01DE
label_01C6:
  v_mov_b32_e32 v44, 0
  v_mov_b32_e32 v45, 0
  v_mov_b32_e32 v46, 0
  v_mov_b32_e32 v47, 0
  v_mov_b32_e32 v48, 0
  v_mov_b32_e32 v49, 0
  v_mov_b32_e32 v50, 0
  v_mov_b32_e32 v51, 0
  v_mov_b32_e32 v52, 0
  v_mov_b32_e32 v53, 0
  v_mov_b32_e32 v54, 0
  v_mov_b32_e32 v55, 0
  v_mov_b32_e32 v56, 0
  v_mov_b32_e32 v57, 0
  v_mov_b32_e32 v58, 0
  v_mov_b32_e32 v59, 0
  v_mov_b32_e32 v60, 0
  v_mov_b32_e32 v61, 0
  v_mov_b32_e32 v62, 0
  v_mov_b32_e32 v63, 0
  v_mov_b32_e32 v64, 0
  v_mov_b32_e32 v65, 0
  v_mov_b32_e32 v66, 0
  v_mov_b32_e32 v67, 0
label_01DE:
  s_add_u32 m0, 0, s42
  buffer_load_dword v19, s[4:7], 0 offen lds
  s_add_u32 m0, 0x100, s42
  buffer_load_dword v20, s[4:7], 0 offen lds
  s_add_u32 m0, 0x200, s42
  buffer_load_dword v21, s[4:7], 0 offen lds
  s_add_u32 m0, 0x300, s42
  buffer_load_dword v22, s[4:7], 0 offen lds
  s_add_u32 m0, 0x400, s42
  buffer_load_dword v23, s[4:7], 0 offen lds
  s_add_u32 m0, 0x500, s42
  buffer_load_dword v24, s[4:7], 0 offen lds
  s_add_u32 m0, 0x600, s42
  buffer_load_dword v25, s[4:7], 0 offen lds
  s_add_u32 m0, 0x700, s42
  buffer_load_dword v26, s[4:7], 0 offen lds
  s_add_u32 m0, 0x800, s42
  buffer_load_dword v27, s[4:7], 0 offen lds
  s_add_u32 m0, 0x900, s42
  buffer_load_dword v28, s[4:7], 0 offen lds
  s_add_u32 m0, 0xa00, s42
  buffer_load_dword v29, s[4:7], 0 offen lds
  s_add_u32 m0, 0xb00, s42
  buffer_load_dword v30, s[4:7], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  s_sub_u32 s6, s6, s40
  s_add_u32 m0, 0, s45
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 m0, 0x100, s45
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 m0, 0x200, s45
  buffer_load_dword v34, s[8:11], 0 offen lds
  s_add_u32 m0, 0x300, s45
  buffer_load_dword v35, s[8:11], 0 offen lds
  s_add_u32 m0, 0x400, s45
  buffer_load_dword v36, s[8:11], 0 offen lds
  s_add_u32 m0, 0x500, s45
  buffer_load_dword v37, s[8:11], 0 offen lds
  s_add_u32 m0, 0x600, s45
  buffer_load_dword v38, s[8:11], 0 offen lds
  s_add_u32 m0, 0x700, s45
  buffer_load_dword v39, s[8:11], 0 offen lds
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  s_sub_u32 s10, s10, s41
  s_add_u32 m0, 0, s43
  buffer_load_dword v19, s[4:7], 0 offen lds
  s_add_u32 m0, 0x100, s43
  buffer_load_dword v20, s[4:7], 0 offen lds
  s_add_u32 m0, 0x200, s43
  buffer_load_dword v21, s[4:7], 0 offen lds
  s_add_u32 m0, 0x300, s43
  buffer_load_dword v22, s[4:7], 0 offen lds
  s_add_u32 m0, 0x400, s43
  buffer_load_dword v23, s[4:7], 0 offen lds
  s_add_u32 m0, 0x500, s43
  buffer_load_dword v24, s[4:7], 0 offen lds
  s_add_u32 m0, 0x600, s43
  buffer_load_dword v25, s[4:7], 0 offen lds
  s_add_u32 m0, 0x700, s43
  buffer_load_dword v26, s[4:7], 0 offen lds
  s_add_u32 m0, 0x800, s43
  buffer_load_dword v27, s[4:7], 0 offen lds
  s_add_u32 m0, 0x900, s43
  buffer_load_dword v28, s[4:7], 0 offen lds
  s_add_u32 m0, 0xa00, s43
  buffer_load_dword v29, s[4:7], 0 offen lds
  s_add_u32 m0, 0xb00, s43
  buffer_load_dword v30, s[4:7], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  s_sub_u32 s6, s6, s40
  s_add_u32 m0, 0, s46
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 m0, 0x100, s46
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 m0, 0x200, s46
  buffer_load_dword v34, s[8:11], 0 offen lds
  s_add_u32 m0, 0x300, s46
  buffer_load_dword v35, s[8:11], 0 offen lds
  s_add_u32 m0, 0x400, s46
  buffer_load_dword v36, s[8:11], 0 offen lds
  s_add_u32 m0, 0x500, s46
  buffer_load_dword v37, s[8:11], 0 offen lds
  s_add_u32 m0, 0x600, s46
  buffer_load_dword v38, s[8:11], 0 offen lds
  s_add_u32 m0, 0x700, s46
  buffer_load_dword v39, s[8:11], 0 offen lds
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  s_sub_u32 s10, s10, s41
  s_add_u32 m0, 0, s44
  buffer_load_dword v19, s[4:7], 0 offen lds
  s_add_u32 m0, 0x100, s44
  buffer_load_dword v20, s[4:7], 0 offen lds
  s_add_u32 m0, 0x200, s44
  buffer_load_dword v21, s[4:7], 0 offen lds
  s_add_u32 m0, 0x300, s44
  buffer_load_dword v22, s[4:7], 0 offen lds
  s_add_u32 m0, 0x400, s44
  buffer_load_dword v23, s[4:7], 0 offen lds
  s_add_u32 m0, 0x500, s44
  buffer_load_dword v24, s[4:7], 0 offen lds
  s_add_u32 m0, 0x600, s44
  buffer_load_dword v25, s[4:7], 0 offen lds
  s_add_u32 m0, 0x700, s44
  buffer_load_dword v26, s[4:7], 0 offen lds
  s_add_u32 m0, 0x800, s44
  buffer_load_dword v27, s[4:7], 0 offen lds
  s_add_u32 m0, 0x900, s44
  buffer_load_dword v28, s[4:7], 0 offen lds
  s_add_u32 m0, 0xa00, s44
  buffer_load_dword v29, s[4:7], 0 offen lds
  s_add_u32 m0, 0xb00, s44
  buffer_load_dword v30, s[4:7], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  s_sub_u32 s6, s6, s40
  s_add_u32 m0, 0, s47
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 m0, 0x100, s47
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 m0, 0x200, s47
  buffer_load_dword v34, s[8:11], 0 offen lds
  s_add_u32 m0, 0x300, s47
  buffer_load_dword v35, s[8:11], 0 offen lds
  s_add_u32 m0, 0x400, s47
  buffer_load_dword v36, s[8:11], 0 offen lds
  s_add_u32 m0, 0x500, s47
  buffer_load_dword v37, s[8:11], 0 offen lds
  s_add_u32 m0, 0x600, s47
  buffer_load_dword v38, s[8:11], 0 offen lds
  s_add_u32 m0, 0x700, s47
  buffer_load_dword v39, s[8:11], 0 offen lds
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  s_sub_u32 s10, s10, s41
  s_mov_b32 s34, s27
  s_mov_b32 s33, 0
  s_mul_i32 s31, s23, 0x60
  s_sub_i32 s51, s25, s31
  s_waitcnt vmcnt(40)
  s_barrier
  ds_read_b128 a[0:3], v31
  ds_read_b128 a[4:7], v31 offset:64
  ds_read_b128 a[8:11], v31 offset:512
  ds_read_b128 a[12:15], v31 offset:576
  ds_read_b128 a[16:19], v31 offset:1024
  ds_read_b128 a[20:23], v31 offset:1088
  ds_read_b128 a[24:27], v31 offset:1536
  ds_read_b128 a[28:31], v31 offset:1600
  ds_read_b128 a[32:35], v31 offset:2048
  ds_read_b128 a[36:39], v31 offset:2112
  ds_read_b128 a[40:43], v31 offset:2560
  ds_read_b128 a[44:47], v31 offset:2624
  ds_read_b128 a[96:99], v40 offset:37248
  ds_read_b128 a[100:103], v40 offset:37312
  s_cmp_lt_i32 s24, 2
  s_cbranch_scc0 label_0702
label_02FF:
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s42
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s42
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31 offset:12416
  ds_read_b128 a[52:55], v31 offset:12480
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s42
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s42
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:12928
  ds_read_b128 a[60:63], v31 offset:12992
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s42
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s42
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:13440
  ds_read_b128 a[68:71], v31 offset:13504
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s42
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s42
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:13952
  ds_read_b128 a[76:79], v31 offset:14016
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s42
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s42
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:14464
  ds_read_b128 a[84:87], v31 offset:14528
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s42
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s42
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:14976
  ds_read_b128 a[92:95], v31 offset:15040
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s45
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s45
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:45568
  ds_read_b128 a[108:111], v40 offset:45632
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s45
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s45
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s45
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s45
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s45
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s45
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s43
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s43
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31 offset:24832
  ds_read_b128 a[4:7], v31 offset:24896
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s43
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s43
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:25344
  ds_read_b128 a[12:15], v31 offset:25408
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s43
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s43
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:25856
  ds_read_b128 a[20:23], v31 offset:25920
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s43
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s43
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:26368
  ds_read_b128 a[28:31], v31 offset:26432
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s43
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s43
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:26880
  ds_read_b128 a[36:39], v31 offset:26944
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s43
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s43
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:27392
  ds_read_b128 a[44:47], v31 offset:27456
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s46
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s46
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:53888
  ds_read_b128 a[100:103], v40 offset:53952
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s46
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s46
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s46
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s46
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s46
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s46
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s44
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s44
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31
  ds_read_b128 a[52:55], v31 offset:64
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s44
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s44
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:512
  ds_read_b128 a[60:63], v31 offset:576
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s44
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s44
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:1024
  ds_read_b128 a[68:71], v31 offset:1088
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s44
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s44
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:1536
  ds_read_b128 a[76:79], v31 offset:1600
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s44
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s44
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:2048
  ds_read_b128 a[84:87], v31 offset:2112
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s44
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s44
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:2560
  ds_read_b128 a[92:95], v31 offset:2624
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s47
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s47
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:37248
  ds_read_b128 a[108:111], v40 offset:37312
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s47
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s47
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s47
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s47
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s47
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s47
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s42
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s42
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31 offset:12416
  ds_read_b128 a[4:7], v31 offset:12480
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s42
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s42
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:12928
  ds_read_b128 a[12:15], v31 offset:12992
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s42
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s42
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:13440
  ds_read_b128 a[20:23], v31 offset:13504
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s42
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s42
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:13952
  ds_read_b128 a[28:31], v31 offset:14016
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s42
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s42
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:14464
  ds_read_b128 a[36:39], v31 offset:14528
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s42
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s42
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:14976
  ds_read_b128 a[44:47], v31 offset:15040
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s45
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s45
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:45568
  ds_read_b128 a[100:103], v40 offset:45632
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s45
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s45
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s45
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s45
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s45
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s45
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s43
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s43
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31 offset:24832
  ds_read_b128 a[52:55], v31 offset:24896
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s43
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s43
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:25344
  ds_read_b128 a[60:63], v31 offset:25408
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s43
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s43
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:25856
  ds_read_b128 a[68:71], v31 offset:25920
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s43
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s43
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:26368
  ds_read_b128 a[76:79], v31 offset:26432
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s43
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s43
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:26880
  ds_read_b128 a[84:87], v31 offset:26944
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s43
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s43
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:27392
  ds_read_b128 a[92:95], v31 offset:27456
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s46
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s46
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:53888
  ds_read_b128 a[108:111], v40 offset:53952
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s46
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s46
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s46
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s46
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s46
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s46
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s44
  buffer_load_dword v19, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s44
  buffer_load_dword v20, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31
  ds_read_b128 a[4:7], v31 offset:64
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s44
  buffer_load_dword v21, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s44
  buffer_load_dword v22, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:512
  ds_read_b128 a[12:15], v31 offset:576
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s44
  buffer_load_dword v23, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s44
  buffer_load_dword v24, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:1024
  ds_read_b128 a[20:23], v31 offset:1088
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s44
  buffer_load_dword v25, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s44
  buffer_load_dword v26, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:1536
  ds_read_b128 a[28:31], v31 offset:1600
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s44
  buffer_load_dword v27, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s44
  buffer_load_dword v28, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:2048
  ds_read_b128 a[36:39], v31 offset:2112
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s44
  buffer_load_dword v29, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s44
  buffer_load_dword v30, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:2560
  ds_read_b128 a[44:47], v31 offset:2624
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s47
  buffer_load_dword v32, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s47
  buffer_load_dword v33, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:37248
  ds_read_b128 a[100:103], v40 offset:37312
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s47
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s47
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s47
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s47
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s47
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s47
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_branch label_02FF
label_0702:
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s42
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31 offset:12416
  ds_read_b128 a[52:55], v31 offset:12480
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s42
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s42
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:12928
  ds_read_b128 a[60:63], v31 offset:12992
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s42
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s42
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:13440
  ds_read_b128 a[68:71], v31 offset:13504
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s42
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s42
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:13952
  ds_read_b128 a[76:79], v31 offset:14016
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s42
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s42
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:14464
  ds_read_b128 a[84:87], v31 offset:14528
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s42
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s42
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:14976
  ds_read_b128 a[92:95], v31 offset:15040
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s42
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s45
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:45568
  ds_read_b128 a[108:111], v40 offset:45632
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s45
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s45
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s45
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s45
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s45
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s45
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s45
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s43
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31 offset:24832
  ds_read_b128 a[4:7], v31 offset:24896
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s43
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s43
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:25344
  ds_read_b128 a[12:15], v31 offset:25408
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s43
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s43
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:25856
  ds_read_b128 a[20:23], v31 offset:25920
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s43
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s43
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:26368
  ds_read_b128 a[28:31], v31 offset:26432
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s43
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s43
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:26880
  ds_read_b128 a[36:39], v31 offset:26944
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s43
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s43
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:27392
  ds_read_b128 a[44:47], v31 offset:27456
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s43
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s46
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:53888
  ds_read_b128 a[100:103], v40 offset:53952
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s46
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s46
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s46
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s46
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s46
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s46
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s46
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s44
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31
  ds_read_b128 a[52:55], v31 offset:64
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s44
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s44
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:512
  ds_read_b128 a[60:63], v31 offset:576
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s44
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s44
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:1024
  ds_read_b128 a[68:71], v31 offset:1088
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s44
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s44
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:1536
  ds_read_b128 a[76:79], v31 offset:1600
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s44
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s44
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:2048
  ds_read_b128 a[84:87], v31 offset:2112
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s44
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s44
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:2560
  ds_read_b128 a[92:95], v31 offset:2624
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s44
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s47
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:37248
  ds_read_b128 a[108:111], v40 offset:37312
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s47
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s47
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s47
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s47
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s47
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s47
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s47
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s42
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31 offset:12416
  ds_read_b128 a[4:7], v31 offset:12480
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s42
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s42
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:12928
  ds_read_b128 a[12:15], v31 offset:12992
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s42
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s42
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:13440
  ds_read_b128 a[20:23], v31 offset:13504
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s42
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s42
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:13952
  ds_read_b128 a[28:31], v31 offset:14016
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s42
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s42
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:14464
  ds_read_b128 a[36:39], v31 offset:14528
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s42
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s42
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:14976
  ds_read_b128 a[44:47], v31 offset:15040
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s42
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s45
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:45568
  ds_read_b128 a[100:103], v40 offset:45632
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s45
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s45
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s45
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s45
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s45
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s45
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s45
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[96:97], a[0:1], v[44:47]
  s_add_u32 m0, 0, s43
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[48:51], v31 offset:24832
  ds_read_b128 a[52:55], v31 offset:24896
  v_mfma_f32_16x16x16_bf16 v[44:47], a[98:99], a[2:3], v[44:47]
  s_add_u32 m0, 0x100, s43
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[100:101], a[4:5], v[44:47]
  s_add_u32 m0, 0x200, s43
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[56:59], v31 offset:25344
  ds_read_b128 a[60:63], v31 offset:25408
  v_mfma_f32_16x16x16_bf16 v[44:47], a[102:103], a[6:7], v[44:47]
  s_add_u32 m0, 0x300, s43
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[96:97], a[8:9], v[48:51]
  s_add_u32 m0, 0x400, s43
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[64:67], v31 offset:25856
  ds_read_b128 a[68:71], v31 offset:25920
  v_mfma_f32_16x16x16_bf16 v[48:51], a[98:99], a[10:11], v[48:51]
  s_add_u32 m0, 0x500, s43
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[100:101], a[12:13], v[48:51]
  s_add_u32 m0, 0x600, s43
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[72:75], v31 offset:26368
  ds_read_b128 a[76:79], v31 offset:26432
  v_mfma_f32_16x16x16_bf16 v[48:51], a[102:103], a[14:15], v[48:51]
  s_add_u32 m0, 0x700, s43
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[96:97], a[16:17], v[52:55]
  s_add_u32 m0, 0x800, s43
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[80:83], v31 offset:26880
  ds_read_b128 a[84:87], v31 offset:26944
  v_mfma_f32_16x16x16_bf16 v[52:55], a[98:99], a[18:19], v[52:55]
  s_add_u32 m0, 0x900, s43
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[100:101], a[20:21], v[52:55]
  s_add_u32 m0, 0xa00, s43
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[88:91], v31 offset:27392
  ds_read_b128 a[92:95], v31 offset:27456
  v_mfma_f32_16x16x16_bf16 v[52:55], a[102:103], a[22:23], v[52:55]
  s_add_u32 m0, 0xb00, s43
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[96:97], a[24:25], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s46
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[104:107], v40 offset:53888
  ds_read_b128 a[108:111], v40 offset:53952
  v_mfma_f32_16x16x16_bf16 v[56:59], a[98:99], a[26:27], v[56:59]
  s_add_u32 m0, 0x100, s46
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[100:101], a[28:29], v[56:59]
  s_add_u32 m0, 0x200, s46
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[102:103], a[30:31], v[56:59]
  s_add_u32 m0, 0x300, s46
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[96:97], a[32:33], v[60:63]
  s_add_u32 m0, 0x400, s46
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[98:99], a[34:35], v[60:63]
  s_add_u32 m0, 0x500, s46
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[100:101], a[36:37], v[60:63]
  s_add_u32 m0, 0x600, s46
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[102:103], a[38:39], v[60:63]
  s_add_u32 m0, 0x700, s46
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[96:97], a[40:41], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[98:99], a[42:43], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[100:101], a[44:45], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[102:103], a[46:47], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_waitcnt vmcnt(20) lgkmcnt(0)
  s_barrier
  v_mfma_f32_16x16x16_bf16 v[44:47], a[104:105], a[48:49], v[44:47]
  s_add_u32 m0, 0, s44
  buffer_load_dword v19, s[4:7], 0 offen lds
  ds_read_b128 a[0:3], v31
  ds_read_b128 a[4:7], v31 offset:64
  v_mfma_f32_16x16x16_bf16 v[44:47], a[106:107], a[50:51], v[44:47]
  s_add_u32 m0, 0x100, s44
  buffer_load_dword v20, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[44:47], a[108:109], a[52:53], v[44:47]
  s_add_u32 m0, 0x200, s44
  buffer_load_dword v21, s[4:7], 0 offen lds
  ds_read_b128 a[8:11], v31 offset:512
  ds_read_b128 a[12:15], v31 offset:576
  v_mfma_f32_16x16x16_bf16 v[44:47], a[110:111], a[54:55], v[44:47]
  s_add_u32 m0, 0x300, s44
  buffer_load_dword v22, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[104:105], a[56:57], v[48:51]
  s_add_u32 m0, 0x400, s44
  buffer_load_dword v23, s[4:7], 0 offen lds
  ds_read_b128 a[16:19], v31 offset:1024
  ds_read_b128 a[20:23], v31 offset:1088
  v_mfma_f32_16x16x16_bf16 v[48:51], a[106:107], a[58:59], v[48:51]
  s_add_u32 m0, 0x500, s44
  buffer_load_dword v24, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[48:51], a[108:109], a[60:61], v[48:51]
  s_add_u32 m0, 0x600, s44
  buffer_load_dword v25, s[4:7], 0 offen lds
  ds_read_b128 a[24:27], v31 offset:1536
  ds_read_b128 a[28:31], v31 offset:1600
  v_mfma_f32_16x16x16_bf16 v[48:51], a[110:111], a[62:63], v[48:51]
  s_add_u32 m0, 0x700, s44
  buffer_load_dword v26, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[104:105], a[64:65], v[52:55]
  s_add_u32 m0, 0x800, s44
  buffer_load_dword v27, s[4:7], 0 offen lds
  ds_read_b128 a[32:35], v31 offset:2048
  ds_read_b128 a[36:39], v31 offset:2112
  v_mfma_f32_16x16x16_bf16 v[52:55], a[106:107], a[66:67], v[52:55]
  s_add_u32 m0, 0x900, s44
  buffer_load_dword v28, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[52:55], a[108:109], a[68:69], v[52:55]
  s_add_u32 m0, 0xa00, s44
  buffer_load_dword v29, s[4:7], 0 offen lds
  ds_read_b128 a[40:43], v31 offset:2560
  ds_read_b128 a[44:47], v31 offset:2624
  v_mfma_f32_16x16x16_bf16 v[52:55], a[110:111], a[70:71], v[52:55]
  s_add_u32 m0, 0xb00, s44
  buffer_load_dword v30, s[4:7], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[104:105], a[72:73], v[56:59]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s40, s40, 0
  s_add_u32 m0, 0, s47
  buffer_load_dword v32, s[8:11], 0 offen lds
  s_add_u32 s4, s40, s4
  s_addc_u32 s5, 0, s5
  ds_read_b128 a[96:99], v40 offset:37248
  ds_read_b128 a[100:103], v40 offset:37312
  v_mfma_f32_16x16x16_bf16 v[56:59], a[106:107], a[74:75], v[56:59]
  s_add_u32 m0, 0x100, s47
  buffer_load_dword v33, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[108:109], a[76:77], v[56:59]
  s_add_u32 m0, 0x200, s47
  buffer_load_dword v34, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[56:59], a[110:111], a[78:79], v[56:59]
  s_add_u32 m0, 0x300, s47
  buffer_load_dword v35, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[104:105], a[80:81], v[60:63]
  s_add_u32 m0, 0x400, s47
  buffer_load_dword v36, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[106:107], a[82:83], v[60:63]
  s_add_u32 m0, 0x500, s47
  buffer_load_dword v37, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[108:109], a[84:85], v[60:63]
  s_add_u32 m0, 0x600, s47
  buffer_load_dword v38, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[60:63], a[110:111], a[86:87], v[60:63]
  s_add_u32 m0, 0x700, s47
  buffer_load_dword v39, s[8:11], 0 offen lds
  v_mfma_f32_16x16x16_bf16 v[64:67], a[104:105], a[88:89], v[64:67]
  s_add_u32 s31, 0x100, s33
  s_cmp_lt_u32 s31, s34
  s_cselect_b32 s41, s41, 0
  s_add_u32 s8, s41, s8
  s_addc_u32 s9, 0, s9
  v_mfma_f32_16x16x16_bf16 v[64:67], a[106:107], a[90:91], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[108:109], a[92:93], v[64:67]
  v_mfma_f32_16x16x16_bf16 v[64:67], a[110:111], a[94:95], v[64:67]
  s_addk_i32 s33, 0x40
  s_cmp_lt_i32 s33, s34
  s_cbranch_scc0 label_0B05
  s_branch label_0702
label_0B05:
  s_cmp_le_u32 s48, 1
  s_cbranch_scc1 label_0FD5
  s_mov_b32 s31, 0x60
  s_cmp_lt_u32 s51, s31
  s_cbranch_scc1 label_0CE0
  v_mov_b32_e32 v5, 0
  s_and_b32 s17, s17, 0xffff
  s_cmp_lt_u32 s50, 1
  s_cbranch_scc0 label_0BC7
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_u32_u24_e32 v31, 0x44, v4
  v_and_b32_e32 v4, 15, v0
  v_mul_lo_u32 v5, 4, v4
  v_add_u32_e32 v31, v5, v31
  s_mul_i32 s31, s24, 0x110
  v_add_u32_e32 v31, s31, v31
  v_lshlrev_b32_e32 v31, 2, v31
  v_lshrrev_b32_e32 v4, 2, v0
  v_mul_u32_u24_e32 v40, 0x44, v4
  v_and_b32_e32 v4, 3, v0
  v_add_u32_e32 v40, v4, v40
  s_mul_i32 s31, s24, 4
  v_add_u32_e32 v40, s31, v40
  v_lshlrev_b32_e32 v40, 2, v40
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_barrier
  ds_write_b128 v31, v[44:47]
  ds_write_b128 v31, v[48:51] offset:4352
  ds_write_b128 v31, v[52:55] offset:8704
  ds_write_b128 v31, v[56:59] offset:13056
  ds_write_b128 v31, v[60:63] offset:17408
  ds_write_b128 v31, v[64:67] offset:21760
  s_waitcnt lgkmcnt(0)
  s_barrier
  ds_read_b32 v44, v40
  ds_read_b32 v45, v40 offset:64
  ds_read_b32 v46, v40 offset:128
  ds_read_b32 v47, v40 offset:192
  ds_read_b32 v48, v40 offset:4352
  ds_read_b32 v49, v40 offset:4416
  ds_read_b32 v50, v40 offset:4480
  ds_read_b32 v51, v40 offset:4544
  ds_read_b32 v52, v40 offset:8704
  ds_read_b32 v53, v40 offset:8768
  ds_read_b32 v54, v40 offset:8832
  ds_read_b32 v55, v40 offset:8896
  ds_read_b32 v56, v40 offset:13056
  ds_read_b32 v57, v40 offset:13120
  ds_read_b32 v58, v40 offset:13184
  ds_read_b32 v59, v40 offset:13248
  ds_read_b32 v60, v40 offset:17408
  ds_read_b32 v61, v40 offset:17472
  ds_read_b32 v62, v40 offset:17536
  ds_read_b32 v63, v40 offset:17600
  ds_read_b32 v64, v40 offset:21760
  ds_read_b32 v65, v40 offset:21824
  ds_read_b32 v66, v40 offset:21888
  ds_read_b32 v67, v40 offset:21952
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s31, s30, 4
  v_mov_b32_e32 v4, v18
  global_atomic_add_f32 v4, v44, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v45, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v46, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v47, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v48, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v49, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v50, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v51, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v52, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v53, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v54, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v55, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v56, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v57, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v58, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v59, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v60, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v61, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v62, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v63, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v64, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v65, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v66, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_add_f32 v4, v67, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_branch label_1098
label_0BC7:
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_u32_u24_e32 v31, 34, v4
  v_and_b32_e32 v4, 15, v0
  v_mul_lo_u32 v5, 2, v4
  v_add_u32_e32 v31, v5, v31
  s_mul_i32 s31, s24, 0x88
  v_add_u32_e32 v31, s31, v31
  v_lshlrev_b32_e32 v31, 2, v31
  v_and_b32_e32 v4, 31, v0
  v_lshrrev_b32_e32 v5, 1, v4
  v_mul_u32_u24_e32 v40, 34, v5
  v_and_b32_e32 v5, 1, v4
  v_add_u32_e32 v40, v5, v40
  v_lshrrev_b32_e32 v4, 5, v0
  v_mul_u32_u24_e32 v4, 8, v4
  v_add_u32_e32 v40, v4, v40
  s_mul_i32 s31, s24, 2
  v_add_u32_e32 v40, s31, v40
  v_lshlrev_b32_e32 v40, 2, v40
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_barrier
  v_cmp_u_f32_e64 s[56:57], v44, v44
  v_add3_u32 v8, v44, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v45, v45
  v_add3_u32 v8, v45, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v68, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v46, v46
  v_add3_u32 v8, v46, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v47, v47
  v_add3_u32 v8, v47, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v69, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v48, v48
  v_add3_u32 v8, v48, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v49, v49
  v_add3_u32 v8, v49, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v70, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v50, v50
  v_add3_u32 v8, v50, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v51, v51
  v_add3_u32 v8, v51, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v71, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v52, v52
  v_add3_u32 v8, v52, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v53, v53
  v_add3_u32 v8, v53, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v72, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v54, v54
  v_add3_u32 v8, v54, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v55, v55
  v_add3_u32 v8, v55, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v73, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v56, v56
  v_add3_u32 v8, v56, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v57, v57
  v_add3_u32 v8, v57, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v74, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v58, v58
  v_add3_u32 v8, v58, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v59, v59
  v_add3_u32 v8, v59, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v75, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v60, v60
  v_add3_u32 v8, v60, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v61, v61
  v_add3_u32 v8, v61, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v76, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v62, v62
  v_add3_u32 v8, v62, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v63, v63
  v_add3_u32 v8, v63, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v77, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v64, v64
  v_add3_u32 v8, v64, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v65, v65
  v_add3_u32 v8, v65, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v78, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v66, v66
  v_add3_u32 v8, v66, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v67, v67
  v_add3_u32 v8, v67, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v79, v5, v4, s35
  ds_write_b64 v31, v[68:69]
  ds_write_b64 v31, v[70:71] offset:2176
  ds_write_b64 v31, v[72:73] offset:4352
  ds_write_b64 v31, v[74:75] offset:6528
  ds_write_b64 v31, v[76:77] offset:8704
  ds_write_b64 v31, v[78:79] offset:10880
  s_waitcnt lgkmcnt(0)
  s_barrier
  ds_read_b32 v68, v40
  ds_read_b32 v69, v40 offset:64
  ds_read_b32 v70, v40 offset:2176
  ds_read_b32 v71, v40 offset:2240
  ds_read_b32 v72, v40 offset:4352
  ds_read_b32 v73, v40 offset:4416
  ds_read_b32 v74, v40 offset:6528
  ds_read_b32 v75, v40 offset:6592
  ds_read_b32 v76, v40 offset:8704
  ds_read_b32 v77, v40 offset:8768
  ds_read_b32 v78, v40 offset:10880
  ds_read_b32 v79, v40 offset:10944
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s31, s30, 8
  v_mov_b32_e32 v4, v18
  global_atomic_pk_add_bf16 v4, v68, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v69, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v70, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v71, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v72, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v73, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v74, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v75, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v76, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v77, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v78, s[16:17]
  v_add_u32_e64 v4, v4, s31
  global_atomic_pk_add_bf16 v4, v79, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_branch label_1098
label_0CE0:
  v_mov_b32_e32 v5, 0
  s_and_b32 s17, s17, 0xffff
  s_cmp_lt_u32 s50, 1
  s_cbranch_scc0 label_0DEA
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_u32_u24_e32 v31, 0x44, v4
  v_and_b32_e32 v4, 15, v0
  v_mul_lo_u32 v5, 4, v4
  v_add_u32_e32 v31, v5, v31
  s_mul_i32 s31, s24, 0x110
  v_add_u32_e32 v31, s31, v31
  v_lshlrev_b32_e32 v31, 2, v31
  v_lshrrev_b32_e32 v4, 2, v0
  v_mul_u32_u24_e32 v40, 0x44, v4
  v_and_b32_e32 v4, 3, v0
  v_add_u32_e32 v40, v4, v40
  s_mul_i32 s31, s24, 4
  v_add_u32_e32 v40, s31, v40
  v_lshlrev_b32_e32 v40, 2, v40
  s_lshr_b32 s31, s51, 2
  s_and_b32 s32, s51, 3
  s_cmp_lt_u32 s24, s32
  s_cselect_b32 s32, 1, 0
  s_add_u32 s51, s31, s32
  s_mov_b32 s33, 0
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_barrier
  ds_write_b128 v31, v[44:47]
  ds_write_b128 v31, v[48:51] offset:4352
  ds_write_b128 v31, v[52:55] offset:8704
  ds_write_b128 v31, v[56:59] offset:13056
  ds_write_b128 v31, v[60:63] offset:17408
  ds_write_b128 v31, v[64:67] offset:21760
  s_waitcnt lgkmcnt(0)
  s_barrier
  ds_read_b32 v44, v40
  ds_read_b32 v45, v40 offset:64
  ds_read_b32 v46, v40 offset:128
  ds_read_b32 v47, v40 offset:192
  ds_read_b32 v48, v40 offset:4352
  ds_read_b32 v49, v40 offset:4416
  ds_read_b32 v50, v40 offset:4480
  ds_read_b32 v51, v40 offset:4544
  ds_read_b32 v52, v40 offset:8704
  ds_read_b32 v53, v40 offset:8768
  ds_read_b32 v54, v40 offset:8832
  ds_read_b32 v55, v40 offset:8896
  ds_read_b32 v56, v40 offset:13056
  ds_read_b32 v57, v40 offset:13120
  ds_read_b32 v58, v40 offset:13184
  ds_read_b32 v59, v40 offset:13248
  ds_read_b32 v60, v40 offset:17408
  ds_read_b32 v61, v40 offset:17472
  ds_read_b32 v62, v40 offset:17536
  ds_read_b32 v63, v40 offset:17600
  ds_read_b32 v64, v40 offset:21760
  ds_read_b32 v65, v40 offset:21824
  ds_read_b32 v66, v40 offset:21888
  ds_read_b32 v67, v40 offset:21952
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s31, s30, 4
  v_mov_b32_e32 v4, v18
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v44, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v45, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v46, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v47, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v48, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v49, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v50, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v51, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v52, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v53, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v54, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v55, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v56, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v57, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v58, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v59, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v60, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v61, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v62, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v63, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v64, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v65, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v66, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  global_atomic_add_f32 v4, v67, s[16:17]
  v_add_u32_e64 v4, v4, s31
  s_addk_i32 s33, 0x1
  s_branch label_1098
label_0DEA:
  v_lshrrev_b32_e32 v4, 4, v0
  v_mul_u32_u24_e32 v31, 34, v4
  v_and_b32_e32 v4, 15, v0
  v_mul_lo_u32 v5, 2, v4
  v_add_u32_e32 v31, v5, v31
  s_mul_i32 s31, s24, 0x88
  v_add_u32_e32 v31, s31, v31
  v_lshlrev_b32_e32 v31, 2, v31
  v_and_b32_e32 v4, 31, v0
  v_lshrrev_b32_e32 v5, 1, v4
  v_mul_u32_u24_e32 v40, 34, v5
  v_and_b32_e32 v5, 1, v4
  v_add_u32_e32 v40, v5, v40
  v_lshrrev_b32_e32 v4, 5, v0
  v_mul_u32_u24_e32 v4, 8, v4
  v_add_u32_e32 v40, v4, v40
  s_mul_i32 s31, s24, 2
  v_add_u32_e32 v40, s31, v40
  v_lshlrev_b32_e32 v40, 2, v40
  s_lshr_b32 s31, s51, 2
  s_and_b32 s32, s51, 3
  s_cmp_lt_u32 s24, s32
  s_cselect_b32 s32, 1, 0
  s_add_u32 s51, s31, s32
  s_mov_b32 s33, 0
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_barrier
  v_cmp_u_f32_e64 s[56:57], v44, v44
  v_add3_u32 v8, v44, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v45, v45
  v_add3_u32 v8, v45, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v68, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v46, v46
  v_add3_u32 v8, v46, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v47, v47
  v_add3_u32 v8, v47, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v69, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v48, v48
  v_add3_u32 v8, v48, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v49, v49
  v_add3_u32 v8, v49, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v70, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v50, v50
  v_add3_u32 v8, v50, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v51, v51
  v_add3_u32 v8, v51, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v71, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v52, v52
  v_add3_u32 v8, v52, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v53, v53
  v_add3_u32 v8, v53, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v72, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v54, v54
  v_add3_u32 v8, v54, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v55, v55
  v_add3_u32 v8, v55, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v73, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v56, v56
  v_add3_u32 v8, v56, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v57, v57
  v_add3_u32 v8, v57, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v74, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v58, v58
  v_add3_u32 v8, v58, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v59, v59
  v_add3_u32 v8, v59, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v75, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v60, v60
  v_add3_u32 v8, v60, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v61, v61
  v_add3_u32 v8, v61, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v76, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v62, v62
  v_add3_u32 v8, v62, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v63, v63
  v_add3_u32 v8, v63, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v77, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v64, v64
  v_add3_u32 v8, v64, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v65, v65
  v_add3_u32 v8, v65, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v78, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v66, v66
  v_add3_u32 v8, v66, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v67, v67
  v_add3_u32 v8, v67, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v79, v5, v4, s35
  ds_write_b64 v31, v[68:69]
  ds_write_b64 v31, v[70:71] offset:2176
  ds_write_b64 v31, v[72:73] offset:4352
  ds_write_b64 v31, v[74:75] offset:6528
  ds_write_b64 v31, v[76:77] offset:8704
  ds_write_b64 v31, v[78:79] offset:10880
  s_waitcnt lgkmcnt(0)
  s_barrier
  ds_read_b32 v68, v40
  ds_read_b32 v69, v40 offset:64
  ds_read_b32 v70, v40 offset:2176
  ds_read_b32 v71, v40 offset:2240
  ds_read_b32 v72, v40 offset:4352
  ds_read_b32 v73, v40 offset:4416
  ds_read_b32 v74, v40 offset:6528
  ds_read_b32 v75, v40 offset:6592
  ds_read_b32 v76, v40 offset:8704
  ds_read_b32 v77, v40 offset:8768
  ds_read_b32 v78, v40 offset:10880
  ds_read_b32 v79, v40 offset:10944
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s31, s30, 8
  v_mov_b32_e32 v4, v18
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v68, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v68, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v69, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v69, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v70, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v70, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v71, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v71, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v72, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v72, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v73, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v73, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v74, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v74, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v75, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v75, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v76, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v76, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v77, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v77, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v78, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v78, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, -1
  s_mov_b32 s55, 0
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v79, s[16:17]
  s_addk_i32 s33, 0x1
  s_cmp_lt_i32 s33, s51
  s_cbranch_scc0 label_1098
  s_mov_b32 s54, 0
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  global_atomic_pk_add_bf16 v4, v79, s[16:17]
  s_addk_i32 s33, 0x1
  s_mov_b32 s54, -1
  s_mov_b32 s55, -1
  s_mov_b64 exec, s[54:55]
  v_add_u32_e64 v4, v4, s31
  s_branch label_1098
label_0FD5:
  s_cmp_lt_u32 s50, 1
  s_cbranch_scc0 label_0FE4
  buffer_store_dwordx4 v[44:47], v12, s[16:19], 0 offen
  buffer_store_dwordx4 v[48:51], v13, s[16:19], 0 offen
  buffer_store_dwordx4 v[52:55], v14, s[16:19], 0 offen
  buffer_store_dwordx4 v[56:59], v15, s[16:19], 0 offen
  buffer_store_dwordx4 v[60:63], v16, s[16:19], 0 offen
  buffer_store_dwordx4 v[64:67], v17, s[16:19], 0 offen
  s_branch label_1098
label_0FE4:
  v_cmp_u_f32_e64 s[56:57], v44, v44
  v_add3_u32 v8, v44, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v45, v45
  v_add3_u32 v8, v45, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v68, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v46, v46
  v_add3_u32 v8, v46, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v47, v47
  v_add3_u32 v8, v47, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v69, v5, v4, s35
  buffer_store_dwordx2 v[68:69], v12, s[16:19], 0 offen
  v_cmp_u_f32_e64 s[56:57], v48, v48
  v_add3_u32 v8, v48, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v49, v49
  v_add3_u32 v8, v49, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v70, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v50, v50
  v_add3_u32 v8, v50, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v51, v51
  v_add3_u32 v8, v51, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v71, v5, v4, s35
  buffer_store_dwordx2 v[70:71], v13, s[16:19], 0 offen
  v_cmp_u_f32_e64 s[56:57], v52, v52
  v_add3_u32 v8, v52, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v53, v53
  v_add3_u32 v8, v53, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v72, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v54, v54
  v_add3_u32 v8, v54, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v55, v55
  v_add3_u32 v8, v55, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v73, v5, v4, s35
  buffer_store_dwordx2 v[72:73], v14, s[16:19], 0 offen
  v_cmp_u_f32_e64 s[56:57], v56, v56
  v_add3_u32 v8, v56, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v57, v57
  v_add3_u32 v8, v57, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v74, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v58, v58
  v_add3_u32 v8, v58, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v59, v59
  v_add3_u32 v8, v59, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v75, v5, v4, s35
  buffer_store_dwordx2 v[74:75], v15, s[16:19], 0 offen
  v_cmp_u_f32_e64 s[56:57], v60, v60
  v_add3_u32 v8, v60, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v61, v61
  v_add3_u32 v8, v61, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v76, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v62, v62
  v_add3_u32 v8, v62, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v63, v63
  v_add3_u32 v8, v63, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v77, v5, v4, s35
  buffer_store_dwordx2 v[76:77], v16, s[16:19], 0 offen
  v_cmp_u_f32_e64 s[56:57], v64, v64
  v_add3_u32 v8, v64, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v65, v65
  v_add3_u32 v8, v65, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v78, v5, v4, s35
  v_cmp_u_f32_e64 s[56:57], v66, v66
  v_add3_u32 v8, v66, v11, 1
  v_cndmask_b32_e64 v4, v8, v10, s[56:57]
  v_cmp_u_f32_e64 s[56:57], v67, v67
  v_add3_u32 v8, v67, v11, 1
  v_cndmask_b32_e64 v5, v8, v10, s[56:57]
  v_perm_b32 v79, v5, v4, s35
  buffer_store_dwordx2 v[78:79], v17, s[16:19], 0 offen
label_1098:
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_endpgm

.section	.rodata,"a",@progbits
.p2align	6, 0x0
.amdhsa_kernel gemm
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_group_segment_fixed_size 65536
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 336
  .amdhsa_next_free_vgpr 512 // don't use .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_accum_offset 256 // it's compute_pgm_rsrc3 & 0x3f
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .actual_access:  read_write
        .address_space:  global
        .name:           D
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         8
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           C
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         24
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         40
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           B
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         56
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           alpha
        .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           beta
        .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideD0
        .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideD1
        .offset:         112
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         116
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         120
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         124
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideC0
        .offset:         128
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         132
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         136
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         140
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideC1
        .offset:         144
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         148
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         152
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         156
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideA0
        .offset:         160
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         164
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         168
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         172
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideA1
        .offset:         176
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         180
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         184
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         188
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideB0
        .offset:         192
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         196
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         200
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         204
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideB1
        .offset:         208
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         212
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         216
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         220
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Mdim
        .offset:         224
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         228
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         232
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         236
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Ndim
        .offset:         240
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         244
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         248
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         252
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Kdim
        .offset:         256
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         260
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         264
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         268
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           splitk
        .offset:         272
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         276
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         280
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         284
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           is_out_b16
        .offset:         288
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         292
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         296
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         300
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           Bias
        .offset:         304
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         312
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           add_bias
        .offset:         320
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         324
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         328
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         332
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 336
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .name:           gemm
    .private_segment_fixed_size: 0
    .sgpr_count:     112
    .sgpr_spill_count: 0
    .symbol:         gemm.kd // NOTE: HIP specifically looks for the .kd suffix
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 0
  - 1
...

    .end_amdgpu_metadata
