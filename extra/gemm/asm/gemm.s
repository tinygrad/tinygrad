  // ** global buffers
  s_load_dwordx2  s[28:29], s[0:1], 0x0    // C
  s_load_dwordx4  s[32:35], s[0:1], 0x8    // A, B
  // ** others kernel args
  s_load_dword    s24, s[0:1], 0x18        // N
  s_load_dword    s54, s[0:1], 0x1C        // num work groups
  s_waitcnt lgkmcnt(0)
  // "info"
  s_mov_b32 s51, 1             // gemm_info = 1
  s_mov_b32 s53, 1             // kernel_info0 = 1
  s_mov_b32 s11, 0x40010020    // kernel_info1 = 0x40010020
  // sizes / strides
  s_mov_b32 s25, s24           // sizesFree1 = N
  s_mov_b32 s26, 1             // sizesFree2 = BATCH
  s_mov_b32 s27, s24           // sizesSum0  = K (== N)
  // Strides: major=N, minor=0 (addr = base + idx0*N + idx1*0)
  s_mov_b32 s36, s24           // strideD0
  s_mov_b32 s37, 0             // strideD1
  s_mov_b32 s38, s24           // strideC0
  s_mov_b32 s39, 0             // strideC1
  s_mov_b32 s40, s24           // strideA0
  s_mov_b32 s41, 0             // strideA1
  s_mov_b32 s42, s24           // strideB0
  s_mov_b32 s43, 0             // strideB1
  // ** workgroup mapping
  s_lshr_b32 s52, s51, 30
  s_and_b32 s51, 0x3fffffff, s51
  s_cmp_eq_u32 s52, 0
  s_and_b32 s10, s53, 0xffff0000
  s_lshr_b32 s10, s10, 16
  s_and_b32 s50, s53, 0xffff
  s_mov_b32 s5, s52
  s_mov_b32 m0, 0x20800
  v_mov_b32_e32 v134, v0
  s_lshr_b32 s60, s11, 16
  s_ff1_i32_b32 s60, s60
  s_lshr_b32 s61, s11, 22
  v_and_b32_e32 v5, 63, v134
  v_and_b32_e32 v4, 15, v5
  v_lshlrev_b32_e32 v4, 6, v4
  v_lshlrev_b32_e32 v4, 3, v4
  v_lshrrev_b32_e32 v5, 4, v5
  v_lshl_add_u32 v4, v5, 3, v4
  v_lshrrev_b32_e32 v8, 6, v134
  v_and_b32_e32 v8, 1, v8
  v_lshl_add_u32 v4, v8, 13, v4
  v_and_b32_e32 v6, 63, v134
  v_and_b32_e32 v5, 15, v6
  v_lshlrev_b32_e32 v5, 6, v5
  v_lshlrev_b32_e32 v5, 3, v5
  v_lshrrev_b32_e32 v6, 4, v6
  v_lshl_add_u32 v5, v6, 3, v5
  v_lshrrev_b32_e32 v7, 7, v134
  v_and_b32_e32 v7, 1, v7
  v_lshl_add_u32 v5, v7, 13, v5
  v_lshrrev_b32_e32 v6, 6, v134
  v_lshrrev_b32_e32 v6, 2, v6
  s_mov_b32 s53, 64
  v_mul_lo_u32 v6, s53, v6
  v_add_lshl_u32 v2, v6, v4, 1
  v_lshrrev_b32_e32 v7, 10, v2
  v_lshl_add_u32 v2, v7, 4, v2
  v_lshrrev_b32_e32 v4, 6, v134
  v_lshrrev_b32_e32 v4, 2, v4
  v_mul_lo_u32 v4, s53, v4
  v_add_lshl_u32 v3, v4, v5, 1
  v_lshrrev_b32_e32 v6, 10, v3
  v_lshl_add_u32 v3, v6, 4, v3
  v_add_co_u32_e32 v3, vcc, 0x8200, v3
  v_add_u32_e32 v132, 0x10400, v2
  v_xor_b32_e32 v132, v132, v2
  v_add_u32_e32 v133, 0x10400, v3
  v_xor_b32_e32 v133, v133, v3
  v_lshrrev_b32_e32 v4, 3, v134
  v_and_b32_e32 v5, 7, v134
  v_lshlrev_b32_e32 v5, 3, v5
  v_mov_b32_e32 v8, v5
  v_lshrrev_b32_e32 v6, 3, v134
  v_and_b32_e32 v7, 7, v134
  v_lshlrev_b32_e32 v7, 3, v7
  v_mov_b32_e32 v9, v7
  v_mul_u32_u24_e32 v10, 64, v4
  v_add_lshl_u32 v10, v8, v10, 1
  v_lshrrev_b32_e32 v12, 10, v10
  v_lshl_add_u32 v10, v12, 4, v10
  s_nop 0
  v_readfirstlane_b32 s46, v10
  s_nop 0
  s_add_u32 s48, s46, 0x10400
  s_xor_b32 s48, s48, s46
  v_mul_u32_u24_e32 v10, 64, v6
  v_add_lshl_u32 v10, v9, v10, 1
  v_lshrrev_b32_e32 v12, 10, v10
  v_lshl_add_u32 v10, v12, 4, v10
  v_add_co_u32_e32 v10, vcc, 0x8200, v10
  s_nop 0
  v_readfirstlane_b32 s47, v10
  s_nop 0
  s_add_u32 s49, s47, 0x10400
  s_xor_b32 s49, s49, s47
  v_mov_b32_e32 v12, 0x100
  v_mov_b32_e32 v11, s24
  v_cvt_f32_u32_e32 v10, v12
  v_rcp_iflag_f32_e32 v10, v10
  v_cvt_f32_u32_e32 v13, v11
  v_mul_f32_e32 v10, v10, v13
  v_cvt_u32_f32_e32 v10, v10
  v_mul_u32_u24_e32 v13, v10, v12
  v_sub_u32_e32 v13, v11, v13
  v_cmp_ne_u32_e64 vcc, v13, 0
  v_addc_co_u32_e64 v10, vcc, v10, 0, vcc
  v_mov_b32_e32 v12, 0x100
  v_mov_b32_e32 v11, s25
  v_readfirstlane_b32 s14, v10
  v_cvt_f32_u32_e32 v10, v12
  v_rcp_iflag_f32_e32 v10, v10
  v_cvt_f32_u32_e32 v13, v11
  v_mul_f32_e32 v10, v10, v13
  v_cvt_u32_f32_e32 v10, v10
  v_mul_u32_u24_e32 v13, v10, v12
  v_sub_u32_e32 v13, v11, v13
  v_cmp_ne_u32_e64 vcc, v13, 0
  v_addc_co_u32_e64 v10, vcc, v10, 0, vcc
  s_nop 0
  v_readfirstlane_b32 s15, v10
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s52, s14, s15
  s_and_b32 s53, s50, 0x3fff
  s_mul_i32 s52, s52, s53
  v_cvt_f32_u32_e32 v10, s52
  v_rcp_iflag_f32_e32 v10, v10
  v_cvt_f32_u32_e32 v11, s2
  v_mul_f32_e32 v10, v10, v11
  v_cvt_u32_f32_e32 v10, v10
  v_mul_u32_u24_e64 v11, v10, s52
  v_sub_u32_e32 v11, s2, v11
  v_cmpx_eq_u32_e64 exec, v11, s52
  v_add_u32_e32 v10, 1, v10
  s_mov_b64 exec, -1
  v_cmpx_gt_u32_e64 exec, v11, s52
  v_sub_u32_e64 v10, v10, 1
  s_mov_b64 exec, -1
  v_readfirstlane_b32 s52, v10
  s_mov_b32 s4, s52
  s_mul_i32 s52, s15, s14
  s_mul_i32 s52, s52, s4
  s_mul_i32 s52, s52, s53
  s_sub_u32 s2, s2, s52
  v_cvt_f32_u32_e32 v10, s14
  v_rcp_iflag_f32_e32 v10, v10
  v_cvt_f32_u32_e32 v11, s2
  v_mul_f32_e32 v10, v10, v11
  v_cvt_u32_f32_e32 v10, v10
  v_mul_u32_u24_e64 v11, v10, s14
  v_sub_u32_e32 v11, s2, v11
  v_cmpx_eq_u32_e64 exec, v11, s14
  v_add_u32_e32 v10, 1, v10
  s_mov_b64 exec, -1
  v_cmpx_gt_u32_e64 exec, v11, s14
  v_sub_u32_e64 v10, v10, 1
  s_mov_b64 exec, -1
  v_readfirstlane_b32 s52, v10
  s_mov_b32 s3, s52
  s_mul_i32 s52, s3, s14
  s_sub_u32 s2, s2, s52
  s_sub_u32 s32, s32, 16
  s_subb_u32 s33, s33, 0
  s_sub_u32 s34, s34, 16
  s_subb_u32 s35, s35, 0
  s_and_b32 s84, s50, 0x3fff
  s_mov_b64 s[6:7], 0
  s_mov_b32 s8, 1
  s_mov_b32 s9, 1

  s_sext_i32_i16 s11, s11
  v_mul_lo_u32 v10, s40, v4
  v_add_co_u32_e32 v0, vcc, v5, v10
  v_add_u32_e32 v0, 8, v0
  v_lshlrev_b32_e32 v0, 1, v0
  s_mul_i32 s70, s40, 32
  s_lshl_b32 s70, s70, 1
  s_mul_i32 s71, s40, 64
  s_lshl_b32 s71, s71, 1
  s_mul_i32 s72, s40, 0x60
  s_lshl_b32 s72, s72, 1
  s_mul_i32 s73, s40, 0x80
  s_lshl_b32 s73, s73, 1
  s_mul_i32 s74, s40, 0xa0
  s_lshl_b32 s74, s74, 1
  s_mul_i32 s75, s40, 0xc0
  s_lshl_b32 s75, s75, 1
  s_mul_i32 s76, s40, 0xe0
  s_lshl_b32 s76, s76, 1
  v_mul_lo_u32 v10, s42, v6
  v_add_co_u32_e32 v1, vcc, v7, v10
  v_add_u32_e32 v1, 8, v1
  v_lshlrev_b32_e32 v1, 1, v1
  s_mul_i32 s77, s42, 32
  s_lshl_b32 s77, s77, 1
  s_mul_i32 s78, s42, 64
  s_lshl_b32 s78, s78, 1
  s_mul_i32 s79, s42, 0x60
  s_lshl_b32 s79, s79, 1
  s_mul_i32 s80, s42, 0x80
  s_lshl_b32 s80, s80, 1
  s_mul_i32 s81, s42, 0xa0
  s_lshl_b32 s81, s81, 1
  s_mul_i32 s82, s42, 0xc0
  s_lshl_b32 s82, s82, 1
  s_mul_i32 s83, s42, 0xe0
  s_lshl_b32 s83, s83, 1
  s_mul_hi_u32 s87, s2, 0x100
  s_mul_i32 s86, s2, 0x100
  s_mul_hi_u32 s87, s86, s40
  s_mul_i32 s86, s86, s40
  s_and_b32 s84, s50, 0x8000
  s_cbranch_scc1 skip_offset_A
  s_mul_hi_u32 s85, 64, s6
  s_mul_i32 s84, 64, s6

skip_offset_A:
  s_add_u32 s86, s86, s84
  s_addc_u32 s87, s87, s85
  s_mov_b64 s[60:61], 1
  s_sub_u32 s84, s27, 1
  s_mul_hi_u32 s85, 1, s84
  s_mul_i32 s84, 1, s84
  s_add_u32 s60, s60, s84
  s_addc_u32 s61, s61, s85
  s_sub_u32 s84, s24, 1
  s_mul_hi_u32 s85, s40, s84
  s_mul_i32 s84, s40, s84
  s_add_u32 s60, s60, s84
  s_addc_u32 s61, s61, s85
  s_sub_u32 s60, s60, s86
  s_subb_u32 s61, s61, s87
  s_lshl_b64 s[60:61], s[60:61], 1
  s_add_u32 s60, s60, 16
  s_addc_u32 s61, s61, 0
  s_cmp_eq_u32 s61, 0
  s_cselect_b32 s54, s60, -1
  s_mul_hi_u32 s85, s41, s4
  s_mul_i32 s84, s41, s4
  s_add_u32 s86, s86, s84
  s_addc_u32 s87, s87, s85
  s_lshl_b64 s[86:87], s[86:87], 1
  s_add_u32 s52, s32, s86
  s_addc_u32 s53, s33, s87
  s_mov_b32 s55, 0x20000
  s_mul_hi_u32 s87, s3, 0x100
  s_mul_i32 s86, s3, 0x100
  s_mul_hi_u32 s87, s86, s42
  s_mul_i32 s86, s86, s42
  s_and_b32 s84, s50, 0x8000
  s_cbranch_scc1 skip_offset_B
  s_mul_hi_u32 s85, 64, s6
  s_mul_i32 s84, 64, s6

skip_offset_B:
  s_add_u32 s86, s86, s84
  s_addc_u32 s87, s87, s85
  s_mov_b64 s[62:63], 1
  s_sub_u32 s84, s27, 1
  s_mul_hi_u32 s85, 1, s84
  s_mul_i32 s84, 1, s84
  s_add_u32 s62, s62, s84
  s_addc_u32 s63, s63, s85
  s_sub_u32 s84, s25, 1
  s_mul_hi_u32 s85, s42, s84
  s_mul_i32 s84, s42, s84
  s_add_u32 s62, s62, s84
  s_addc_u32 s63, s63, s85
  s_sub_u32 s62, s62, s86
  s_subb_u32 s63, s63, s87
  s_lshl_b64 s[62:63], s[62:63], 1
  s_add_u32 s62, s62, 16
  s_addc_u32 s63, s63, 0
  s_cmp_eq_u32 s63, 0
  s_cselect_b32 s58, s62, -1
  s_mul_hi_u32 s85, s43, s4
  s_mul_i32 s84, s43, s4
  s_add_u32 s86, s86, s84
  s_addc_u32 s87, s87, s85
  s_lshl_b64 s[86:87], s[86:87], 1
  s_add_u32 s56, s34, s86
  s_addc_u32 s57, s35, s87
  s_mov_b32 s59, 0x20000
  s_and_b32 s85, s50, 0x3fff
  s_mul_i32 s85, s85, 0x80
  s_and_b32 s84, s50, 0x8000
  s_cselect_b32 s68, 0x80, s85
  s_and_b32 s85, s50, 0x3fff
  s_mul_i32 s85, s85, 0x80
  s_and_b32 s84, s50, 0x8000
  s_cselect_b32 s69, 0x80, s85
  s_lshr_b32 s12, s27, 6
  s_and_b32 s84, s50, 0x3fff
  s_mov_b32 s13, s12
  s_and_b32 s86, s10, 0x1f00
  s_lshr_b32 s86, s86, 8
  s_and_b32 s87, s10, 0xe000
  s_and_b32 s10, s10, 0xff
  s_mov_b32 s84, s10
  s_lshl_b32 s85, s84, s86
  s_cmp_ge_u32 s13, s85
  s_sub_u32 s85, s84, 1
  s_cmp_ge_u32 s84, 1
  s_cselect_b32 s51, s85, 0
  s_cmp_eq_u32 s87, 0x2000
  s_and_b32 s51, s51, s84
  s_lshl_b32 s51, s51, s86
  s_mul_hi_i32 s85, s51, s68
  s_mul_i32 s84, s51, s68
  s_mul_hi_i32 s65, s12, s68
  s_mul_i32 s64, s12, s68
  s_sub_u32 s64, s68, s64
  s_subb_u32 s65, 0, s65
  s_add_u32 s52, s52, s84
  s_addc_u32 s53, s53, s85
  s_sub_u32 s60, s60, s84
  s_subb_u32 s61, s61, s85
  s_cmp_eq_u32 s61, 0
  s_cselect_b32 s54, s60, -1
  s_mul_hi_i32 s85, s51, s69
  s_mul_i32 s84, s51, s69
  s_mul_hi_i32 s67, s12, s69
  s_mul_i32 s66, s12, s69
  s_sub_u32 s66, s69, s66
  s_subb_u32 s67, 0, s67
  s_add_u32 s56, s56, s84
  s_addc_u32 s57, s57, s85
  s_sub_u32 s62, s62, s84
  s_subb_u32 s63, s63, s85
  s_cmp_eq_u32 s63, 0
  s_cselect_b32 s58, s62, -1
  s_add_u32 s51, s51, 2
  s_cmp_eq_u32 s12, 0
  s_cbranch_scc1 init_output_buffers
  s_mov_b32 m0, s46
  buffer_load_dwordx4 v0, s[52:55], 0 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s70 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s71 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s72 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s73 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s74 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s75 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s76 offen lds
  s_mov_b32 m0, s47
  buffer_load_dwordx4 v1, s[56:59], 0 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s77 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s78 offen lds
  s_add_u32 m0, m0, 0x1040
  v_accvgpr_write_b32 a0, 0
  v_accvgpr_write_b32 a1, 0
  v_accvgpr_write_b32 a2, 0
  v_accvgpr_write_b32 a3, 0
  v_accvgpr_write_b32 a4, 0
  v_accvgpr_write_b32 a5, 0
  v_accvgpr_write_b32 a6, 0
  v_accvgpr_write_b32 a7, 0
  v_accvgpr_write_b32 a8, 0
  v_accvgpr_write_b32 a9, 0
  v_accvgpr_write_b32 a10, 0
  v_accvgpr_write_b32 a11, 0
  v_accvgpr_write_b32 a12, 0
  v_accvgpr_write_b32 a13, 0
  v_accvgpr_write_b32 a14, 0
  v_accvgpr_write_b32 a15, 0
  v_mov_b64_e32 v[6:7], 0
  v_mov_b64_e32 v[8:9], 0
  v_mfma_f32_32x32x16_bf16 a[16:31], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[32:47], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[48:63], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[64:79], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[80:95], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[96:111], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[112:127], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[128:143], v[6:9], v[6:9], a[0:15]
  buffer_load_dwordx4 v1, s[56:59], s79 offen lds
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_32x32x16_bf16 a[144:159], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[160:175], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[176:191], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[192:207], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[208:223], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[224:239], v[6:9], v[6:9], a[0:15]
  v_mfma_f32_32x32x16_bf16 a[240:255], v[6:9], v[6:9], a[0:15]
  buffer_load_dwordx4 v1, s[56:59], s80 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s81 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s82 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s83 offen lds
  s_add_u32 s86, s12, 1
  s_cmp_eq_u32 s51, s86
  s_cselect_b32 s84, s64, s68
  s_cselect_b32 s85, s65, 0
  s_add_u32 s52, s52, s84
  s_addc_u32 s53, s53, s85
  s_sub_u32 s60, s60, s84
  s_subb_u32 s61, s61, s85
  s_cmp_eq_u32 s61, 0
  s_cselect_b32 s54, s60, -1
  s_add_u32 s86, s12, 1
  s_cmp_eq_u32 s51, s86
  s_cselect_b32 s84, s66, s69
  s_cselect_b32 s85, s67, 0
  s_add_u32 s56, s56, s84
  s_addc_u32 s57, s57, s85
  s_sub_u32 s62, s62, s84
  s_subb_u32 s63, s63, s85
  s_cmp_eq_u32 s63, 0
  s_cselect_b32 s58, s62, -1

init_output_buffers:
  s_mov_b64 s[16:17], s[28:29]
  s_mov_b32 s18, 0x80000000
  s_mov_b32 s19, 0x20000
  s_mov_b64 s[20:21], s[30:31]
  s_mov_b32 s22, 0x80000000
  s_mov_b32 s23, 0x20000
  s_mul_i32 s86, 0x100, s3
  s_mul_hi_u32 s85, s86, s38
  s_mul_i32 s84, s86, s38
  s_lshl_b64 s[84:85], s[84:85], s8
  s_add_u32 s20, s30, s84
  s_addc_u32 s21, s31, s85
  s_mul_hi_u32 s85, s86, s36
  s_mul_i32 s84, s86, s36
  s_lshl_b64 s[84:85], s[84:85], s9
  s_add_u32 s16, s28, s84
  s_addc_u32 s17, s29, s85
  s_mul_hi_u32 s85, s4, s39
  s_mul_i32 s84, s4, s39
  s_lshl_b64 s[84:85], s[84:85], s8
  s_add_u32 s20, s20, s84
  s_addc_u32 s21, s21, s85
  s_mul_hi_u32 s85, s4, s37
  s_mul_i32 s84, s4, s37
  s_lshl_b64 s[84:85], s[84:85], s9
  s_add_u32 s16, s16, s84
  s_addc_u32 s17, s17, s85
  s_mul_hi_u32 s85, s24, s6
  s_mul_i32 s84, s24, s6
  s_sub_u32 s86, s25, 1
  s_mul_i32 s86, s86, s6
  s_mul_hi_u32 s87, s86, s38
  s_mul_i32 s86, s86, s38
  s_add_u32 s84, s84, s86
  s_addc_u32 s85, s85, s87
  s_sub_u32 s86, s26, 1
  s_mul_i32 s86, s86, s6
  s_mul_hi_u32 s87, s86, s39
  s_mul_i32 s86, s86, s39
  s_add_u32 s84, s84, s86
  s_addc_u32 s85, s85, s87
  s_lshl_b64 s[84:85], s[84:85], 2
  s_add_u32 s16, s16, s84
  s_addc_u32 s17, s17, s85
  s_xor_b32 s46, s48, s46
  s_xor_b32 s47, s49, s47
  s_cmp_eq_u32 s12, 1
  s_cbranch_scc1 after_prefetch
  s_mov_b32 m0, s46
  buffer_load_dwordx4 v0, s[52:55], 0 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s70 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s71 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s72 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s73 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s74 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s75 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v0, s[52:55], s76 offen lds
  s_mov_b32 m0, s47
  buffer_load_dwordx4 v1, s[56:59], 0 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s77 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s78 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s79 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s80 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s81 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s82 offen lds
  s_add_u32 m0, m0, 0x1040
  buffer_load_dwordx4 v1, s[56:59], s83 offen lds
  s_xor_b32 s46, s48, s46
  s_xor_b32 s47, s49, s47

after_prefetch:
  s_waitcnt vmcnt(24)
  s_barrier
  ds_read_b128 v[4:7], v2
  ds_read_b128 v[8:11], v2 offset:128
  ds_read_b128 v[12:15], v2 offset:256
  ds_read_b128 v[16:19], v2 offset:384
  ds_read_b128 v[20:23], v2 offset:512
  ds_read_b128 v[24:27], v2 offset:640
  ds_read_b128 v[28:31], v2 offset:768
  ds_read_b128 v[32:35], v2 offset:896
  s_waitcnt vmcnt(16)
  s_barrier
  ds_read_b128 v[68:71], v3
  ds_read_b128 v[72:75], v3 offset:128
  ds_read_b128 v[76:79], v3 offset:256
  ds_read_b128 v[80:83], v3 offset:384
  ds_read_b128 v[84:87], v3 offset:512
  ds_read_b128 v[88:91], v3 offset:640
  ds_read_b128 v[92:95], v3 offset:768
  ds_read_b128 v[96:99], v3 offset:896
  s_waitcnt lgkmcnt(0)
  s_cmp_eq_u32 s12, 1
  s_cbranch_scc1 final_compute
  s_cmp_le_u32 s12, 2
  s_cbranch_scc1 loop_epilogue

main_loop:
  v_mfma_f32_16x16x32_bf16 a[0:3], v[68:71], v[4:7], a[0:3]
  ds_read_b128 v[36:39], v2 offset:64
  v_mfma_f32_16x16x32_bf16 a[4:7], v[68:71], v[8:11], a[4:7]
  s_cmp_eq_u32 s12, s51
  s_cselect_b32 s84, s64, s68
  v_mfma_f32_16x16x32_bf16 a[8:11], v[68:71], v[12:15], a[8:11]
  ds_read_b128 v[40:43], v2 offset:192
  v_mfma_f32_16x16x32_bf16 a[12:15], v[68:71], v[16:19], a[12:15]
  s_cselect_b32 s85, s65, 0
  s_add_u32 s52, s52, s84
  v_mfma_f32_16x16x32_bf16 a[16:19], v[68:71], v[20:23], a[16:19]
  ds_read_b128 v[44:47], v2 offset:320
  v_mfma_f32_16x16x32_bf16 a[20:23], v[68:71], v[24:27], a[20:23]
  s_addc_u32 s53, s53, s85
  s_sub_u32 s60, s60, s84
  v_mfma_f32_16x16x32_bf16 a[24:27], v[68:71], v[28:31], a[24:27]
  ds_read_b128 v[48:51], v2 offset:448
  v_mfma_f32_16x16x32_bf16 a[28:31], v[68:71], v[32:35], a[28:31]
  s_subb_u32 s61, s61, s85
  s_cmp_eq_u32 s61, 0
  v_mfma_f32_16x16x32_bf16 a[32:35], v[72:75], v[4:7], a[32:35]
  ds_read_b128 v[52:55], v2 offset:576
  v_mfma_f32_16x16x32_bf16 a[36:39], v[72:75], v[8:11], a[36:39]
  s_cselect_b32 s54, s60, -1
  s_cmp_eq_u32 s12, s51
  v_mfma_f32_16x16x32_bf16 a[40:43], v[72:75], v[12:15], a[40:43]
  ds_read_b128 v[56:59], v2 offset:704
  v_mfma_f32_16x16x32_bf16 a[44:47], v[72:75], v[16:19], a[44:47]
  s_cselect_b32 s84, s66, s69
  s_cselect_b32 s85, s67, 0
  v_mfma_f32_16x16x32_bf16 a[48:51], v[72:75], v[20:23], a[48:51]
  ds_read_b128 v[60:63], v2 offset:832
  v_mfma_f32_16x16x32_bf16 a[52:55], v[72:75], v[24:27], a[52:55]
  s_add_u32 s56, s56, s84
  s_addc_u32 s57, s57, s85
  v_mfma_f32_16x16x32_bf16 a[56:59], v[72:75], v[28:31], a[56:59]
  ds_read_b128 v[64:67], v2 offset:960
  v_mfma_f32_16x16x32_bf16 a[60:63], v[72:75], v[32:35], a[60:63]
  s_mov_b32 m0, s46
  s_sub_u32 s62, s62, s84
  v_mfma_f32_16x16x32_bf16 a[64:67], v[76:79], v[4:7], a[64:67]
  s_subb_u32 s63, s63, s85
  s_cmp_eq_u32 s63, 0
  v_mfma_f32_16x16x32_bf16 a[68:71], v[76:79], v[8:11], a[68:71]
  s_cselect_b32 s58, s62, -1
  v_mfma_f32_16x16x32_bf16 a[72:75], v[76:79], v[12:15], a[72:75]
  v_mfma_f32_16x16x32_bf16 a[76:79], v[76:79], v[16:19], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[76:79], v[20:23], a[80:83]
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[84:87], v[76:79], v[24:27], a[84:87]
  s_barrier
  v_mfma_f32_16x16x32_bf16 a[88:91], v[76:79], v[28:31], a[88:91]
  buffer_load_dwordx4 v0, s[52:55], 0 offen lds
  v_mfma_f32_16x16x32_bf16 a[92:95], v[76:79], v[32:35], a[92:95]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[96:99], v[80:83], v[4:7], a[96:99]
  ds_read_b128 v[100:103], v3 offset:64
  v_mfma_f32_16x16x32_bf16 a[100:103], v[80:83], v[8:11], a[100:103]
  buffer_load_dwordx4 v0, s[52:55], s70 offen lds
  v_mfma_f32_16x16x32_bf16 a[104:107], v[80:83], v[12:15], a[104:107]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[108:111], v[80:83], v[16:19], a[108:111]
  ds_read_b128 v[104:107], v3 offset:192
  v_mfma_f32_16x16x32_bf16 a[112:115], v[80:83], v[20:23], a[112:115]
  buffer_load_dwordx4 v0, s[52:55], s71 offen lds
  v_mfma_f32_16x16x32_bf16 a[116:119], v[80:83], v[24:27], a[116:119]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[120:123], v[80:83], v[28:31], a[120:123]
  ds_read_b128 v[108:111], v3 offset:320
  v_mfma_f32_16x16x32_bf16 a[124:127], v[80:83], v[32:35], a[124:127]
  buffer_load_dwordx4 v0, s[52:55], s72 offen lds
  v_mfma_f32_16x16x32_bf16 a[128:131], v[84:87], v[4:7], a[128:131]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[132:135], v[84:87], v[8:11], a[132:135]
  ds_read_b128 v[112:115], v3 offset:448
  v_mfma_f32_16x16x32_bf16 a[136:139], v[84:87], v[12:15], a[136:139]
  buffer_load_dwordx4 v0, s[52:55], s73 offen lds
  v_mfma_f32_16x16x32_bf16 a[140:143], v[84:87], v[16:19], a[140:143]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[144:147], v[84:87], v[20:23], a[144:147]
  ds_read_b128 v[116:119], v3 offset:576
  v_mfma_f32_16x16x32_bf16 a[148:151], v[84:87], v[24:27], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[84:87], v[28:31], a[152:155]
  ds_read_b128 v[120:123], v3 offset:704
  v_mfma_f32_16x16x32_bf16 a[156:159], v[84:87], v[32:35], a[156:159]
  v_mfma_f32_16x16x32_bf16 a[160:163], v[88:91], v[4:7], a[160:163]
  ds_read_b128 v[124:127], v3 offset:832
  v_mfma_f32_16x16x32_bf16 a[164:167], v[88:91], v[8:11], a[164:167]
  v_mfma_f32_16x16x32_bf16 a[168:171], v[88:91], v[12:15], a[168:171]
  ds_read_b128 v[128:131], v3 offset:960
  v_mfma_f32_16x16x32_bf16 a[172:175], v[88:91], v[16:19], a[172:175]
  v_mfma_f32_16x16x32_bf16 a[176:179], v[88:91], v[20:23], a[176:179]
  v_mfma_f32_16x16x32_bf16 a[180:183], v[88:91], v[24:27], a[180:183]
  v_mfma_f32_16x16x32_bf16 a[184:187], v[88:91], v[28:31], a[184:187]
  v_mfma_f32_16x16x32_bf16 a[188:191], v[88:91], v[32:35], a[188:191]
  v_mfma_f32_16x16x32_bf16 a[192:195], v[92:95], v[4:7], a[192:195]
  v_mfma_f32_16x16x32_bf16 a[196:199], v[92:95], v[8:11], a[196:199]
  v_mfma_f32_16x16x32_bf16 a[200:203], v[92:95], v[12:15], a[200:203]
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[204:207], v[92:95], v[16:19], a[204:207]
  s_barrier
  v_mfma_f32_16x16x32_bf16 a[208:211], v[92:95], v[20:23], a[208:211]
  buffer_load_dwordx4 v0, s[52:55], s74 offen lds
  v_mfma_f32_16x16x32_bf16 a[212:215], v[92:95], v[24:27], a[212:215]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[216:219], v[92:95], v[28:31], a[216:219]
  v_mfma_f32_16x16x32_bf16 a[220:223], v[92:95], v[32:35], a[220:223]
  buffer_load_dwordx4 v0, s[52:55], s75 offen lds
  v_mfma_f32_16x16x32_bf16 a[224:227], v[96:99], v[4:7], a[224:227]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[228:231], v[96:99], v[8:11], a[228:231]
  v_mfma_f32_16x16x32_bf16 a[232:235], v[96:99], v[12:15], a[232:235]
  buffer_load_dwordx4 v0, s[52:55], s76 offen lds
  v_mfma_f32_16x16x32_bf16 a[236:239], v[96:99], v[16:19], a[236:239]
  s_mov_b32 m0, s47
  v_mfma_f32_16x16x32_bf16 a[240:243], v[96:99], v[20:23], a[240:243]
  v_mfma_f32_16x16x32_bf16 a[244:247], v[96:99], v[24:27], a[244:247]
  buffer_load_dwordx4 v1, s[56:59], 0 offen lds
  v_mfma_f32_16x16x32_bf16 a[248:251], v[96:99], v[28:31], a[248:251]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[32:35], a[252:255]
  v_mfma_f32_16x16x32_bf16 a[0:3], v[100:103], v[36:39], a[0:3]
  buffer_load_dwordx4 v1, s[56:59], s77 offen lds
  v_mfma_f32_16x16x32_bf16 a[4:7], v[100:103], v[40:43], a[4:7]
  s_add_u32 m0, m0, 0x1040
  s_xor_b32 s46, s48, s46
  v_mfma_f32_16x16x32_bf16 a[8:11], v[100:103], v[44:47], a[8:11]
  v_mfma_f32_16x16x32_bf16 a[12:15], v[100:103], v[48:51], a[12:15]
  v_mfma_f32_16x16x32_bf16 a[16:19], v[100:103], v[52:55], a[16:19]
  v_mfma_f32_16x16x32_bf16 a[20:23], v[100:103], v[56:59], a[20:23]
  v_mfma_f32_16x16x32_bf16 a[24:27], v[100:103], v[60:63], a[24:27]
  v_mfma_f32_16x16x32_bf16 a[28:31], v[100:103], v[64:67], a[28:31]
  v_mfma_f32_16x16x32_bf16 a[32:35], v[104:107], v[36:39], a[32:35]
  v_mfma_f32_16x16x32_bf16 a[36:39], v[104:107], v[40:43], a[36:39]
  v_mfma_f32_16x16x32_bf16 a[40:43], v[104:107], v[44:47], a[40:43]
  v_mfma_f32_16x16x32_bf16 a[44:47], v[104:107], v[48:51], a[44:47]
  v_mfma_f32_16x16x32_bf16 a[48:51], v[104:107], v[52:55], a[48:51]
  v_mfma_f32_16x16x32_bf16 a[52:55], v[104:107], v[56:59], a[52:55]
  v_mfma_f32_16x16x32_bf16 a[56:59], v[104:107], v[60:63], a[56:59]
  v_mfma_f32_16x16x32_bf16 a[60:63], v[104:107], v[64:67], a[60:63]
  v_mfma_f32_16x16x32_bf16 a[64:67], v[108:111], v[36:39], a[64:67]
  v_mfma_f32_16x16x32_bf16 a[68:71], v[108:111], v[40:43], a[68:71]
  v_mfma_f32_16x16x32_bf16 a[72:75], v[108:111], v[44:47], a[72:75]
  v_mfma_f32_16x16x32_bf16 a[76:79], v[108:111], v[48:51], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[108:111], v[52:55], a[80:83]
  v_xor_b32_e32 v2, v132, v2
  v_xor_b32_e32 v3, v133, v3
  v_mfma_f32_16x16x32_bf16 a[84:87], v[108:111], v[56:59], a[84:87]
  buffer_load_dwordx4 v1, s[56:59], s78 offen lds
  v_mfma_f32_16x16x32_bf16 a[88:91], v[108:111], v[60:63], a[88:91]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[92:95], v[108:111], v[64:67], a[92:95]
  buffer_load_dwordx4 v1, s[56:59], s79 offen lds
  v_mfma_f32_16x16x32_bf16 a[96:99], v[112:115], v[36:39], a[96:99]
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[100:103], v[112:115], v[40:43], a[100:103]
  buffer_load_dwordx4 v1, s[56:59], s80 offen lds
  v_mfma_f32_16x16x32_bf16 a[104:107], v[112:115], v[44:47], a[104:107]
  v_mfma_f32_16x16x32_bf16 a[108:111], v[112:115], v[48:51], a[108:111]
  s_waitcnt vmcnt(13)
  v_mfma_f32_16x16x32_bf16 a[112:115], v[112:115], v[52:55], a[112:115]
  s_barrier
  v_mfma_f32_16x16x32_bf16 a[116:119], v[112:115], v[56:59], a[116:119]
  ds_read_b128 v[4:7], v2
  v_mfma_f32_16x16x32_bf16 a[120:123], v[112:115], v[60:63], a[120:123]
  ds_read_b128 v[8:11], v2 offset:128
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[124:127], v[112:115], v[64:67], a[124:127]
  ds_read_b128 v[12:15], v2 offset:256
  v_mfma_f32_16x16x32_bf16 a[128:131], v[116:119], v[36:39], a[128:131]
  buffer_load_dwordx4 v1, s[56:59], s81 offen lds
  v_mfma_f32_16x16x32_bf16 a[132:135], v[116:119], v[40:43], a[132:135]
  ds_read_b128 v[16:19], v2 offset:384
  v_mfma_f32_16x16x32_bf16 a[136:139], v[116:119], v[44:47], a[136:139]
  ds_read_b128 v[20:23], v2 offset:512
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[140:143], v[116:119], v[48:51], a[140:143]
  v_mfma_f32_16x16x32_bf16 a[144:147], v[116:119], v[52:55], a[144:147]
  buffer_load_dwordx4 v1, s[56:59], s82 offen lds
  v_mfma_f32_16x16x32_bf16 a[148:151], v[116:119], v[56:59], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[116:119], v[60:63], a[152:155]
  ds_read_b128 v[24:27], v2 offset:640
  s_add_u32 m0, m0, 0x1040
  v_mfma_f32_16x16x32_bf16 a[156:159], v[116:119], v[64:67], a[156:159]
  ds_read_b128 v[28:31], v2 offset:768
  v_mfma_f32_16x16x32_bf16 a[160:163], v[120:123], v[36:39], a[160:163]
  ds_read_b128 v[32:35], v2 offset:896
  v_mfma_f32_16x16x32_bf16 a[164:167], v[120:123], v[40:43], a[164:167]
  ds_read_b128 v[68:71], v3
  v_mfma_f32_16x16x32_bf16 a[168:171], v[120:123], v[44:47], a[168:171]
  ds_read_b128 v[72:75], v3 offset:128
  v_mfma_f32_16x16x32_bf16 a[172:175], v[120:123], v[48:51], a[172:175]
  v_mfma_f32_16x16x32_bf16 a[176:179], v[120:123], v[52:55], a[176:179]
  v_mfma_f32_16x16x32_bf16 a[180:183], v[120:123], v[56:59], a[180:183]
  ds_read_b128 v[76:79], v3 offset:256
  v_mfma_f32_16x16x32_bf16 a[184:187], v[120:123], v[60:63], a[184:187]
  v_mfma_f32_16x16x32_bf16 a[188:191], v[120:123], v[64:67], a[188:191]
  v_mfma_f32_16x16x32_bf16 a[192:195], v[124:127], v[36:39], a[192:195]
  ds_read_b128 v[80:83], v3 offset:384
  v_mfma_f32_16x16x32_bf16 a[196:199], v[124:127], v[40:43], a[196:199]
  v_mfma_f32_16x16x32_bf16 a[200:203], v[124:127], v[44:47], a[200:203]
  ds_read_b128 v[84:87], v3 offset:512
  v_mfma_f32_16x16x32_bf16 a[204:207], v[124:127], v[48:51], a[204:207]
  v_mfma_f32_16x16x32_bf16 a[208:211], v[124:127], v[52:55], a[208:211]
  v_mfma_f32_16x16x32_bf16 a[212:215], v[124:127], v[56:59], a[212:215]
  ds_read_b128 v[88:91], v3 offset:640
  v_mfma_f32_16x16x32_bf16 a[216:219], v[124:127], v[60:63], a[216:219]
  v_mfma_f32_16x16x32_bf16 a[220:223], v[124:127], v[64:67], a[220:223]
  v_mfma_f32_16x16x32_bf16 a[224:227], v[128:131], v[36:39], a[224:227]
  ds_read_b128 v[92:95], v3 offset:768
  v_mfma_f32_16x16x32_bf16 a[228:231], v[128:131], v[40:43], a[228:231]
  v_mfma_f32_16x16x32_bf16 a[232:235], v[128:131], v[44:47], a[232:235]
  v_mfma_f32_16x16x32_bf16 a[236:239], v[128:131], v[48:51], a[236:239]
  ds_read_b128 v[96:99], v3 offset:896
  v_mfma_f32_16x16x32_bf16 a[240:243], v[128:131], v[52:55], a[240:243]
  buffer_load_dwordx4 v1, s[56:59], s83 offen lds
  v_mfma_f32_16x16x32_bf16 a[244:247], v[128:131], v[56:59], a[244:247]
  s_xor_b32 s47, s49, s47
  s_sub_u32 s12, s12, 1
  v_mfma_f32_16x16x32_bf16 a[248:251], v[128:131], v[60:63], a[248:251]
  s_cmp_eq_i32 s12, 2
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[252:255], v[128:131], v[64:67], a[252:255]
  s_cbranch_scc0 main_loop

loop_epilogue:
  v_mfma_f32_16x16x32_bf16 a[0:3], v[68:71], v[4:7], a[0:3]
  ds_read_b128 v[36:39], v2 offset:64
  v_mfma_f32_16x16x32_bf16 a[4:7], v[68:71], v[8:11], a[4:7]
  v_mfma_f32_16x16x32_bf16 a[8:11], v[68:71], v[12:15], a[8:11]
  ds_read_b128 v[100:103], v3 offset:64
  v_mfma_f32_16x16x32_bf16 a[12:15], v[68:71], v[16:19], a[12:15]
  v_mfma_f32_16x16x32_bf16 a[16:19], v[68:71], v[20:23], a[16:19]
  ds_read_b128 v[40:43], v2 offset:192
  v_mfma_f32_16x16x32_bf16 a[20:23], v[68:71], v[24:27], a[20:23]
  v_mfma_f32_16x16x32_bf16 a[24:27], v[68:71], v[28:31], a[24:27]
  ds_read_b128 v[44:47], v2 offset:320
  v_mfma_f32_16x16x32_bf16 a[28:31], v[68:71], v[32:35], a[28:31]
  v_mfma_f32_16x16x32_bf16 a[32:35], v[72:75], v[4:7], a[32:35]
  ds_read_b128 v[48:51], v2 offset:448
  v_mfma_f32_16x16x32_bf16 a[36:39], v[72:75], v[8:11], a[36:39]
  v_mfma_f32_16x16x32_bf16 a[40:43], v[72:75], v[12:15], a[40:43]
  ds_read_b128 v[52:55], v2 offset:576
  v_mfma_f32_16x16x32_bf16 a[44:47], v[72:75], v[16:19], a[44:47]
  v_mfma_f32_16x16x32_bf16 a[48:51], v[72:75], v[20:23], a[48:51]
  ds_read_b128 v[56:59], v2 offset:704
  v_mfma_f32_16x16x32_bf16 a[52:55], v[72:75], v[24:27], a[52:55]
  v_mfma_f32_16x16x32_bf16 a[56:59], v[72:75], v[28:31], a[56:59]
  ds_read_b128 v[60:63], v2 offset:832
  v_mfma_f32_16x16x32_bf16 a[60:63], v[72:75], v[32:35], a[60:63]
  v_mfma_f32_16x16x32_bf16 a[64:67], v[76:79], v[4:7], a[64:67]
  ds_read_b128 v[64:67], v2 offset:960
  v_mfma_f32_16x16x32_bf16 a[68:71], v[76:79], v[8:11], a[68:71]
  v_mfma_f32_16x16x32_bf16 a[72:75], v[76:79], v[12:15], a[72:75]
  ds_read_b128 v[104:107], v3 offset:192
  v_mfma_f32_16x16x32_bf16 a[76:79], v[76:79], v[16:19], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[76:79], v[20:23], a[80:83]
  ds_read_b128 v[108:111], v3 offset:320
  v_mfma_f32_16x16x32_bf16 a[84:87], v[76:79], v[24:27], a[84:87]
  v_mfma_f32_16x16x32_bf16 a[88:91], v[76:79], v[28:31], a[88:91]
  ds_read_b128 v[112:115], v3 offset:448
  v_mfma_f32_16x16x32_bf16 a[92:95], v[76:79], v[32:35], a[92:95]
  v_mfma_f32_16x16x32_bf16 a[96:99], v[80:83], v[4:7], a[96:99]
  ds_read_b128 v[116:119], v3 offset:576
  v_mfma_f32_16x16x32_bf16 a[100:103], v[80:83], v[8:11], a[100:103]
  v_mfma_f32_16x16x32_bf16 a[104:107], v[80:83], v[12:15], a[104:107]
  ds_read_b128 v[120:123], v3 offset:704
  v_mfma_f32_16x16x32_bf16 a[108:111], v[80:83], v[16:19], a[108:111]
  v_mfma_f32_16x16x32_bf16 a[112:115], v[80:83], v[20:23], a[112:115]
  ds_read_b128 v[124:127], v3 offset:832
  v_mfma_f32_16x16x32_bf16 a[116:119], v[80:83], v[24:27], a[116:119]
  v_mfma_f32_16x16x32_bf16 a[120:123], v[80:83], v[28:31], a[120:123]
  ds_read_b128 v[128:131], v3 offset:960
  v_mfma_f32_16x16x32_bf16 a[124:127], v[80:83], v[32:35], a[124:127]
  v_mfma_f32_16x16x32_bf16 a[128:131], v[84:87], v[4:7], a[128:131]
  v_mfma_f32_16x16x32_bf16 a[132:135], v[84:87], v[8:11], a[132:135]
  v_mfma_f32_16x16x32_bf16 a[136:139], v[84:87], v[12:15], a[136:139]
  v_mfma_f32_16x16x32_bf16 a[140:143], v[84:87], v[16:19], a[140:143]
  v_mfma_f32_16x16x32_bf16 a[144:147], v[84:87], v[20:23], a[144:147]
  v_mfma_f32_16x16x32_bf16 a[148:151], v[84:87], v[24:27], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[84:87], v[28:31], a[152:155]
  v_mfma_f32_16x16x32_bf16 a[156:159], v[84:87], v[32:35], a[156:159]
  v_mfma_f32_16x16x32_bf16 a[160:163], v[88:91], v[4:7], a[160:163]
  v_mfma_f32_16x16x32_bf16 a[164:167], v[88:91], v[8:11], a[164:167]
  v_mfma_f32_16x16x32_bf16 a[168:171], v[88:91], v[12:15], a[168:171]
  v_mfma_f32_16x16x32_bf16 a[172:175], v[88:91], v[16:19], a[172:175]
  v_mfma_f32_16x16x32_bf16 a[176:179], v[88:91], v[20:23], a[176:179]
  v_mfma_f32_16x16x32_bf16 a[180:183], v[88:91], v[24:27], a[180:183]
  v_mfma_f32_16x16x32_bf16 a[184:187], v[88:91], v[28:31], a[184:187]
  v_mfma_f32_16x16x32_bf16 a[188:191], v[88:91], v[32:35], a[188:191]
  v_mfma_f32_16x16x32_bf16 a[192:195], v[92:95], v[4:7], a[192:195]
  v_mfma_f32_16x16x32_bf16 a[196:199], v[92:95], v[8:11], a[196:199]
  v_mfma_f32_16x16x32_bf16 a[200:203], v[92:95], v[12:15], a[200:203]
  v_mfma_f32_16x16x32_bf16 a[204:207], v[92:95], v[16:19], a[204:207]
  v_mfma_f32_16x16x32_bf16 a[208:211], v[92:95], v[20:23], a[208:211]
  v_mfma_f32_16x16x32_bf16 a[212:215], v[92:95], v[24:27], a[212:215]
  v_mfma_f32_16x16x32_bf16 a[216:219], v[92:95], v[28:31], a[216:219]
  v_mfma_f32_16x16x32_bf16 a[220:223], v[92:95], v[32:35], a[220:223]
  v_mfma_f32_16x16x32_bf16 a[224:227], v[96:99], v[4:7], a[224:227]
  v_mfma_f32_16x16x32_bf16 a[228:231], v[96:99], v[8:11], a[228:231]
  v_mfma_f32_16x16x32_bf16 a[232:235], v[96:99], v[12:15], a[232:235]
  v_mfma_f32_16x16x32_bf16 a[236:239], v[96:99], v[16:19], a[236:239]
  v_mfma_f32_16x16x32_bf16 a[240:243], v[96:99], v[20:23], a[240:243]
  v_mfma_f32_16x16x32_bf16 a[244:247], v[96:99], v[24:27], a[244:247]
  v_mfma_f32_16x16x32_bf16 a[248:251], v[96:99], v[28:31], a[248:251]
  v_xor_b32_e32 v2, v132, v2
  v_xor_b32_e32 v3, v133, v3
  v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[32:35], a[252:255]
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[0:3], v[100:103], v[36:39], a[0:3]
  v_mfma_f32_16x16x32_bf16 a[4:7], v[100:103], v[40:43], a[4:7]
  v_mfma_f32_16x16x32_bf16 a[8:11], v[100:103], v[44:47], a[8:11]
  v_mfma_f32_16x16x32_bf16 a[12:15], v[100:103], v[48:51], a[12:15]
  v_mfma_f32_16x16x32_bf16 a[16:19], v[100:103], v[52:55], a[16:19]
  v_mfma_f32_16x16x32_bf16 a[20:23], v[100:103], v[56:59], a[20:23]
  v_mfma_f32_16x16x32_bf16 a[24:27], v[100:103], v[60:63], a[24:27]
  v_mfma_f32_16x16x32_bf16 a[28:31], v[100:103], v[64:67], a[28:31]
  v_mfma_f32_16x16x32_bf16 a[32:35], v[104:107], v[36:39], a[32:35]
  v_mfma_f32_16x16x32_bf16 a[36:39], v[104:107], v[40:43], a[36:39]
  v_mfma_f32_16x16x32_bf16 a[40:43], v[104:107], v[44:47], a[40:43]
  v_mfma_f32_16x16x32_bf16 a[44:47], v[104:107], v[48:51], a[44:47]
  v_mfma_f32_16x16x32_bf16 a[48:51], v[104:107], v[52:55], a[48:51]
  v_mfma_f32_16x16x32_bf16 a[52:55], v[104:107], v[56:59], a[52:55]
  v_mfma_f32_16x16x32_bf16 a[56:59], v[104:107], v[60:63], a[56:59]
  v_mfma_f32_16x16x32_bf16 a[60:63], v[104:107], v[64:67], a[60:63]
  v_mfma_f32_16x16x32_bf16 a[64:67], v[108:111], v[36:39], a[64:67]
  v_mfma_f32_16x16x32_bf16 a[68:71], v[108:111], v[40:43], a[68:71]
  v_mfma_f32_16x16x32_bf16 a[72:75], v[108:111], v[44:47], a[72:75]
  v_mfma_f32_16x16x32_bf16 a[76:79], v[108:111], v[48:51], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[108:111], v[52:55], a[80:83]
  v_mfma_f32_16x16x32_bf16 a[84:87], v[108:111], v[56:59], a[84:87]
  v_mfma_f32_16x16x32_bf16 a[88:91], v[108:111], v[60:63], a[88:91]
  v_mfma_f32_16x16x32_bf16 a[92:95], v[108:111], v[64:67], a[92:95]
  v_mfma_f32_16x16x32_bf16 a[96:99], v[112:115], v[36:39], a[96:99]
  v_mfma_f32_16x16x32_bf16 a[100:103], v[112:115], v[40:43], a[100:103]
  v_mfma_f32_16x16x32_bf16 a[104:107], v[112:115], v[44:47], a[104:107]
  v_mfma_f32_16x16x32_bf16 a[108:111], v[112:115], v[48:51], a[108:111]
  v_mfma_f32_16x16x32_bf16 a[112:115], v[112:115], v[52:55], a[112:115]
  v_mfma_f32_16x16x32_bf16 a[116:119], v[112:115], v[56:59], a[116:119]
  v_mfma_f32_16x16x32_bf16 a[120:123], v[112:115], v[60:63], a[120:123]
  v_mfma_f32_16x16x32_bf16 a[124:127], v[112:115], v[64:67], a[124:127]
  v_mfma_f32_16x16x32_bf16 a[128:131], v[116:119], v[36:39], a[128:131]
  v_mfma_f32_16x16x32_bf16 a[132:135], v[116:119], v[40:43], a[132:135]
  v_mfma_f32_16x16x32_bf16 a[136:139], v[116:119], v[44:47], a[136:139]
  v_mfma_f32_16x16x32_bf16 a[140:143], v[116:119], v[48:51], a[140:143]
  v_mfma_f32_16x16x32_bf16 a[144:147], v[116:119], v[52:55], a[144:147]
  v_mfma_f32_16x16x32_bf16 a[148:151], v[116:119], v[56:59], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[116:119], v[60:63], a[152:155]
  v_mfma_f32_16x16x32_bf16 a[156:159], v[116:119], v[64:67], a[156:159]
  v_mfma_f32_16x16x32_bf16 a[160:163], v[120:123], v[36:39], a[160:163]
  v_mfma_f32_16x16x32_bf16 a[164:167], v[120:123], v[40:43], a[164:167]
  s_waitcnt vmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[168:171], v[120:123], v[44:47], a[168:171]
  s_barrier
  v_mfma_f32_16x16x32_bf16 a[172:175], v[120:123], v[48:51], a[172:175]
  ds_read_b128 v[4:7], v2
  v_mfma_f32_16x16x32_bf16 a[176:179], v[120:123], v[52:55], a[176:179]
  ds_read_b128 v[68:71], v3
  v_mfma_f32_16x16x32_bf16 a[180:183], v[120:123], v[56:59], a[180:183]
  ds_read_b128 v[8:11], v2 offset:128
  v_mfma_f32_16x16x32_bf16 a[184:187], v[120:123], v[60:63], a[184:187]
  ds_read_b128 v[12:15], v2 offset:256
  v_mfma_f32_16x16x32_bf16 a[188:191], v[120:123], v[64:67], a[188:191]
  ds_read_b128 v[16:19], v2 offset:384
  v_mfma_f32_16x16x32_bf16 a[192:195], v[124:127], v[36:39], a[192:195]
  ds_read_b128 v[20:23], v2 offset:512
  v_mfma_f32_16x16x32_bf16 a[196:199], v[124:127], v[40:43], a[196:199]
  ds_read_b128 v[24:27], v2 offset:640
  v_mfma_f32_16x16x32_bf16 a[200:203], v[124:127], v[44:47], a[200:203]
  ds_read_b128 v[28:31], v2 offset:768
  v_mfma_f32_16x16x32_bf16 a[204:207], v[124:127], v[48:51], a[204:207]
  ds_read_b128 v[32:35], v2 offset:896
  v_mfma_f32_16x16x32_bf16 a[208:211], v[124:127], v[52:55], a[208:211]
  ds_read_b128 v[72:75], v3 offset:128
  v_mfma_f32_16x16x32_bf16 a[212:215], v[124:127], v[56:59], a[212:215]
  ds_read_b128 v[76:79], v3 offset:256
  v_mfma_f32_16x16x32_bf16 a[216:219], v[124:127], v[60:63], a[216:219]
  ds_read_b128 v[80:83], v3 offset:384
  v_mfma_f32_16x16x32_bf16 a[220:223], v[124:127], v[64:67], a[220:223]
  ds_read_b128 v[84:87], v3 offset:512
  v_mfma_f32_16x16x32_bf16 a[224:227], v[128:131], v[36:39], a[224:227]
  ds_read_b128 v[88:91], v3 offset:640
  v_mfma_f32_16x16x32_bf16 a[228:231], v[128:131], v[40:43], a[228:231]
  ds_read_b128 v[92:95], v3 offset:768
  v_mfma_f32_16x16x32_bf16 a[232:235], v[128:131], v[44:47], a[232:235]
  ds_read_b128 v[96:99], v3 offset:896
  v_mfma_f32_16x16x32_bf16 a[236:239], v[128:131], v[48:51], a[236:239]
  v_mfma_f32_16x16x32_bf16 a[240:243], v[128:131], v[52:55], a[240:243]
  v_mfma_f32_16x16x32_bf16 a[244:247], v[128:131], v[56:59], a[244:247]
  v_mfma_f32_16x16x32_bf16 a[248:251], v[128:131], v[60:63], a[248:251]
  v_mfma_f32_16x16x32_bf16 a[252:255], v[128:131], v[64:67], a[252:255]

final_compute:
  s_and_b32 s8, s50, 0x3fff
  s_and_b32 s84, 0xff, s24
  s_add_u32 s85, -1, s14
  s_cmp_ge_u32 s2, s85
  s_cselect_b32 s84, s84, 0
  s_and_b32 s84, 0xff, s25
  s_add_u32 s85, -1, s15
  s_cmp_ge_u32 s3, s85
  s_cselect_b32 s84, s84, 0
  v_mfma_f32_16x16x32_bf16 a[0:3], v[68:71], v[4:7], a[0:3]
  ds_read_b128 v[36:39], v2 offset:64
  v_mfma_f32_16x16x32_bf16 a[4:7], v[68:71], v[8:11], a[4:7]
  v_mfma_f32_16x16x32_bf16 a[8:11], v[68:71], v[12:15], a[8:11]
  ds_read_b128 v[100:103], v3 offset:64
  v_mfma_f32_16x16x32_bf16 a[12:15], v[68:71], v[16:19], a[12:15]
  v_mfma_f32_16x16x32_bf16 a[16:19], v[68:71], v[20:23], a[16:19]
  ds_read_b128 v[40:43], v2 offset:192
  v_mfma_f32_16x16x32_bf16 a[20:23], v[68:71], v[24:27], a[20:23]
  v_mfma_f32_16x16x32_bf16 a[24:27], v[68:71], v[28:31], a[24:27]
  ds_read_b128 v[44:47], v2 offset:320
  v_mfma_f32_16x16x32_bf16 a[28:31], v[68:71], v[32:35], a[28:31]
  v_mfma_f32_16x16x32_bf16 a[32:35], v[72:75], v[4:7], a[32:35]
  ds_read_b128 v[48:51], v2 offset:448
  v_mfma_f32_16x16x32_bf16 a[36:39], v[72:75], v[8:11], a[36:39]
  v_mfma_f32_16x16x32_bf16 a[40:43], v[72:75], v[12:15], a[40:43]
  ds_read_b128 v[52:55], v2 offset:576
  v_mfma_f32_16x16x32_bf16 a[44:47], v[72:75], v[16:19], a[44:47]
  v_mfma_f32_16x16x32_bf16 a[48:51], v[72:75], v[20:23], a[48:51]
  ds_read_b128 v[56:59], v2 offset:704
  v_mfma_f32_16x16x32_bf16 a[52:55], v[72:75], v[24:27], a[52:55]
  v_mfma_f32_16x16x32_bf16 a[56:59], v[72:75], v[28:31], a[56:59]
  ds_read_b128 v[60:63], v2 offset:832
  v_mfma_f32_16x16x32_bf16 a[60:63], v[72:75], v[32:35], a[60:63]
  v_mfma_f32_16x16x32_bf16 a[64:67], v[76:79], v[4:7], a[64:67]
  ds_read_b128 v[64:67], v2 offset:960
  v_mfma_f32_16x16x32_bf16 a[68:71], v[76:79], v[8:11], a[68:71]
  v_mfma_f32_16x16x32_bf16 a[72:75], v[76:79], v[12:15], a[72:75]
  ds_read_b128 v[104:107], v3 offset:192
  v_mfma_f32_16x16x32_bf16 a[76:79], v[76:79], v[16:19], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[76:79], v[20:23], a[80:83]
  ds_read_b128 v[108:111], v3 offset:320
  v_mfma_f32_16x16x32_bf16 a[84:87], v[76:79], v[24:27], a[84:87]
  v_mfma_f32_16x16x32_bf16 a[88:91], v[76:79], v[28:31], a[88:91]
  ds_read_b128 v[112:115], v3 offset:448
  v_mfma_f32_16x16x32_bf16 a[92:95], v[76:79], v[32:35], a[92:95]
  v_mfma_f32_16x16x32_bf16 a[96:99], v[80:83], v[4:7], a[96:99]
  ds_read_b128 v[116:119], v3 offset:576
  v_mfma_f32_16x16x32_bf16 a[100:103], v[80:83], v[8:11], a[100:103]
  v_mfma_f32_16x16x32_bf16 a[104:107], v[80:83], v[12:15], a[104:107]
  ds_read_b128 v[120:123], v3 offset:704
  v_mfma_f32_16x16x32_bf16 a[108:111], v[80:83], v[16:19], a[108:111]
  v_mfma_f32_16x16x32_bf16 a[112:115], v[80:83], v[20:23], a[112:115]
  ds_read_b128 v[124:127], v3 offset:832
  v_mfma_f32_16x16x32_bf16 a[116:119], v[80:83], v[24:27], a[116:119]
  v_mfma_f32_16x16x32_bf16 a[120:123], v[80:83], v[28:31], a[120:123]
  ds_read_b128 v[128:131], v3 offset:960
  v_mfma_f32_16x16x32_bf16 a[124:127], v[80:83], v[32:35], a[124:127]
  v_mfma_f32_16x16x32_bf16 a[128:131], v[84:87], v[4:7], a[128:131]
  v_mfma_f32_16x16x32_bf16 a[132:135], v[84:87], v[8:11], a[132:135]
  v_mfma_f32_16x16x32_bf16 a[136:139], v[84:87], v[12:15], a[136:139]
  v_mfma_f32_16x16x32_bf16 a[140:143], v[84:87], v[16:19], a[140:143]
  v_mfma_f32_16x16x32_bf16 a[144:147], v[84:87], v[20:23], a[144:147]
  v_mfma_f32_16x16x32_bf16 a[148:151], v[84:87], v[24:27], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[84:87], v[28:31], a[152:155]
  v_mfma_f32_16x16x32_bf16 a[156:159], v[84:87], v[32:35], a[156:159]
  v_mfma_f32_16x16x32_bf16 a[160:163], v[88:91], v[4:7], a[160:163]
  v_mfma_f32_16x16x32_bf16 a[164:167], v[88:91], v[8:11], a[164:167]
  v_mfma_f32_16x16x32_bf16 a[168:171], v[88:91], v[12:15], a[168:171]
  v_mfma_f32_16x16x32_bf16 a[172:175], v[88:91], v[16:19], a[172:175]
  v_mfma_f32_16x16x32_bf16 a[176:179], v[88:91], v[20:23], a[176:179]
  v_mfma_f32_16x16x32_bf16 a[180:183], v[88:91], v[24:27], a[180:183]
  v_mfma_f32_16x16x32_bf16 a[184:187], v[88:91], v[28:31], a[184:187]
  v_mfma_f32_16x16x32_bf16 a[188:191], v[88:91], v[32:35], a[188:191]
  v_mfma_f32_16x16x32_bf16 a[192:195], v[92:95], v[4:7], a[192:195]
  v_mfma_f32_16x16x32_bf16 a[196:199], v[92:95], v[8:11], a[196:199]
  v_mfma_f32_16x16x32_bf16 a[200:203], v[92:95], v[12:15], a[200:203]
  v_mfma_f32_16x16x32_bf16 a[204:207], v[92:95], v[16:19], a[204:207]
  v_mfma_f32_16x16x32_bf16 a[208:211], v[92:95], v[20:23], a[208:211]
  v_mfma_f32_16x16x32_bf16 a[212:215], v[92:95], v[24:27], a[212:215]
  v_mfma_f32_16x16x32_bf16 a[216:219], v[92:95], v[28:31], a[216:219]
  v_mfma_f32_16x16x32_bf16 a[220:223], v[92:95], v[32:35], a[220:223]
  v_mfma_f32_16x16x32_bf16 a[224:227], v[96:99], v[4:7], a[224:227]
  v_mfma_f32_16x16x32_bf16 a[228:231], v[96:99], v[8:11], a[228:231]
  v_mfma_f32_16x16x32_bf16 a[232:235], v[96:99], v[12:15], a[232:235]
  v_mfma_f32_16x16x32_bf16 a[236:239], v[96:99], v[16:19], a[236:239]
  v_mfma_f32_16x16x32_bf16 a[240:243], v[96:99], v[20:23], a[240:243]
  v_mfma_f32_16x16x32_bf16 a[244:247], v[96:99], v[24:27], a[244:247]
  v_mfma_f32_16x16x32_bf16 a[248:251], v[96:99], v[28:31], a[248:251]
  v_mfma_f32_16x16x32_bf16 a[252:255], v[96:99], v[32:35], a[252:255]
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x32_bf16 a[0:3], v[100:103], v[36:39], a[0:3]
  v_mfma_f32_16x16x32_bf16 a[4:7], v[100:103], v[40:43], a[4:7]
  v_mfma_f32_16x16x32_bf16 a[8:11], v[100:103], v[44:47], a[8:11]
  v_mfma_f32_16x16x32_bf16 a[12:15], v[100:103], v[48:51], a[12:15]
  v_mfma_f32_16x16x32_bf16 a[16:19], v[100:103], v[52:55], a[16:19]
  v_mfma_f32_16x16x32_bf16 a[20:23], v[100:103], v[56:59], a[20:23]
  v_mfma_f32_16x16x32_bf16 a[24:27], v[100:103], v[60:63], a[24:27]
  v_mfma_f32_16x16x32_bf16 a[28:31], v[100:103], v[64:67], a[28:31]
  v_mfma_f32_16x16x32_bf16 a[32:35], v[104:107], v[36:39], a[32:35]
  v_mfma_f32_16x16x32_bf16 a[36:39], v[104:107], v[40:43], a[36:39]
  v_mfma_f32_16x16x32_bf16 a[40:43], v[104:107], v[44:47], a[40:43]
  v_mfma_f32_16x16x32_bf16 a[44:47], v[104:107], v[48:51], a[44:47]
  v_mfma_f32_16x16x32_bf16 a[48:51], v[104:107], v[52:55], a[48:51]
  v_mfma_f32_16x16x32_bf16 a[52:55], v[104:107], v[56:59], a[52:55]
  v_mfma_f32_16x16x32_bf16 a[56:59], v[104:107], v[60:63], a[56:59]
  v_mfma_f32_16x16x32_bf16 a[60:63], v[104:107], v[64:67], a[60:63]
  v_mfma_f32_16x16x32_bf16 a[64:67], v[108:111], v[36:39], a[64:67]
  v_mfma_f32_16x16x32_bf16 a[68:71], v[108:111], v[40:43], a[68:71]
  v_mfma_f32_16x16x32_bf16 a[72:75], v[108:111], v[44:47], a[72:75]
  v_mfma_f32_16x16x32_bf16 a[76:79], v[108:111], v[48:51], a[76:79]
  v_mfma_f32_16x16x32_bf16 a[80:83], v[108:111], v[52:55], a[80:83]
  v_mfma_f32_16x16x32_bf16 a[84:87], v[108:111], v[56:59], a[84:87]
  v_mfma_f32_16x16x32_bf16 a[88:91], v[108:111], v[60:63], a[88:91]
  v_mfma_f32_16x16x32_bf16 a[92:95], v[108:111], v[64:67], a[92:95]
  v_mfma_f32_16x16x32_bf16 a[96:99], v[112:115], v[36:39], a[96:99]
  v_mfma_f32_16x16x32_bf16 a[100:103], v[112:115], v[40:43], a[100:103]
  v_mfma_f32_16x16x32_bf16 a[104:107], v[112:115], v[44:47], a[104:107]
  v_mfma_f32_16x16x32_bf16 a[108:111], v[112:115], v[48:51], a[108:111]
  v_mfma_f32_16x16x32_bf16 a[112:115], v[112:115], v[52:55], a[112:115]
  v_mfma_f32_16x16x32_bf16 a[116:119], v[112:115], v[56:59], a[116:119]
  v_mfma_f32_16x16x32_bf16 a[120:123], v[112:115], v[60:63], a[120:123]
  v_mfma_f32_16x16x32_bf16 a[124:127], v[112:115], v[64:67], a[124:127]
  v_mfma_f32_16x16x32_bf16 a[128:131], v[116:119], v[36:39], a[128:131]
  v_mfma_f32_16x16x32_bf16 a[132:135], v[116:119], v[40:43], a[132:135]
  v_mfma_f32_16x16x32_bf16 a[136:139], v[116:119], v[44:47], a[136:139]
  v_mfma_f32_16x16x32_bf16 a[140:143], v[116:119], v[48:51], a[140:143]
  v_mfma_f32_16x16x32_bf16 a[144:147], v[116:119], v[52:55], a[144:147]
  v_mfma_f32_16x16x32_bf16 a[148:151], v[116:119], v[56:59], a[148:151]
  v_mfma_f32_16x16x32_bf16 a[152:155], v[116:119], v[60:63], a[152:155]
  v_mfma_f32_16x16x32_bf16 a[156:159], v[116:119], v[64:67], a[156:159]
  v_mfma_f32_16x16x32_bf16 a[160:163], v[120:123], v[36:39], a[160:163]
  v_mfma_f32_16x16x32_bf16 a[164:167], v[120:123], v[40:43], a[164:167]
  v_mfma_f32_16x16x32_bf16 a[168:171], v[120:123], v[44:47], a[168:171]
  v_mfma_f32_16x16x32_bf16 a[172:175], v[120:123], v[48:51], a[172:175]
  v_mfma_f32_16x16x32_bf16 a[176:179], v[120:123], v[52:55], a[176:179]
  v_mfma_f32_16x16x32_bf16 a[180:183], v[120:123], v[56:59], a[180:183]
  v_mfma_f32_16x16x32_bf16 a[184:187], v[120:123], v[60:63], a[184:187]
  v_mfma_f32_16x16x32_bf16 a[188:191], v[120:123], v[64:67], a[188:191]
  v_mfma_f32_16x16x32_bf16 a[192:195], v[124:127], v[36:39], a[192:195]
  v_mfma_f32_16x16x32_bf16 a[196:199], v[124:127], v[40:43], a[196:199]
  v_mfma_f32_16x16x32_bf16 a[200:203], v[124:127], v[44:47], a[200:203]
  v_mfma_f32_16x16x32_bf16 a[204:207], v[124:127], v[48:51], a[204:207]
  v_mfma_f32_16x16x32_bf16 a[208:211], v[124:127], v[52:55], a[208:211]
  v_mfma_f32_16x16x32_bf16 a[212:215], v[124:127], v[56:59], a[212:215]
  v_mfma_f32_16x16x32_bf16 a[216:219], v[124:127], v[60:63], a[216:219]
  v_mfma_f32_16x16x32_bf16 a[220:223], v[124:127], v[64:67], a[220:223]
  v_mfma_f32_16x16x32_bf16 a[224:227], v[128:131], v[36:39], a[224:227]
  v_mfma_f32_16x16x32_bf16 a[228:231], v[128:131], v[40:43], a[228:231]
  v_mfma_f32_16x16x32_bf16 a[232:235], v[128:131], v[44:47], a[232:235]
  v_mfma_f32_16x16x32_bf16 a[236:239], v[128:131], v[48:51], a[236:239]
  v_mfma_f32_16x16x32_bf16 a[240:243], v[128:131], v[52:55], a[240:243]
  v_mfma_f32_16x16x32_bf16 a[244:247], v[128:131], v[56:59], a[244:247]
  v_mfma_f32_16x16x32_bf16 a[248:251], v[128:131], v[60:63], a[248:251]
  v_mfma_f32_16x16x32_bf16 a[252:255], v[128:131], v[64:67], a[252:255]

  v_lshrrev_b32_e32 v4, 6, v134
  v_lshrrev_b32_e32 v5, 1, v4
  v_mul_lo_u32 v5, 16, v5
  v_and_b32_e32 v1, 63, v134
  v_lshrrev_b32_e32 v1, 4, v1
  v_lshlrev_b32_e32 v1, 2, v1
  v_add_lshl_u32 v1, v5, v1, 3
  v_mul_lo_u32 v2, v1, s38
  v_mul_lo_u32 v3, v1, s36
  v_and_b32_e32 v0, 1, v4
  v_mul_lo_u32 v0, 16, v0
  v_and_b32_e32 v5, 15, v134
  v_add_lshl_u32 v0, v5, v0, 3
  s_mul_i32 s8, 0x100, s2
  v_add_u32_e32 v0, s8, v0
  s_mul_i32 s8, 0x100, s3
  v_add_u32_e32 v1, s8, v1

  v_add_lshl_u32 v11, v3, v0, 1
  v_accvgpr_read_b32 v16, a0
  v_accvgpr_read_b32 v17, a4
  v_accvgpr_read_b32 v18, a8
  v_accvgpr_read_b32 v19, a12
  v_accvgpr_read_b32 v20, a16
  v_accvgpr_read_b32 v21, a20
  v_accvgpr_read_b32 v22, a24
  v_accvgpr_read_b32 v23, a28
  v_accvgpr_read_b32 v24, a32
  v_accvgpr_read_b32 v25, a36
  v_accvgpr_read_b32 v26, a40
  v_accvgpr_read_b32 v27, a44
  v_accvgpr_read_b32 v28, a48
  v_accvgpr_read_b32 v29, a52
  v_accvgpr_read_b32 v30, a56
  v_accvgpr_read_b32 v31, a60
  v_accvgpr_read_b32 v32, a64
  v_accvgpr_read_b32 v33, a68
  v_accvgpr_read_b32 v34, a72
  v_accvgpr_read_b32 v35, a76
  v_accvgpr_read_b32 v36, a80
  v_accvgpr_read_b32 v37, a84
  v_accvgpr_read_b32 v38, a88
  v_accvgpr_read_b32 v39, a92
  v_accvgpr_read_b32 v40, a96
  v_accvgpr_read_b32 v41, a100
  v_accvgpr_read_b32 v42, a104
  v_accvgpr_read_b32 v43, a108
  v_accvgpr_read_b32 v44, a112
  v_accvgpr_read_b32 v45, a116
  v_accvgpr_read_b32 v46, a120
  v_accvgpr_read_b32 v47, a124
  v_accvgpr_read_b32 v48, a128
  v_accvgpr_read_b32 v49, a132
  v_accvgpr_read_b32 v50, a136
  v_accvgpr_read_b32 v51, a140
  v_accvgpr_read_b32 v52, a144
  v_accvgpr_read_b32 v53, a148
  v_accvgpr_read_b32 v54, a152
  v_accvgpr_read_b32 v55, a156
  v_accvgpr_read_b32 v56, a160
  v_accvgpr_read_b32 v57, a164
  v_accvgpr_read_b32 v58, a168
  v_accvgpr_read_b32 v59, a172
  v_accvgpr_read_b32 v60, a176
  v_accvgpr_read_b32 v61, a180
  v_accvgpr_read_b32 v62, a184
  v_accvgpr_read_b32 v63, a188
  v_accvgpr_read_b32 v64, a192
  v_accvgpr_read_b32 v65, a196
  v_accvgpr_read_b32 v66, a200
  v_accvgpr_read_b32 v67, a204
  v_accvgpr_read_b32 v68, a208
  v_accvgpr_read_b32 v69, a212
  v_accvgpr_read_b32 v70, a216
  v_accvgpr_read_b32 v71, a220
  v_accvgpr_read_b32 v72, a224
  v_accvgpr_read_b32 v73, a228
  v_accvgpr_read_b32 v74, a232
  v_accvgpr_read_b32 v75, a236
  v_accvgpr_read_b32 v76, a240
  v_accvgpr_read_b32 v77, a244
  v_accvgpr_read_b32 v78, a248
  v_accvgpr_read_b32 v79, a252
  v_accvgpr_read_b32 v80, a1
  v_accvgpr_read_b32 v81, a5
  v_accvgpr_read_b32 v82, a9
  v_accvgpr_read_b32 v83, a13
  v_accvgpr_read_b32 v84, a17
  v_accvgpr_read_b32 v85, a21
  v_accvgpr_read_b32 v86, a25
  v_accvgpr_read_b32 v87, a29
  v_accvgpr_read_b32 v88, a33
  v_accvgpr_read_b32 v89, a37
  v_accvgpr_read_b32 v90, a41
  v_accvgpr_read_b32 v91, a45
  v_accvgpr_read_b32 v92, a49
  v_accvgpr_read_b32 v93, a53
  v_accvgpr_read_b32 v94, a57
  v_accvgpr_read_b32 v95, a61
  v_accvgpr_read_b32 v96, a65
  v_accvgpr_read_b32 v97, a69
  v_accvgpr_read_b32 v98, a73
  v_accvgpr_read_b32 v99, a77
  v_accvgpr_read_b32 v100, a81
  v_accvgpr_read_b32 v101, a85
  v_accvgpr_read_b32 v102, a89
  v_accvgpr_read_b32 v103, a93
  v_accvgpr_read_b32 v104, a97
  v_accvgpr_read_b32 v105, a101
  v_accvgpr_read_b32 v106, a105
  v_accvgpr_read_b32 v107, a109
  v_accvgpr_read_b32 v108, a113
  v_accvgpr_read_b32 v109, a117
  v_accvgpr_read_b32 v110, a121
  v_accvgpr_read_b32 v111, a125
  v_accvgpr_read_b32 v112, a129
  v_accvgpr_read_b32 v113, a133
  v_accvgpr_read_b32 v114, a137
  v_accvgpr_read_b32 v115, a141
  v_accvgpr_read_b32 v116, a145
  v_accvgpr_read_b32 v117, a149
  v_accvgpr_read_b32 v118, a153
  v_accvgpr_read_b32 v119, a157
  v_accvgpr_read_b32 v120, a161
  v_accvgpr_read_b32 v121, a165
  v_accvgpr_read_b32 v122, a169
  v_accvgpr_read_b32 v123, a173
  v_accvgpr_read_b32 v124, a177
  v_accvgpr_read_b32 v125, a181
  v_accvgpr_read_b32 v126, a185
  v_accvgpr_read_b32 v127, a189
  v_accvgpr_read_b32 v136, a193
  v_accvgpr_read_b32 v137, a197
  v_accvgpr_read_b32 v138, a201
  v_accvgpr_read_b32 v139, a205
  v_accvgpr_read_b32 v140, a209
  v_accvgpr_read_b32 v141, a213
  v_accvgpr_read_b32 v142, a217
  v_accvgpr_read_b32 v143, a221
  v_accvgpr_read_b32 v144, a225
  v_accvgpr_read_b32 v145, a229
  v_accvgpr_read_b32 v146, a233
  v_accvgpr_read_b32 v147, a237
  v_accvgpr_read_b32 v148, a241
  v_accvgpr_read_b32 v149, a245
  v_accvgpr_read_b32 v150, a249
  v_accvgpr_read_b32 v151, a253
  v_accvgpr_read_b32 v152, a2
  v_accvgpr_read_b32 v153, a6
  v_accvgpr_read_b32 v154, a10
  v_accvgpr_read_b32 v155, a14
  v_accvgpr_read_b32 v156, a18
  v_accvgpr_read_b32 v157, a22
  v_accvgpr_read_b32 v158, a26
  v_accvgpr_read_b32 v159, a30
  v_accvgpr_read_b32 v160, a34
  v_accvgpr_read_b32 v161, a38
  v_accvgpr_read_b32 v162, a42
  v_accvgpr_read_b32 v163, a46
  v_accvgpr_read_b32 v164, a50
  v_accvgpr_read_b32 v165, a54
  v_accvgpr_read_b32 v166, a58
  v_accvgpr_read_b32 v167, a62
  v_accvgpr_read_b32 v168, a66
  v_accvgpr_read_b32 v169, a70
  v_accvgpr_read_b32 v170, a74
  v_accvgpr_read_b32 v171, a78
  v_accvgpr_read_b32 v172, a82
  v_accvgpr_read_b32 v173, a86
  v_accvgpr_read_b32 v174, a90
  v_accvgpr_read_b32 v175, a94
  v_accvgpr_read_b32 v176, a98
  v_accvgpr_read_b32 v177, a102
  v_accvgpr_read_b32 v178, a106
  v_accvgpr_read_b32 v179, a110
  v_accvgpr_read_b32 v180, a114
  v_accvgpr_read_b32 v181, a118
  v_accvgpr_read_b32 v182, a122
  v_accvgpr_read_b32 v183, a126
  v_accvgpr_read_b32 v184, a130
  v_accvgpr_read_b32 v185, a134
  v_accvgpr_read_b32 v186, a138
  v_accvgpr_read_b32 v187, a142
  v_accvgpr_read_b32 v188, a146
  v_accvgpr_read_b32 v189, a150
  v_accvgpr_read_b32 v190, a154
  v_accvgpr_read_b32 v191, a158
  v_accvgpr_read_b32 v192, a162
  v_accvgpr_read_b32 v193, a166
  v_accvgpr_read_b32 v194, a170
  v_accvgpr_read_b32 v195, a174
  v_accvgpr_read_b32 v196, a178
  v_accvgpr_read_b32 v197, a182
  v_accvgpr_read_b32 v198, a186
  v_accvgpr_read_b32 v199, a190
  v_accvgpr_read_b32 v200, a194
  v_accvgpr_read_b32 v201, a198
  v_accvgpr_read_b32 v202, a202
  v_accvgpr_read_b32 v203, a206
  v_accvgpr_read_b32 v204, a210
  v_accvgpr_read_b32 v205, a214
  v_accvgpr_read_b32 v206, a218
  v_accvgpr_read_b32 v207, a222
  v_accvgpr_read_b32 v208, a226
  v_accvgpr_read_b32 v209, a230
  v_accvgpr_read_b32 v210, a234
  v_accvgpr_read_b32 v211, a238
  v_accvgpr_read_b32 v212, a242
  v_accvgpr_read_b32 v213, a246
  v_accvgpr_read_b32 v214, a250
  v_accvgpr_read_b32 v215, a254
  v_accvgpr_read_b32 v216, a3
  v_accvgpr_read_b32 v217, a7
  v_accvgpr_read_b32 v218, a11
  v_accvgpr_read_b32 v219, a15
  v_accvgpr_read_b32 v220, a19
  v_accvgpr_read_b32 v221, a23
  v_accvgpr_read_b32 v222, a27
  v_accvgpr_read_b32 v223, a31
  v_accvgpr_read_b32 v224, a35
  v_accvgpr_read_b32 v225, a39
  v_accvgpr_read_b32 v226, a43
  v_accvgpr_read_b32 v227, a47
  v_accvgpr_read_b32 v228, a51
  v_accvgpr_read_b32 v229, a55
  v_accvgpr_read_b32 v230, a59
  v_accvgpr_read_b32 v231, a63
  v_accvgpr_read_b32 v232, a67
  v_accvgpr_read_b32 v233, a71
  v_accvgpr_read_b32 v234, a75
  v_accvgpr_read_b32 v235, a79
  v_accvgpr_read_b32 v236, a83
  v_accvgpr_read_b32 v237, a87
  v_accvgpr_read_b32 v238, a91
  v_accvgpr_read_b32 v239, a95
  v_accvgpr_read_b32 v240, a99
  v_accvgpr_read_b32 v241, a103
  v_accvgpr_read_b32 v242, a107
  v_accvgpr_read_b32 v243, a111
  v_accvgpr_read_b32 v244, a115
  v_accvgpr_read_b32 v245, a119
  v_accvgpr_read_b32 v246, a123
  v_accvgpr_read_b32 v247, a127
  v_mov_b32_e32 v8, 0xffff0000
  v_mov_b32_e32 v9, 0x7fff0000
  v_mov_b32_e32 v10, 0x7fff
  v_cvt_pk_bf16_f32 v16, v16, v17
  v_cvt_pk_bf16_f32 v17, v18, v19
  v_cvt_pk_bf16_f32 v18, v20, v21
  v_cvt_pk_bf16_f32 v19, v22, v23
  buffer_store_dwordx4 v[16:19], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v24, v24, v25
  v_cvt_pk_bf16_f32 v25, v26, v27
  v_cvt_pk_bf16_f32 v26, v28, v29
  v_cvt_pk_bf16_f32 v27, v30, v31
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[24:27], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v32, v32, v33
  v_cvt_pk_bf16_f32 v33, v34, v35
  v_cvt_pk_bf16_f32 v34, v36, v37
  v_cvt_pk_bf16_f32 v35, v38, v39
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[32:35], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v40, v40, v41
  v_cvt_pk_bf16_f32 v41, v42, v43
  v_cvt_pk_bf16_f32 v42, v44, v45
  v_cvt_pk_bf16_f32 v43, v46, v47
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[40:43], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v48, v48, v49
  v_cvt_pk_bf16_f32 v49, v50, v51
  v_cvt_pk_bf16_f32 v50, v52, v53
  v_cvt_pk_bf16_f32 v51, v54, v55
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[48:51], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v56, v56, v57
  v_cvt_pk_bf16_f32 v57, v58, v59
  v_cvt_pk_bf16_f32 v58, v60, v61
  v_cvt_pk_bf16_f32 v59, v62, v63
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[56:59], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v64, v64, v65
  v_cvt_pk_bf16_f32 v65, v66, v67
  v_cvt_pk_bf16_f32 v66, v68, v69
  v_cvt_pk_bf16_f32 v67, v70, v71
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[64:67], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v72, v72, v73
  v_cvt_pk_bf16_f32 v73, v74, v75
  v_cvt_pk_bf16_f32 v74, v76, v77
  v_cvt_pk_bf16_f32 v75, v78, v79
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[72:75], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v80, v80, v81
  v_cvt_pk_bf16_f32 v81, v82, v83
  v_cvt_pk_bf16_f32 v82, v84, v85
  v_cvt_pk_bf16_f32 v83, v86, v87
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[80:83], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v88, v88, v89
  v_cvt_pk_bf16_f32 v89, v90, v91
  v_cvt_pk_bf16_f32 v90, v92, v93
  v_cvt_pk_bf16_f32 v91, v94, v95
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[88:91], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v96, v96, v97
  v_cvt_pk_bf16_f32 v97, v98, v99
  v_cvt_pk_bf16_f32 v98, v100, v101
  v_cvt_pk_bf16_f32 v99, v102, v103
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[96:99], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v104, v104, v105
  v_cvt_pk_bf16_f32 v105, v106, v107
  v_cvt_pk_bf16_f32 v106, v108, v109
  v_cvt_pk_bf16_f32 v107, v110, v111
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[104:107], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v112, v112, v113
  v_cvt_pk_bf16_f32 v113, v114, v115
  v_cvt_pk_bf16_f32 v114, v116, v117
  v_cvt_pk_bf16_f32 v115, v118, v119
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[112:115], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v120, v120, v121
  v_cvt_pk_bf16_f32 v121, v122, v123
  v_cvt_pk_bf16_f32 v122, v124, v125
  v_cvt_pk_bf16_f32 v123, v126, v127
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[120:123], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v136, v136, v137
  v_cvt_pk_bf16_f32 v137, v138, v139
  v_cvt_pk_bf16_f32 v138, v140, v141
  v_cvt_pk_bf16_f32 v139, v142, v143
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[136:139], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v144, v144, v145
  v_cvt_pk_bf16_f32 v145, v146, v147
  v_cvt_pk_bf16_f32 v146, v148, v149
  v_cvt_pk_bf16_f32 v147, v150, v151
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[144:147], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v152, v152, v153
  v_cvt_pk_bf16_f32 v153, v154, v155
  v_cvt_pk_bf16_f32 v154, v156, v157
  v_cvt_pk_bf16_f32 v155, v158, v159
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[152:155], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v160, v160, v161
  v_cvt_pk_bf16_f32 v161, v162, v163
  v_cvt_pk_bf16_f32 v162, v164, v165
  v_cvt_pk_bf16_f32 v163, v166, v167
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[160:163], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v168, v168, v169
  v_cvt_pk_bf16_f32 v169, v170, v171
  v_cvt_pk_bf16_f32 v170, v172, v173
  v_cvt_pk_bf16_f32 v171, v174, v175
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[168:171], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v176, v176, v177
  v_cvt_pk_bf16_f32 v177, v178, v179
  v_cvt_pk_bf16_f32 v178, v180, v181
  v_cvt_pk_bf16_f32 v179, v182, v183
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[176:179], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v184, v184, v185
  v_cvt_pk_bf16_f32 v185, v186, v187
  v_cvt_pk_bf16_f32 v186, v188, v189
  v_cvt_pk_bf16_f32 v187, v190, v191
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[184:187], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v192, v192, v193
  v_cvt_pk_bf16_f32 v193, v194, v195
  v_cvt_pk_bf16_f32 v194, v196, v197
  v_cvt_pk_bf16_f32 v195, v198, v199
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[192:195], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v200, v200, v201
  v_cvt_pk_bf16_f32 v201, v202, v203
  v_cvt_pk_bf16_f32 v202, v204, v205
  v_cvt_pk_bf16_f32 v203, v206, v207
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[200:203], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v208, v208, v209
  v_cvt_pk_bf16_f32 v209, v210, v211
  v_cvt_pk_bf16_f32 v210, v212, v213
  v_cvt_pk_bf16_f32 v211, v214, v215
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[208:211], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v216, v216, v217
  v_cvt_pk_bf16_f32 v217, v218, v219
  v_cvt_pk_bf16_f32 v218, v220, v221
  v_cvt_pk_bf16_f32 v219, v222, v223
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[216:219], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v224, v224, v225
  v_cvt_pk_bf16_f32 v225, v226, v227
  v_cvt_pk_bf16_f32 v226, v228, v229
  v_cvt_pk_bf16_f32 v227, v230, v231
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[224:227], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v232, v232, v233
  v_cvt_pk_bf16_f32 v233, v234, v235
  v_cvt_pk_bf16_f32 v234, v236, v237
  v_cvt_pk_bf16_f32 v235, v238, v239
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[232:235], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v240, v240, v241
  v_cvt_pk_bf16_f32 v241, v242, v243
  v_cvt_pk_bf16_f32 v242, v244, v245
  v_cvt_pk_bf16_f32 v243, v246, v247
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[240:243], v11, s[16:19], 0 offen nt
  s_nop 0
  v_accvgpr_read_b32 v16, a131
  v_accvgpr_read_b32 v17, a135
  v_accvgpr_read_b32 v18, a139
  v_accvgpr_read_b32 v19, a143
  v_accvgpr_read_b32 v20, a147
  v_accvgpr_read_b32 v21, a151
  v_accvgpr_read_b32 v22, a155
  v_accvgpr_read_b32 v23, a159
  v_accvgpr_read_b32 v24, a163
  v_accvgpr_read_b32 v25, a167
  v_accvgpr_read_b32 v26, a171
  v_accvgpr_read_b32 v27, a175
  v_accvgpr_read_b32 v28, a179
  v_accvgpr_read_b32 v29, a183
  v_accvgpr_read_b32 v30, a187
  v_accvgpr_read_b32 v31, a191
  v_accvgpr_read_b32 v32, a195
  v_accvgpr_read_b32 v33, a199
  v_accvgpr_read_b32 v34, a203
  v_accvgpr_read_b32 v35, a207
  v_accvgpr_read_b32 v36, a211
  v_accvgpr_read_b32 v37, a215
  v_accvgpr_read_b32 v38, a219
  v_accvgpr_read_b32 v39, a223
  v_accvgpr_read_b32 v40, a227
  v_accvgpr_read_b32 v41, a231
  v_accvgpr_read_b32 v42, a235
  v_accvgpr_read_b32 v43, a239
  v_accvgpr_read_b32 v44, a243
  v_accvgpr_read_b32 v45, a247
  v_accvgpr_read_b32 v46, a251
  v_accvgpr_read_b32 v47, a255
  v_mov_b32_e32 v8, 0xffff0000
  v_mov_b32_e32 v9, 0x7fff0000
  v_mov_b32_e32 v10, 0x7fff
  v_cvt_pk_bf16_f32 v16, v16, v17
  v_cvt_pk_bf16_f32 v17, v18, v19
  v_cvt_pk_bf16_f32 v18, v20, v21
  v_cvt_pk_bf16_f32 v19, v22, v23
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[16:19], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v24, v24, v25
  v_cvt_pk_bf16_f32 v25, v26, v27
  v_cvt_pk_bf16_f32 v26, v28, v29
  v_cvt_pk_bf16_f32 v27, v30, v31
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[24:27], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v32, v32, v33
  v_cvt_pk_bf16_f32 v33, v34, v35
  v_cvt_pk_bf16_f32 v34, v36, v37
  v_cvt_pk_bf16_f32 v35, v38, v39
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[32:35], v11, s[16:19], 0 offen nt
  v_cvt_pk_bf16_f32 v40, v40, v41
  v_cvt_pk_bf16_f32 v41, v42, v43
  v_cvt_pk_bf16_f32 v42, v44, v45
  v_cvt_pk_bf16_f32 v43, v46, v47
  s_lshl_b32 s12, s36, 1
  s_add_u32 s16, s16, s12
  s_addc_u32 s17, s17, 0
  buffer_store_dwordx4 v[40:43], v11, s[16:19], 0 offen nt
  s_nop 0
  s_endpgm
