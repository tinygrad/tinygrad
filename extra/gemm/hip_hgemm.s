

/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureDefault
.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.text
.protected Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8
.globl Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8
.p2align 8
.type Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 66 // sgprs
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 32 x 4 */
/* SubGroup= 4 x 32 */
/* VectorWidth=1 */
/* GlobalLoadVectorWidthA=8, GlobalLoadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.kernels:
  - .name: Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8
    .symbol: 'Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            sizeC
        .size:            8
        .offset:          0
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeA
        .size:            8
        .offset:          8
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeB
        .size:            8
        .offset:          16
        .value_kind:      by_value
        .value_type:      u64
      - .name:            D
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            OffsetD
        .size:            8
        .offset:          56
        .value_kind:      by_value
        .value_type:      u64
      - .name:            OffsetC
        .size:            8
        .offset:          64
        .value_kind:      by_value
        .value_type:      u64
      - .name:            OffsetA
        .size:            8
        .offset:          72
        .value_kind:      by_value
        .value_type:      u64
      - .name:            OffsetB
        .size:            8
        .offset:          80
        .value_kind:      by_value
        .value_type:      u64
      - .name:            alpha
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      f16
      - .name:            beta
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      f16
      - .name:            strideD0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      i32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          148
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          152
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          156
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          160
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          164
        .value_kind:      by_value
        .value_type:      u32
      - .name:            padding
        .size:            4
        .offset:          168
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   25088
    .kernarg_segment_align:      8
    .kernarg_segment_size:       176
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 66
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_HB_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS0_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_MMFGLC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS1_SU0_SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW1_SNLL0_TSGRA0_TSGRB0_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW1_VFLRP0_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_nc_u32 \dst, \src0 \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_nc_i32 \dst, \src0 \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_add_co_ci_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_nc_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_nc_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
.endm

.macro _v_dot2acc_f32_f16 dst, src0, src1
v_dot2acc_f32_f16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmp_lt_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmp_lt_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmp_lt_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmp_lt_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmp_lt_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmp_lt_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmp_eq_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmp_eq_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmp_eq_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmp_eq_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmp_eq_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmp_eq_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmp_le_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmp_le_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmp_le_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmp_le_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmp_le_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmp_le_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmp_gt_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmp_gt_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmp_gt_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmp_gt_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmp_gt_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmp_gt_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmp_ne_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmp_ne_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmp_ne_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmp_ne_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmp_ne_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmp_ne_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmp_lg_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmp_lg_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmp_lg_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmp_lg_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmp_lg_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmp_lg_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmp_ge_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmp_ge_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmp_ge_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmp_ge_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmp_ge_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmp_ge_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmp_o_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmp_o_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmp_o_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmp_o_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmp_o_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmp_o_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmp_u_i16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmp_u_i32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmp_u_i64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmp_u_u16 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmp_u_u32 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmp_u_u64 \dst, \src0, \src1
   s_mov_b32 exec_lo \dst
.endm
.macro _v_mac_f32 c:req, a:req, b:req
    v_fmac_f32 \c, \a, \b
.endmacro

/* scale global load macros */
.macro _s_load_b32 dst base offset
    s_load_b32 \dst \base \offset
.endm

.macro _s_load_b64 dst base offset
    s_load_b64 \dst \base \offset
.endm

.macro _s_load_b128 dst base offset
    s_load_b128 \dst \base \offset
.endm

.macro _s_load_b256 dst base offset
    s_load_b256 \dst \base \offset
.endm

.macro _s_load_b512 dst base offset
    s_load_b512 \dst \base \offset
.endm


/* ds operation macros */
.macro _ds_load_u8 dst src offset
    ds_load_u8 \dst \src \offset
.endm

.macro _ds_load_u8_d16_hi dst src offset
    ds_load_u8_d16_hi \dst \src \offset
.endm

.macro _ds_load_u16 dst src offset
    ds_load_u16 \dst \src \offset
.endm

.macro _ds_load_u16_d16_hi dst src offset
    ds_load_u16_d16_hi \dst \src \offset
.endm

.macro _ds_load_b32 dst src offset
    ds_load_b32 \dst \src \offset
.endm

.macro _ds_load_b64 dst src offset
    ds_load_b64 \dst \src \offset
.endm

.macro _ds_load_b128 dst src offset
    ds_load_b128 \dst \src \offset
.endm

.macro _ds_store_b8 dst src offset
    ds_store_b8 \dst \src \offset
.endm

.macro _ds_store_b8_d16_hi dst src offset
    ds_store_b8_d16_hi \dst \src \offset
.endm

.macro _ds_store_b16 dst src offset
    ds_store_b16 \dst \src \offset
.endm

.macro _ds_store_b16_d16_hi dst src offset
    ds_store_b16_d16_hi \dst \src \offset
.endm

.macro _ds_store_b32 dst src offset
    ds_store_b32 \dst \src \offset
.endm

.macro _ds_store_b64 dst src offset
    ds_store_b64 \dst \src \offset
.endm

.macro _ds_store_b128 dst src offset
    ds_store_b128 \dst \src \offset
.endm

.macro _ds_load2_b32 dst src offset1 offset2
    ds_load2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_load2_b64 dst src offset1 offset2
    ds_load2_b64 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b32 dst src offset1 offset2
    ds_store2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b64 dst src offset1 offset2
    ds_store2_b64 \dst \src \offset1 \offset2
.endm


/* buffer memory operation macros */
.macro _buffer_load_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_b32 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_b64 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b96 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_b96 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b128 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_b128 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_d16_b16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_d16_hi_b16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_d16_u8 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_d16_hi_u8 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_u16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_u16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b32_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_b32 \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_u16_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_u16 \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b32 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b32 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b64 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b64 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b96 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b96 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b128 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b128 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b16 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_d16_hi_b16 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_b8 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_d16_hi_b8 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap_b32 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap_b64 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm


/* buffer memory operation macros */
.macro _global_load_b32 dst base src ioffset md0 md1 md2
    global_load_b32 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b64 dst base src ioffset md0 md1 md2
    global_load_b64 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b96 dst base src ioffset md0 md1 md2
    global_load_b96 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b128 dst base src ioffset md0 md1 md2
    global_load_b128 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_b16 dst base src ioffset md0 md1 md2
    global_load_d16_b16 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_hi_b16 dst base src ioffset md0 md1 md2
    global_load_d16_hi_b16 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_u8 dst base src ioffset md0 md1 md2
    global_load_d16_u8 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_hi_u8 dst base src ioffset md0 md1 md2
    global_load_d16_hi_u8 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_u16 dst base src ioffset md0 md1 md2
    global_load_u16 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_store_b32 base src src2 md0 md1 md2
    global_store_b32 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b64 base src src2 md0 md1 md2
    global_store_b64 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b96 base src src2 md0 md1 md2
    global_store_b96 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b128 base src src2 md0 md1 md2
    global_store_b128 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_b16 base src src2 md0 md1 md2
    global_store_d16_b16 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_hi_b16 base src src2 md0 md1 md2
    global_store_d16_hi_b16 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_u8 base src src2 md0 md1 md2
    global_store_d16_u8 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_hi_u8 base src src2 md0 md1 md2
    global_store_d16_hi_u8 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_u16 base src src2 md0 md1 md2
    global_store_u16 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_atomic_cmpswap_b32 tmp base data src ioffset md
    global_atomic_cmpswap_b32 \tmp \base \data \src \ioffset \md
.endm

.macro _global_atomic_cmpswap_b64 tmp base data src ioffset md
    global_atomic_cmpswap_b64 \tmp \base \data \src \ioffset \md
.endm


/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-128), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 130
.set vgprG2LA, 202
.set vgprValuB_X0_I0, 163
.set vgprG2LB, 210
.set vgprLocalWriteAddrA, 195
.set vgprLocalWriteAddrB, 196
.set vgprGlobalReadOffsetA, 197
.set vgprGlobalReadOffsetB, 199
.set vgprLocalReadAddrA, 218
.set vgprLocalReadAddrB, 219
.set vgprSerial, 220
/* Num VGPR=256 */
/* Num AccVGPR=0 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 6
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeA, 24
.set sgprTensor2dSizeB, 26
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprOffsetD, 36
.set sgprOffsetC, 38
.set sgprOffsetA, 40
.set sgprOffsetB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprStridesD, 46
.set sgprStridesC, 48
.set sgprStridesA, 50
.set sgprStridesB, 52
.set sgprSizesFree, 54
.set sgprSizesSum, 57
.set sgprOrigStaggerUIter, 58
.set sgprNumWorkGroups0, 59
.set sgprNumWorkGroups1, 60
.set sgprNumFullBlocks, 61
.set sgprWgmRemainder1, 62
.set sgprMagicNumberWgmRemainder1, 63
.set sgprShadowLimitA, 0
.set sgprShadowLimitB, 28
.set sgprGlobalReadIncsA, 7
.set sgprGlobalReadIncsB, 30
/* max SGPR=66 */

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 128
.set DepthU, 16
.set GSU, 1
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimitA, 0xffffffff
.set BufferLimitB, 0xffffffff
.set BufferOOB, 0xfffff000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x31004000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* format (7b): 4                         */
/* _unusedA (2b): 0                       */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* resource_level (1b): 1                 */
/* _unusedB (1b): 0                       */
/* LLC_noalloc (2b): 0                    */
/* oob_select (2b): 3                     */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x31004000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          // 
v_rcp_f32 v[\vQuotient], v[\vQuotient]             // 
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] // 
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         // 
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] // 
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp1], vcc_lo, 0x0, v[\vRemainder] // 
v_cmp_ne_i32 s[\sTmp], 0x0, v[\vTmp0]              // 
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp] // 
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp0], vcc_lo, v[\vQuotient], v[\vRemainder] // 
_v_add_co_u32 v[\vQuotient], vcc_lo, v[\vQuotient], v[\vRemainder] // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp] // 
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] // 
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vTmp0], vcc_lo, v[\vDividend], v[\vRemainder] // 
v_cmp_ge_u32 s[\sTmp], v[\vDividend], v[\vRemainder] // 
_v_add_co_u32 v[\vRemainder], vcc_lo, 0x1, v[\vQuotient] // 
_v_add_co_u32 v[\vTmp1], vcc_lo, -1, v[\vQuotient] // 
v_cmp_le_u32 vcc_lo, v[\vDivisor], v[\vTmp0]       // 
s_and_b32 vcc_lo, s[\sTmp], vcc_lo                 // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc_lo // 
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp] // 
v_cmp_ne_i32 vcc_lo, 0x0, v[\vDivisor]             // 
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc_lo // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vRemainder], vcc_lo, v[\vDividend], v[\vRemainder] // final result
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x6200                               // LDS clamp at 25088 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* Load Kernel Args */
_s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // 
_s_load_b512 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48 // 
_s_load_b256 s[56:63], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // 

/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

/*lr0I*/
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v1, 1, v2                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v1, 0x4, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16)
_v_add_u32 v0, v1, v0                              // 8. final local read offset: flrOffset = lrOffset + WOffset
/*lr1J*/
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x4, v1                          // 1. N offset: nOffset = nIdx * nStride(16)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v2, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v2, 0x8, v2                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(256)
_v_add_u32 v1, v2, v1                              // 8. final local read offset: flrOffset = lrOffset + WOffset


/* local read addresses: final offsets a */

v_lshlrev_b32 v[vgprLocalReadAddrA], 0x1, v0       // Final Offset: offset = (lro0)*bpe


/* local read addresses: final offsets b */

v_lshlrev_b32 v[vgprLocalReadAddrB], 0x1, v1       // Final Offset: offset = (lro1)*bpe
v_lshrrev_b32 v0, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 8 per block 128
v_lshlrev_b32 v0, 0x4, v0                          // Final Offset: padding 8 per block 128
_v_add_u32 v[vgprLocalReadAddrB], v0, v[vgprLocalReadAddrB] // Final Offset: add padding 8 per block 128


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)


/* global read addresses: tile offset assignment a */

/* LVCA = 16 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_and_b32 v2, 31, v[vgprSerial]                    // v2 = v[vgprSerial] % 32
v_lshrrev_b32 v1, 4, v2                            // v1 = v2 / 16
v_and_b32 v0, 15, v2                               // v0 = v2 % 16
v_readfirstlane_b32 s64, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s64, s64, 0x5                           // WaveId
s_mul_i32 s64, s64, 4                              // Global Read Wave: each wave loads continuous lsp(2)*nrp(2) columns
_v_add_u32 v1, s64, v1                             // Global Read Wave: add back to column index
/* gro-tile *= glvw */
v_lshlrev_b32 v0, 0x3, v0                          // v0 = v0 * 8


/* global read addresses: tile offset assignment b */

/* LVCB = 2 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
v_and_b32 v4, 31, v[vgprSerial]                    // v4 = v[vgprSerial] % 32
v_lshrrev_b32 v2, 1, v4                            // v2 = v4 / 2
v_and_b32 v3, 1, v4                                // v3 = v4 % 2
v_readfirstlane_b32 s64, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s64, s64, 0x5                           // WaveId
s_mul_i32 s64, s64, 32                             // Global Read Wave: each wave loads continuous lsp(16)*nrp(2) columns
_v_add_u32 v2, s64, v2                             // Global Read Wave: add back to column index
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v1 */

/* lwaUnrollAssignmentB = v3 */


/* local write addresses: first offset a */

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v1     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v3, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpe
v_lshrrev_b32 v4, 7, v[vgprLocalWriteAddrB]        // padding 8 per block 128
v_lshlrev_b32 v4, 0x4, v4                          // padding 8 per block 128
_v_add_u32 v[vgprLocalWriteAddrB], v4, v[vgprLocalWriteAddrB] // add padding 8 per block 128
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x1000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048*2







s_waitcnt lgkmcnt(0)                               // wait for 168 bytes of kern args
s_lshl_b64 s[sgprOffsetD:sgprOffsetD+1], s[sgprOffsetD:sgprOffsetD+1], 0x1 // elements offset to bytes offset
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s[sgprOffsetD] // add offset to buffer address
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s[sgprOffsetD+1] // add offset to buffer address
s_lshl_b64 s[sgprOffsetC:sgprOffsetC+1], s[sgprOffsetC:sgprOffsetC+1], 0x1 // elements offset to bytes offset
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s[sgprOffsetC] // add offset to buffer address
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s[sgprOffsetC+1] // add offset to buffer address
s_lshl_b64 s[sgprOffsetA:sgprOffsetA+1], s[sgprOffsetA:sgprOffsetA+1], 0x1 // elements offset to bytes offset
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s[sgprOffsetA] // add offset to buffer address
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s[sgprOffsetA+1] // add offset to buffer address
s_lshl_b64 s[sgprOffsetB:sgprOffsetB+1], s[sgprOffsetB:sgprOffsetB+1], 0x1 // elements offset to bytes offset
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s[sgprOffsetB] // add offset to buffer address
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s[sgprOffsetB+1] // add offset to buffer address
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift

.set OffsetD, UNDEF
.set OffsetC, UNDEF
.set OffsetA, UNDEF
.set OffsetB, UNDEF
.set AddressD, UNDEF
.set AddressC, UNDEF
.set AddressA, UNDEF
.set AddressB, UNDEF

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // Alpha == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if alpha != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:



/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */
s_mov_b32 s35, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s33, s[sgprWorkGroup1], s35           // s_magic mul
s_mul_i32 s32, s[sgprWorkGroup1], s35              // s_magic mul
s_lshr_b64 s[32:33], s[32:33], 31                  // sMagicDiv
s_mul_i32 s33, s32, 8                              // quotient * non-magic divisor
s_sub_u32 s33, s[sgprWorkGroup1], s33              // WorkGroup1=remainder
s_mul_i32 s33, s33, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s33, s33, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
s_cmp_ge_u32 s32, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s35, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s34, s[sgprWgmRemainder1], 8         // 
s_mul_hi_u32 s3, s33, s35                          // s_magic mul
s_mul_i32 s2, s33, s35                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s34 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s33, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s32, s32, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s32 // wg1 += blockId * WGM


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */

v_mov_b32 v4, v0                                   // groA0I_0


/* global read addresses: tile offsets b */

v_mov_b32 v5, v2                                   // groB1J_0
_v_add_co_u32 v6, vcc_lo, 16, v5                   // groB1J_1 += LSPB


/* global read addresses: unroll offsets a */

v_mov_b32 v7, v1                                   // groAL_0
_v_add_co_u32 v8, vcc_lo, 2, v7                    // groAL_1 + LSPA


/* global read addresses: unroll offsets b */

v_mov_b32 v9, v3                                   // groBL_0


/* global read addresses: shift a */

s_mul_i32 s31, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s31, s[sgprSizeI], s31                   // edge = Size0I - WG*MT
s_sub_u32 s31, s31, 8                              // edge -= margin(8)
v_mov_b32 v10, s31                                 // edge vgpr = Size0I- WG*MT - margin(8)
v_min_i32 v4, v10, v4                              // offset = (offset < edge) ? offset(v4) : edge(v10)


/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  7, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  4,  8, 10 // gROA_0_0_1_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  9,  5, 10 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+1,  9,  6, 10 // gROB_0_0_1_0


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s35, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s34, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s34 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s35 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32
s_mul_hi_u32 s33, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s32, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s34, s34, s32                            // accum wg term to tilestart
s_addc_u32 s35, s35, s33                           // accum wg term to tilestart
s_lshl_b64 s[34:35], s[34:35], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s34        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s35       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s35, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s34, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s35, s34, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s34, s34, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s34 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s35 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_mul_hi_u32 s33, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s32, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s34, s34, s32                            // accum wg term to tilestart
s_addc_u32 s35, s35, s33                           // accum wg term to tilestart
s_lshl_b64 s[34:35], s[34:35], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s34        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s35       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)

/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_10                  // skip to ShadowInitStart iter b/c numIter==0


_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0


_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0


/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_10: // 

s_mov_b32 s[sgprSrdD+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s34, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s33, s34, s[sgprStrideC1J]            // CScale s34 by Stride
s_mul_i32 s32, s34, s[sgprStrideC1J]               // CScale s34 by Stride
s_lshl_b64 s[32:33], s[32:33], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s32        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s33       // add hi to SRD
s_mul_hi_u32 s33, s34, s[sgprStrideD1J]            // Scale s34 by Stride
s_mul_i32 s32, s34, s[sgprStrideD1J]               // Scale s34 by Stride
s_lshl_b64 s[32:33], s[32:33], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s33       // add hi to SRD

s_mul_hi_u32 s33, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s32, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[32:33], s[32:33], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s32        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s33       // add hi to SRD
s_mul_hi_u32 s33, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s32, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[32:33], s[32:33], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s33       // add hi to SRD



/* initC: remove C-tile 0-128 from pool */

/* initC: remove AB-tile 130-195 from pool */
v_mov_b32 v[vgprValuC+0], 0x0                      // initC
v_mov_b32 v[vgprValuC+1], 0x0                      // initC
v_mov_b32 v[vgprValuC+2], 0x0                      // initC
v_mov_b32 v[vgprValuC+3], 0x0                      // initC
v_mov_b32 v[vgprValuC+4], 0x0                      // initC
v_mov_b32 v[vgprValuC+5], 0x0                      // initC
v_mov_b32 v[vgprValuC+6], 0x0                      // initC
v_mov_b32 v[vgprValuC+7], 0x0                      // initC
v_mov_b32 v[vgprValuC+8], 0x0                      // initC
v_mov_b32 v[vgprValuC+9], 0x0                      // initC
v_mov_b32 v[vgprValuC+10], 0x0                     // initC
v_mov_b32 v[vgprValuC+11], 0x0                     // initC
v_mov_b32 v[vgprValuC+12], 0x0                     // initC
v_mov_b32 v[vgprValuC+13], 0x0                     // initC
v_mov_b32 v[vgprValuC+14], 0x0                     // initC
v_mov_b32 v[vgprValuC+15], 0x0                     // initC
v_mov_b32 v[vgprValuC+16], 0x0                     // initC
v_mov_b32 v[vgprValuC+17], 0x0                     // initC
v_mov_b32 v[vgprValuC+18], 0x0                     // initC
v_mov_b32 v[vgprValuC+19], 0x0                     // initC
v_mov_b32 v[vgprValuC+20], 0x0                     // initC
v_mov_b32 v[vgprValuC+21], 0x0                     // initC
v_mov_b32 v[vgprValuC+22], 0x0                     // initC
v_mov_b32 v[vgprValuC+23], 0x0                     // initC
v_mov_b32 v[vgprValuC+24], 0x0                     // initC
v_mov_b32 v[vgprValuC+25], 0x0                     // initC
v_mov_b32 v[vgprValuC+26], 0x0                     // initC
v_mov_b32 v[vgprValuC+27], 0x0                     // initC
v_mov_b32 v[vgprValuC+28], 0x0                     // initC
v_mov_b32 v[vgprValuC+29], 0x0                     // initC
v_mov_b32 v[vgprValuC+30], 0x0                     // initC
v_mov_b32 v[vgprValuC+31], 0x0                     // initC
v_mov_b32 v[vgprValuC+32], 0x0                     // initC
v_mov_b32 v[vgprValuC+33], 0x0                     // initC
v_mov_b32 v[vgprValuC+34], 0x0                     // initC
v_mov_b32 v[vgprValuC+35], 0x0                     // initC
v_mov_b32 v[vgprValuC+36], 0x0                     // initC
v_mov_b32 v[vgprValuC+37], 0x0                     // initC
v_mov_b32 v[vgprValuC+38], 0x0                     // initC
v_mov_b32 v[vgprValuC+39], 0x0                     // initC
v_mov_b32 v[vgprValuC+40], 0x0                     // initC
v_mov_b32 v[vgprValuC+41], 0x0                     // initC
v_mov_b32 v[vgprValuC+42], 0x0                     // initC
v_mov_b32 v[vgprValuC+43], 0x0                     // initC
v_mov_b32 v[vgprValuC+44], 0x0                     // initC
v_mov_b32 v[vgprValuC+45], 0x0                     // initC
v_mov_b32 v[vgprValuC+46], 0x0                     // initC
v_mov_b32 v[vgprValuC+47], 0x0                     // initC
v_mov_b32 v[vgprValuC+48], 0x0                     // initC
v_mov_b32 v[vgprValuC+49], 0x0                     // initC
v_mov_b32 v[vgprValuC+50], 0x0                     // initC
v_mov_b32 v[vgprValuC+51], 0x0                     // initC
v_mov_b32 v[vgprValuC+52], 0x0                     // initC
v_mov_b32 v[vgprValuC+53], 0x0                     // initC
v_mov_b32 v[vgprValuC+54], 0x0                     // initC
v_mov_b32 v[vgprValuC+55], 0x0                     // initC
v_mov_b32 v[vgprValuC+56], 0x0                     // initC
v_mov_b32 v[vgprValuC+57], 0x0                     // initC
v_mov_b32 v[vgprValuC+58], 0x0                     // initC
v_mov_b32 v[vgprValuC+59], 0x0                     // initC
v_mov_b32 v[vgprValuC+60], 0x0                     // initC
v_mov_b32 v[vgprValuC+61], 0x0                     // initC
v_mov_b32 v[vgprValuC+62], 0x0                     // initC
v_mov_b32 v[vgprValuC+63], 0x0                     // initC
v_mov_b32 v[vgprValuC+64], 0x0                     // initC
v_mov_b32 v[vgprValuC+65], 0x0                     // initC
v_mov_b32 v[vgprValuC+66], 0x0                     // initC
v_mov_b32 v[vgprValuC+67], 0x0                     // initC
v_mov_b32 v[vgprValuC+68], 0x0                     // initC
v_mov_b32 v[vgprValuC+69], 0x0                     // initC
v_mov_b32 v[vgprValuC+70], 0x0                     // initC
v_mov_b32 v[vgprValuC+71], 0x0                     // initC
v_mov_b32 v[vgprValuC+72], 0x0                     // initC
v_mov_b32 v[vgprValuC+73], 0x0                     // initC
v_mov_b32 v[vgprValuC+74], 0x0                     // initC
v_mov_b32 v[vgprValuC+75], 0x0                     // initC
v_mov_b32 v[vgprValuC+76], 0x0                     // initC
v_mov_b32 v[vgprValuC+77], 0x0                     // initC
v_mov_b32 v[vgprValuC+78], 0x0                     // initC
v_mov_b32 v[vgprValuC+79], 0x0                     // initC
v_mov_b32 v[vgprValuC+80], 0x0                     // initC
v_mov_b32 v[vgprValuC+81], 0x0                     // initC
v_mov_b32 v[vgprValuC+82], 0x0                     // initC
v_mov_b32 v[vgprValuC+83], 0x0                     // initC
v_mov_b32 v[vgprValuC+84], 0x0                     // initC
v_mov_b32 v[vgprValuC+85], 0x0                     // initC
v_mov_b32 v[vgprValuC+86], 0x0                     // initC
v_mov_b32 v[vgprValuC+87], 0x0                     // initC
v_mov_b32 v[vgprValuC+88], 0x0                     // initC
v_mov_b32 v[vgprValuC+89], 0x0                     // initC
v_mov_b32 v[vgprValuC+90], 0x0                     // initC
v_mov_b32 v[vgprValuC+91], 0x0                     // initC
v_mov_b32 v[vgprValuC+92], 0x0                     // initC
v_mov_b32 v[vgprValuC+93], 0x0                     // initC
v_mov_b32 v[vgprValuC+94], 0x0                     // initC
v_mov_b32 v[vgprValuC+95], 0x0                     // initC
v_mov_b32 v[vgprValuC+96], 0x0                     // initC
v_mov_b32 v[vgprValuC+97], 0x0                     // initC
v_mov_b32 v[vgprValuC+98], 0x0                     // initC
v_mov_b32 v[vgprValuC+99], 0x0                     // initC
v_mov_b32 v[vgprValuC+100], 0x0                    // initC
v_mov_b32 v[vgprValuC+101], 0x0                    // initC
v_mov_b32 v[vgprValuC+102], 0x0                    // initC
v_mov_b32 v[vgprValuC+103], 0x0                    // initC
v_mov_b32 v[vgprValuC+104], 0x0                    // initC
v_mov_b32 v[vgprValuC+105], 0x0                    // initC
v_mov_b32 v[vgprValuC+106], 0x0                    // initC
v_mov_b32 v[vgprValuC+107], 0x0                    // initC
v_mov_b32 v[vgprValuC+108], 0x0                    // initC
v_mov_b32 v[vgprValuC+109], 0x0                    // initC
v_mov_b32 v[vgprValuC+110], 0x0                    // initC
v_mov_b32 v[vgprValuC+111], 0x0                    // initC
v_mov_b32 v[vgprValuC+112], 0x0                    // initC
v_mov_b32 v[vgprValuC+113], 0x0                    // initC
v_mov_b32 v[vgprValuC+114], 0x0                    // initC
v_mov_b32 v[vgprValuC+115], 0x0                    // initC
v_mov_b32 v[vgprValuC+116], 0x0                    // initC
v_mov_b32 v[vgprValuC+117], 0x0                    // initC
v_mov_b32 v[vgprValuC+118], 0x0                    // initC
v_mov_b32 v[vgprValuC+119], 0x0                    // initC
v_mov_b32 v[vgprValuC+120], 0x0                    // initC
v_mov_b32 v[vgprValuC+121], 0x0                    // initC
v_mov_b32 v[vgprValuC+122], 0x0                    // initC
v_mov_b32 v[vgprValuC+123], 0x0                    // initC
v_mov_b32 v[vgprValuC+124], 0x0                    // initC
v_mov_b32 v[vgprValuC+125], 0x0                    // initC
v_mov_b32 v[vgprValuC+126], 0x0                    // initC
v_mov_b32 v[vgprValuC+127], 0x0                    // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_11                   // Only branch on scc1
s_getpc_B64 s[32:33]                               // addr of next instr
s_add_i32 s34, PrefetchGlobalLastIterEnd_5, 0x4    // target branch offset
s_add_u32 s32, s32, s34                            // add target branch offset
s_addc_u32 s33, s33, 0                             // add high and carry
s_setpc_b64 s[32:33]                               // branch to PrefetchGlobalLastIterEnd_5
label_NoBranch_11:

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=08wait for global read


/* local write a */
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:512 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 512

/* local write b */
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:576 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576


/* local write swap a */


/* (EPS=1) local write swap internal offset -> 16384 */


/* local write swap b */


/* (EPS=1) local write swap internal offset -> 16384 */





/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_12:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0013: // LoopCopy1 

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-11wait for local write

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //4sync for global read


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */





/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2368 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2880 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3136 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3392 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3648 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3904 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
_ds_load_u16 v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2176 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2688 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2944 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3200 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3712 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3968 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2240 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2496 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:2752 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:3008 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3264 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3520 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:3776 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:4032 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=15 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:1168 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:3472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // lgkmcnt=-1 vmcnt=3wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:16384 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 16384
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:16896 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 16896
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:16384 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 16384
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:16960 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 16960

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 16384 */

/* local read swap offsets b */

/* local read swap internal offset -> 16384 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(4)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
s_nop 1
v_wmma_f16_16x16x16_f16 v[0+0:7+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[0:7]
v_wmma_f16_16x16x16_f16 v[8+0:15+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[8:15]
v_wmma_f16_16x16x16_f16 v[16+0:23+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[16:23]
v_wmma_f16_16x16x16_f16 v[24+0:31+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[24:31]
v_wmma_f16_16x16x16_f16 v[32+0:39+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[32:39]
v_wmma_f16_16x16x16_f16 v[40+0:47+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[40:47]
v_wmma_f16_16x16x16_f16 v[48+0:55+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[48:55]
v_wmma_f16_16x16x16_f16 v[56+0:63+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[56:63]
v_wmma_f16_16x16x16_f16 v[64+0:71+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[64:71]
v_wmma_f16_16x16x16_f16 v[72+0:79+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[72:79]
v_wmma_f16_16x16x16_f16 v[80+0:87+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[80:87]
v_wmma_f16_16x16x16_f16 v[88+0:95+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[88:95]
v_wmma_f16_16x16x16_f16 v[96+0:103+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[96:103]
v_wmma_f16_16x16x16_f16 v[104+0:111+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[104:111]
v_wmma_f16_16x16x16_f16 v[112+0:119+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[112:119]
v_wmma_f16_16x16x16_f16 v[120+0:127+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=64 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */




/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0014: // LoopCopy2 

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-11wait for local write

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //4sync for global read


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */





/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:16384 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:16640 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:16896 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:17152 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:17408 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:17664 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:17920 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:18176 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:18432 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:18688 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:18944 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:19200 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:19456 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:19712 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:19968 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:20224 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:16448 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:16704 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:16960 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:17216 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:17472 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:17728 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:17984 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:18240 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:18496 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:18752 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:19008 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:19264 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:19520 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:19776 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:20032 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:20288 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:16512 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:16768 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:17024 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:17280 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
_ds_load_u16 v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:17536 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:17792 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:18048 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:18304 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:18560 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:18816 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:19072 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:19328 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:19584 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:19840 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:20096 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:20352 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:16576 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:16832 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:17088 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:17344 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:17600 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:17856 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:18112 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:18368 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:18624 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:18880 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:19136 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:19392 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:19648 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:19904 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:20160 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:20416 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=15 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:16384 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16400 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:17536 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:17552 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:18688 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:18704 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:19840 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:19856 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // lgkmcnt=-1 vmcnt=3wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:512 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 512
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:576 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(4)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
s_nop 1
v_wmma_f16_16x16x16_f16 v[0+0:7+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[0:7]
v_wmma_f16_16x16x16_f16 v[8+0:15+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[8:15]
v_wmma_f16_16x16x16_f16 v[16+0:23+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[16:23]
v_wmma_f16_16x16x16_f16 v[24+0:31+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[24:31]
v_wmma_f16_16x16x16_f16 v[32+0:39+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[32:39]
v_wmma_f16_16x16x16_f16 v[40+0:47+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[40:47]
v_wmma_f16_16x16x16_f16 v[48+0:55+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[48:55]
v_wmma_f16_16x16x16_f16 v[56+0:63+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[56:63]
v_wmma_f16_16x16x16_f16 v[64+0:71+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[64:71]
v_wmma_f16_16x16x16_f16 v[72+0:79+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[72:79]
v_wmma_f16_16x16x16_f16 v[80+0:87+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[80:87]
v_wmma_f16_16x16x16_f16 v[88+0:95+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[88:95]
v_wmma_f16_16x16x16_f16 v[96+0:103+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[96:103]
v_wmma_f16_16x16x16_f16 v[104+0:111+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[104:111]
v_wmma_f16_16x16x16_f16 v[112+0:119+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[112:119]
v_wmma_f16_16x16x16_f16 v[120+0:127+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=64 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */




/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
LoopEndL_evenexit_4: // unroll loop eveniter exit
s_branch LoopEndL_2                                // exit unroll loopL (and skip second exit code)
LoopEndL_oddexit_3: // unroll loop odditer exit

/* Select high bank of LDS */
v_xor_b32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB] // swap Red Blk
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin                                      */
/******************************************/

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 OptNLL_End_15                       // Branch if Beta is not zero

s_mov_b32 s32, 0x3c003c00                          // Packed alpha==1.0
s_cmp_eq_u32 s[sgprAlpha], s32                     // alpha == 1.0?
s_cbranch_scc0 OptNLL_End_15                       // branch if alpha != 1

s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required
s_and_b32 s32, 127, s[sgprSizeJ]                   // s32 = s[sgprSizeJ] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required

s_and_b32 s33, 15, s[sgprSizesSum+0]               // s33 = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s33, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_15                       // skip if tail loop required

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local write

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //



/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2368 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2880 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3136 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3392 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3648 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3904 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2176 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2688 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2944 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3200 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3712 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3968 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2240 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2496 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:2752 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:3008 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3264 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3520 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:3776 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:4032 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=15 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:1168 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:3472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
s_nop 1
v_wmma_f16_16x16x16_f16 v[0+0:7+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[0:7]
v_wmma_f16_16x16x16_f16 v[8+0:15+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[8:15]
v_wmma_f16_16x16x16_f16 v[16+0:23+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[16:23]
v_wmma_f16_16x16x16_f16 v[24+0:31+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[24:31]
v_wmma_f16_16x16x16_f16 v[32+0:39+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[32:39]
v_wmma_f16_16x16x16_f16 v[40+0:47+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[40:47]
v_wmma_f16_16x16x16_f16 v[48+0:55+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[48:55]
v_wmma_f16_16x16x16_f16 v[56+0:63+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[56:63]
v_wmma_f16_16x16x16_f16 v[64+0:71+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[64:71]
v_wmma_f16_16x16x16_f16 v[72+0:79+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[72:79]
v_wmma_f16_16x16x16_f16 v[80+0:87+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[80:87]
v_wmma_f16_16x16x16_f16 v[88+0:95+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[88:95]
v_wmma_f16_16x16x16_f16 v[96+0:103+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[96:103]
v_wmma_f16_16x16x16_f16 v[104+0:111+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[104:111]
v_wmma_f16_16x16x16_f16 v[112+0:119+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[112:119]
v_wmma_f16_16x16x16_f16 v[120+0:127+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=64 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/* Stores for OptNLL */
Summation_End_OptNLL_16:
/* endSummation: add vgpr [130...218) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v134, 5, v[vgprSerial]               // v134 = v[vgprSerial] / 32
v_and_b32 v131, 31, v[vgprSerial]                  // v131 = v[vgprSerial] % 32
v_lshrrev_b32 v131, 4, v131                        // v131 = v131 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_lshrrev_b32 v135, 1, v134                        // v135 = v134 / 2
v_mul_lo_u32 v135, 0x10, v135                      // wave coordination offset 1
_v_add_u32 v131, v135, v131                        // coordination 1 = wave_id1 + tid1
v_mul_lo_u32 v132, v131, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v133, v131, s[sgprStrideD1J]          //  offset 1
v_and_b32 v135, 1, v134                            // v135 = v134 % 2
v_mul_lo_u32 v135, 0x10, v135                      // wave coordination offset 0
v_and_b32 v130, 15, v[vgprSerial]                  // v130 = v[vgprSerial] % 16
_v_add_lshl_u32 v130, v135, v130, 0                // coordination 0 = wave_id0 + tid0
s_mul_i32 s31, 128, s[sgprWorkGroup0]              // wgp0 * MT0
_v_add_u32 v130, s31, v130                         // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s31, 128, s[sgprWorkGroup1]              // wgp1 * MT1
_v_add_u32 v131, s31, v131                         // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
GW_B0_E0_19:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=114 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (16,3,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (17,3,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (18,3,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (19,3,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (20,3,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (21,3,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (22,3,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (23,3,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (24,3,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (25,3,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (26,3,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (27,3,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
/* (d1,vc1,d0,vc0)=(16,0,3,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
/* (d1,vc1,d0,vc0)=(17,0,3,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
/* (d1,vc1,d0,vc0)=(18,0,3,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
/* (d1,vc1,d0,vc0)=(19,0,3,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
/* (d1,vc1,d0,vc0)=(20,0,3,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
/* (d1,vc1,d0,vc0)=(21,0,3,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
/* (d1,vc1,d0,vc0)=(22,0,3,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
/* (d1,vc1,d0,vc0)=(23,0,3,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
/* (d1,vc1,d0,vc0)=(24,0,3,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
/* (d1,vc1,d0,vc0)=(25,0,3,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
/* (d1,vc1,d0,vc0)=(26,0,3,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
/* (d1,vc1,d0,vc0)=(27,0,3,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
_v_add_lshl_u32 v136, v133, v130, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=130, coord0Vgpr=130
v_mov_b32 v[vgprValuC+138], v[vgprValuC+0] // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+8] // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+16] // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+24] // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+1] // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+9] // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+17] // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+25] // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+2] // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+10] // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+18] // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+26] // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+3] // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+11] // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+19] // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+27] // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+4] // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+12] // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+20] // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+28] // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+5] // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+13] // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+21] // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+29] // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+6] // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+14] // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+22] // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+30] // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+7] // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+15] // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+23] // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+31] // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+32] // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+40] // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+48] // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+56] // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+33] // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+175], v[vgprValuC+41] // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+176], v[vgprValuC+49] // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+177], v[vgprValuC+57] // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+178], v[vgprValuC+34] // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+179], v[vgprValuC+42] // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+180], v[vgprValuC+50] // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+181], v[vgprValuC+58] // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+182], v[vgprValuC+35] // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+183], v[vgprValuC+43] // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+184], v[vgprValuC+51] // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+185], v[vgprValuC+59] // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+186], v[vgprValuC+36] // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+187], v[vgprValuC+44] // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+188], v[vgprValuC+52] // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+189], v[vgprValuC+60] // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+190], v[vgprValuC+37] // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+191], v[vgprValuC+45] // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+192], v[vgprValuC+53] // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+193], v[vgprValuC+61] // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+194], v[vgprValuC+38] // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+195], v[vgprValuC+46] // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+196], v[vgprValuC+54] // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+197], v[vgprValuC+62] // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+198], v[vgprValuC+39] // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+199], v[vgprValuC+47] // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+200], v[vgprValuC+55] // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+201], v[vgprValuC+63] // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+202], v[vgprValuC+64] // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+203], v[vgprValuC+72] // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+204], v[vgprValuC+80] // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+205], v[vgprValuC+88] // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+206], v[vgprValuC+65] // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+207], v[vgprValuC+73] // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+208], v[vgprValuC+81] // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+209], v[vgprValuC+89] // copy MI out reg to vreg[71]
v_mov_b32 v[vgprValuC+210], v[vgprValuC+66] // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+211], v[vgprValuC+74] // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+212], v[vgprValuC+82] // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+213], v[vgprValuC+90] // copy MI out reg to vreg[75]
v_mov_b32 v[vgprValuC+214], v[vgprValuC+67] // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+215], v[vgprValuC+75] // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+216], v[vgprValuC+83] // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+217], v[vgprValuC+91] // copy MI out reg to vreg[79]
v_mov_b32 v[vgprValuC+221], v[vgprValuC+68] // copy MI out reg to vreg[80]
v_mov_b32 v[vgprValuC+222], v[vgprValuC+76] // copy MI out reg to vreg[81]
v_mov_b32 v[vgprValuC+223], v[vgprValuC+84] // copy MI out reg to vreg[82]
v_mov_b32 v[vgprValuC+224], v[vgprValuC+92] // copy MI out reg to vreg[83]
v_mov_b32 v[vgprValuC+225], v[vgprValuC+69] // copy MI out reg to vreg[84]
v_mov_b32 v[vgprValuC+226], v[vgprValuC+77] // copy MI out reg to vreg[85]
v_mov_b32 v[vgprValuC+227], v[vgprValuC+85] // copy MI out reg to vreg[86]
v_mov_b32 v[vgprValuC+228], v[vgprValuC+93] // copy MI out reg to vreg[87]
v_mov_b32 v[vgprValuC+229], v[vgprValuC+70] // copy MI out reg to vreg[88]
v_mov_b32 v[vgprValuC+230], v[vgprValuC+78] // copy MI out reg to vreg[89]
v_mov_b32 v[vgprValuC+231], v[vgprValuC+86] // copy MI out reg to vreg[90]
v_mov_b32 v[vgprValuC+232], v[vgprValuC+94] // copy MI out reg to vreg[91]
v_mov_b32 v[vgprValuC+233], v[vgprValuC+71] // copy MI out reg to vreg[92]
v_mov_b32 v[vgprValuC+234], v[vgprValuC+79] // copy MI out reg to vreg[93]
v_mov_b32 v[vgprValuC+235], v[vgprValuC+87] // copy MI out reg to vreg[94]
v_mov_b32 v[vgprValuC+236], v[vgprValuC+95] // copy MI out reg to vreg[95]
v_mov_b32 v[vgprValuC+237], v[vgprValuC+96] // copy MI out reg to vreg[96]
v_mov_b32 v[vgprValuC+238], v[vgprValuC+104] // copy MI out reg to vreg[97]
v_mov_b32 v[vgprValuC+239], v[vgprValuC+112] // copy MI out reg to vreg[98]
v_mov_b32 v[vgprValuC+240], v[vgprValuC+120] // copy MI out reg to vreg[99]
v_mov_b32 v[vgprValuC+241], v[vgprValuC+97] // copy MI out reg to vreg[100]
v_mov_b32 v[vgprValuC+242], v[vgprValuC+105] // copy MI out reg to vreg[101]
v_mov_b32 v[vgprValuC+243], v[vgprValuC+113] // copy MI out reg to vreg[102]
v_mov_b32 v[vgprValuC+244], v[vgprValuC+121] // copy MI out reg to vreg[103]
v_mov_b32 v[vgprValuC+245], v[vgprValuC+98] // copy MI out reg to vreg[104]
v_mov_b32 v[vgprValuC+246], v[vgprValuC+106] // copy MI out reg to vreg[105]
v_mov_b32 v[vgprValuC+247], v[vgprValuC+114] // copy MI out reg to vreg[106]
v_mov_b32 v[vgprValuC+248], v[vgprValuC+122] // copy MI out reg to vreg[107]
v_mov_b32 v[vgprValuC+249], v[vgprValuC+99] // copy MI out reg to vreg[108]
v_mov_b32 v[vgprValuC+250], v[vgprValuC+107] // copy MI out reg to vreg[109]
v_mov_b32 v[vgprValuC+251], v[vgprValuC+115] // copy MI out reg to vreg[110]
v_mov_b32 v[vgprValuC+252], v[vgprValuC+123] // copy MI out reg to vreg[111]
v_mov_b32 v[vgprValuC+253], v[vgprValuC+100] // copy MI out reg to vreg[112]
v_mov_b32 v[vgprValuC+254], v[vgprValuC+108] // copy MI out reg to vreg[113]

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v140, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v142, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v144, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v146, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v148, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v150, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v152, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v153, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v154, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v155, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v156, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v157, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v158, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v159, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v160, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v161, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v162, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v163, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v164, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v165, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v166, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v167, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v168, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v169, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v170, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v171, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v172, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v173, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v174, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v175, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v176, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v177, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v178, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v179, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v180, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v181, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v182, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v183, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v184, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v185, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v186, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v187, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v188, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v189, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v190, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v191, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v192, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v193, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v194, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v195, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v196, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v197, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v198, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v199, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v200, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v201, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v202, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v203, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v204, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v205, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v206, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v207, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v208, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v209, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v210, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v211, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v212, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v213, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v214, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v215, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v216, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v217, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v221, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v222, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v223, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v224, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v225, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v226, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v227, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v228, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v229, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v230, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v231, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v232, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v233, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v234, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v235, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v236, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v237, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v238, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v239, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v240, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v241, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v242, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v243, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v244, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v245, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v246, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v247, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v248, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v249, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v250, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v251, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v252, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v253, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v254, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (28,2,0,0:vw1); (28,3,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (29,3,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (30,3,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1); (31,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
/* (d1,vc1,d0,vc0)=(28,0,3,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
/* (d1,vc1,d0,vc0)=(29,0,3,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
/* (d1,vc1,d0,vc0)=(30,0,3,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
/* (d1,vc1,d0,vc0)=(31,0,3,0) */
v_mov_b32 v[vgprValuC+138], v[vgprValuC+116] // copy MI out reg to vreg[114]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+124] // copy MI out reg to vreg[115]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+101] // copy MI out reg to vreg[116]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+109] // copy MI out reg to vreg[117]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+117] // copy MI out reg to vreg[118]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+125] // copy MI out reg to vreg[119]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+102] // copy MI out reg to vreg[120]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+110] // copy MI out reg to vreg[121]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+118] // copy MI out reg to vreg[122]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+126] // copy MI out reg to vreg[123]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+103] // copy MI out reg to vreg[124]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+111] // copy MI out reg to vreg[125]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+119] // copy MI out reg to vreg[126]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+127] // copy MI out reg to vreg[127]

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v140, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v142, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v144, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v146, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v148, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v150, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_21                           // jump to end
label_GW_End_21:

s_endpgm                                           // Kernel End
OptNLL_End_15:


/******************************************/
/* Ord. NoLoadLoop - Begin                                      */
/******************************************/


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local write

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //



/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2368 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2880 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3136 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3392 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3648 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3904 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2176 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2688 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2944 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3200 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3712 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3968 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2240 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2496 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:2752 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:3008 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3264 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3520 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:3776 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:4032 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=15 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:1168 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:3472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
s_nop 1
v_wmma_f16_16x16x16_f16 v[0+0:7+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[0:7]
v_wmma_f16_16x16x16_f16 v[8+0:15+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[8:15]
v_wmma_f16_16x16x16_f16 v[16+0:23+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[16:23]
v_wmma_f16_16x16x16_f16 v[24+0:31+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[24:31]
v_wmma_f16_16x16x16_f16 v[32+0:39+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[32:39]
v_wmma_f16_16x16x16_f16 v[40+0:47+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[40:47]
v_wmma_f16_16x16x16_f16 v[48+0:55+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[48:55]
v_wmma_f16_16x16x16_f16 v[56+0:63+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[56:63]
v_wmma_f16_16x16x16_f16 v[64+0:71+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[64:71]
v_wmma_f16_16x16x16_f16 v[72+0:79+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[72:79]
v_wmma_f16_16x16x16_f16 v[80+0:87+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[80:87]
v_wmma_f16_16x16x16_f16 v[88+0:95+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[88:95]
v_wmma_f16_16x16x16_f16 v[96+0:103+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[96:103]
v_wmma_f16_16x16x16_f16 v[104+0:111+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[104:111]
v_wmma_f16_16x16x16_f16 v[112+0:119+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[112:119]
v_wmma_f16_16x16x16_f16 v[120+0:127+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=64 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

PrefetchGlobalLastIterEnd_5:


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */


v_and_b32 v[vgprLocalWriteAddrA], 0xf03fff, v[vgprLocalWriteAddrA] // reset to Red


/* local write reset offsets b */


v_and_b32 v[vgprLocalWriteAddrB], 0xf03fff, v[vgprLocalWriteAddrB] // reset to Red


//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_cbranch_scc1 SkipTailLoopL_8                     // skip to end of tail loop b/c numIter==0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
_buffer_load_d16_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
_buffer_load_d16_hi_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
_buffer_load_d16_b16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
_buffer_load_d16_hi_b16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
_buffer_load_d16_b16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
_buffer_load_d16_hi_b16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
_buffer_load_d16_b16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
_buffer_load_d16_hi_b16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
_buffer_load_d16_b16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
_buffer_load_d16_hi_b16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
_buffer_load_d16_b16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
_buffer_load_d16_hi_b16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
_buffer_load_d16_b16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
_buffer_load_d16_hi_b16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
_buffer_load_d16_b16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
_buffer_load_d16_hi_b16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:14 // load one buffer value


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
_buffer_load_d16_b16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
_buffer_load_d16_hi_b16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
_buffer_load_d16_b16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
_buffer_load_d16_hi_b16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
_buffer_load_d16_b16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
_buffer_load_d16_hi_b16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
_buffer_load_d16_b16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
_buffer_load_d16_hi_b16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
_buffer_load_d16_b16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
_buffer_load_d16_hi_b16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
_buffer_load_d16_b16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
_buffer_load_d16_hi_b16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
_buffer_load_d16_b16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
_buffer_load_d16_hi_b16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
_buffer_load_d16_b16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
_buffer_load_d16_hi_b16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:14 // load one buffer value

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //


/* Done global A/B reads */




/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:512 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 512


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:576 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_waitcnt_lgkmcnt null, 0                          // extra navi wait
s_barrier //


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x3fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x3fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_6:


/* local read a */

_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+8], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+9], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+10], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+12], v[vgprLocalReadAddrA] offset:2368 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+13], v[vgprLocalReadAddrA] offset:2880 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3136 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+14], v[vgprLocalReadAddrA] offset:3392 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3648 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:3904 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+16], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+17], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+18], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2176 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+20], v[vgprLocalReadAddrA] offset:2432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2688 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+21], v[vgprLocalReadAddrA] offset:2944 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3200 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+22], v[vgprLocalReadAddrA] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3712 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:3968 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=15 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+24], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+25], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=4 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+26], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=5 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=6 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=7 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2240 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=8 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+28], v[vgprLocalReadAddrA] offset:2496 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=9 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:2752 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=10 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+29], v[vgprLocalReadAddrA] offset:3008 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=11 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3264 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=12 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+30], v[vgprLocalReadAddrA] offset:3520 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=13 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:3776 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=14 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v[vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:4032 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=15 oIdx=0 buffer=0 iui=0


/* local read b */

_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:1168 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:3456 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:3472 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s31, 0x1000                              // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc_lo, s31, v[vgprLocalReadAddrA] // lrA += 4096 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s31, 0x20                                // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc_lo, s31, v[vgprLocalReadAddrB] // lrB += 32 (LSU*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read


s_sub_i32 s32, s[sgprLoopCounterL], 1              // calculate 64bit groups index
s_lshr_b32 s33, s32, 2                             // calculate 64bit groups index
s_and_b32 s32, s32, 3                              // calculate shift value
s_sub_i32 s32, 3, s32                              // calculate shift value
s_lshl_b32 s32, s32, 4                             // calculate shift value
v_cmp_eq_i32 s34, s33, 0                           // handle this 64bit group: part 1
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0], v[vgprValuA_X0_I0+0+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+1], v[vgprValuA_X0_I0+0+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0], v[vgprValuA_X0_I0+8+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+1], v[vgprValuA_X0_I0+8+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+16+0:vgprValuA_X0_I0+16+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0], v[vgprValuA_X0_I0+16+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+1], v[vgprValuA_X0_I0+16+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+24+0:vgprValuA_X0_I0+24+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0], v[vgprValuA_X0_I0+24+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+1], v[vgprValuA_X0_I0+24+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+0+0:vgprValuB_X0_I0+0+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0], v[vgprValuB_X0_I0+0+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+1], v[vgprValuB_X0_I0+0+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+8+0:vgprValuB_X0_I0+8+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0], v[vgprValuB_X0_I0+8+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+1], v[vgprValuB_X0_I0+8+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+16+0:vgprValuB_X0_I0+16+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0], v[vgprValuB_X0_I0+16+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+1], v[vgprValuB_X0_I0+16+0+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+24+0:vgprValuB_X0_I0+24+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0], v[vgprValuB_X0_I0+24+0+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+1], v[vgprValuB_X0_I0+24+0+1], v223, s34 // shift if in this 64b group
v_cmp_eq_i32 s34, s33, 1                           // handle this 64bit group: part 1
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+0], v[vgprValuA_X0_I0+8+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+1], v[vgprValuA_X0_I0+8+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+16+2:vgprValuA_X0_I0+16+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+0], v[vgprValuA_X0_I0+16+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+1], v[vgprValuA_X0_I0+16+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+24+2:vgprValuA_X0_I0+24+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+0], v[vgprValuA_X0_I0+24+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+1], v[vgprValuA_X0_I0+24+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+0+2:vgprValuB_X0_I0+0+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+8+2:vgprValuB_X0_I0+8+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+0], v[vgprValuB_X0_I0+8+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+1], v[vgprValuB_X0_I0+8+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+16+2:vgprValuB_X0_I0+16+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+0], v[vgprValuB_X0_I0+16+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+1], v[vgprValuB_X0_I0+16+2+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+24+2:vgprValuB_X0_I0+24+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+0], v[vgprValuB_X0_I0+24+2+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+1], v[vgprValuB_X0_I0+24+2+1], v223, s34 // shift if in this 64b group
v_cmp_lt_i32 s34, s33, 1                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+0], v[vgprValuA_X0_I0+8+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+1], v[vgprValuA_X0_I0+8+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+0], v[vgprValuA_X0_I0+16+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+1], v[vgprValuA_X0_I0+16+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+0], v[vgprValuA_X0_I0+24+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+1], v[vgprValuA_X0_I0+24+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+0], v[vgprValuB_X0_I0+8+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+1], v[vgprValuB_X0_I0+8+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+0], v[vgprValuB_X0_I0+16+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+1], v[vgprValuB_X0_I0+16+2+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+0], v[vgprValuB_X0_I0+24+2+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+1], v[vgprValuB_X0_I0+24+2+1], 0, s34 // shift if in this 64b group
v_cmp_eq_i32 s34, s33, 2                           // handle this 64bit group: part 1
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+0], v[vgprValuA_X0_I0+8+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+1], v[vgprValuA_X0_I0+8+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+0], v[vgprValuA_X0_I0+16+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+1], v[vgprValuA_X0_I0+16+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+0], v[vgprValuA_X0_I0+24+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+1], v[vgprValuA_X0_I0+24+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+0], v[vgprValuB_X0_I0+8+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+1], v[vgprValuB_X0_I0+8+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+0], v[vgprValuB_X0_I0+16+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+1], v[vgprValuB_X0_I0+16+4+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+0], v[vgprValuB_X0_I0+24+4+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+1], v[vgprValuB_X0_I0+24+4+1], v223, s34 // shift if in this 64b group
v_cmp_lt_i32 s34, s33, 2                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+0], v[vgprValuA_X0_I0+8+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+1], v[vgprValuA_X0_I0+8+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+0], v[vgprValuA_X0_I0+16+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+1], v[vgprValuA_X0_I0+16+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+0], v[vgprValuA_X0_I0+24+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+1], v[vgprValuA_X0_I0+24+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+0], v[vgprValuB_X0_I0+8+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+1], v[vgprValuB_X0_I0+8+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+0], v[vgprValuB_X0_I0+16+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+1], v[vgprValuB_X0_I0+16+4+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+0], v[vgprValuB_X0_I0+24+4+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+1], v[vgprValuB_X0_I0+24+4+1], 0, s34 // shift if in this 64b group
v_cmp_eq_i32 s34, s33, 3                           // handle this 64bit group: part 1
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+0+6:vgprValuA_X0_I0+0+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+8+6:vgprValuA_X0_I0+8+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+0], v[vgprValuA_X0_I0+8+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+1], v[vgprValuA_X0_I0+8+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+16+6:vgprValuA_X0_I0+16+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+0], v[vgprValuA_X0_I0+16+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+1], v[vgprValuA_X0_I0+16+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuA_X0_I0+24+6:vgprValuA_X0_I0+24+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+0], v[vgprValuA_X0_I0+24+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+1], v[vgprValuA_X0_I0+24+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+0+6:vgprValuB_X0_I0+0+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+8+6:vgprValuB_X0_I0+8+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+0], v[vgprValuB_X0_I0+8+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+1], v[vgprValuB_X0_I0+8+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+16+6:vgprValuB_X0_I0+16+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+0], v[vgprValuB_X0_I0+16+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+1], v[vgprValuB_X0_I0+16+6+1], v223, s34 // shift if in this 64b group
v_lshlrev_b64 v[222:223], s32, v[vgprValuB_X0_I0+24+6:vgprValuB_X0_I0+24+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+0], v[vgprValuB_X0_I0+24+6+0], v222, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+1], v[vgprValuB_X0_I0+24+6+1], v223, s34 // shift if in this 64b group
v_cmp_lt_i32 s34, s33, 3                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+0], v[vgprValuA_X0_I0+8+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+1], v[vgprValuA_X0_I0+8+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+0], v[vgprValuA_X0_I0+16+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+1], v[vgprValuA_X0_I0+16+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+0], v[vgprValuA_X0_I0+24+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+1], v[vgprValuA_X0_I0+24+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+0], v[vgprValuB_X0_I0+8+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+1], v[vgprValuB_X0_I0+8+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+0], v[vgprValuB_X0_I0+16+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+1], v[vgprValuB_X0_I0+16+6+1], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+0], v[vgprValuB_X0_I0+24+6+0], 0, s34 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+1], v[vgprValuB_X0_I0+24+6+1], 0, s34 // shift if in this 64b group
s_nop 1
v_wmma_f16_16x16x16_f16 v[0+0:7+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[0:7]
v_wmma_f16_16x16x16_f16 v[8+0:15+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[8:15]
v_wmma_f16_16x16x16_f16 v[16+0:23+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[16:23]
v_wmma_f16_16x16x16_f16 v[24+0:31+0], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[24:31]
v_wmma_f16_16x16x16_f16 v[32+0:39+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[32:39]
v_wmma_f16_16x16x16_f16 v[40+0:47+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[40:47]
v_wmma_f16_16x16x16_f16 v[48+0:55+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[48:55]
v_wmma_f16_16x16x16_f16 v[56+0:63+0], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[56:63]
v_wmma_f16_16x16x16_f16 v[64+0:71+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[64:71]
v_wmma_f16_16x16x16_f16 v[72+0:79+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[72:79]
v_wmma_f16_16x16x16_f16 v[80+0:87+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[80:87]
v_wmma_f16_16x16x16_f16 v[88+0:95+0], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[88:95]
v_wmma_f16_16x16x16_f16 v[96+0:103+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[96:103]
v_wmma_f16_16x16x16_f16 v[104+0:111+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[104:111]
v_wmma_f16_16x16x16_f16 v[112+0:119+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[112:119]
v_wmma_f16_16x16x16_f16 v[120+0:127+0], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[120:127]


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x10 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x10 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_6                    // restart LoopL
TailLoopEndL_7:

SkipTailLoopL_8:

Summation_End_28:
/* endSummation: add vgpr [130...218) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */


/* shift vector components d0 */

v_mov_b32 v131, s[sgprWorkGroup0]                  // 
v_mul_i32_i24 v131, -0x80, v131                    // wg*MT
_v_add_co_u32 v131, vcc_lo, s[sgprSizesFree+0], v131 // wgMT = Size - wg*MT
v_mov_b32 v132, 0x80                               // MT
v_min_u32 v131, v132, v131                         // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v130, 5, v[vgprSerial]               // v130 = v[vgprSerial] / 32
v_and_b32 v133, 1, v130                            // v133 = v130 % 2
v_lshrrev_b32 v130, 4, v131                        // v130 = v131 / 16
v_and_b32 v134, 1, v130                            // v134 = v130 % 2
v_cmp_eq_u32 s31, v134, v133                       // wave_id == block_belong_to_wave?
v_cndmask_b32 v131, v132, v131, s31                // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v132, 4, v131                        // v132 = v131 / 16
v_lshlrev_b32 v134, 0x0, v133                      // v134 = v133 * 1
_v_sub_u32 v132, v132, v134                        // 

/* gbReg: glvw block id */
v_lshrrev_b32 v134, 3, v131                        // v134 = v131 / 8

/* tgbReg: glvw block id */
v_lshrrev_b32 v130, 0, v[vgprSerial]               // v130 = v[vgprSerial] / 1
v_and_b32 v135, 15, v130                           // v135 = v130 % 16
                                                   // v135 = v135 * 1 (multiplier is 1, do nothing)
v_lshrrev_b32 v135, 3, v135                        // v135 = v135 / 8
v_lshlrev_b32 v133, 0x1, v133                      // v133 = v133 * 2
_v_add_co_u32 v135, vcc_lo, v133, v135             // tgbReg = (tid_coal * continOut) / GLVW
_v_sub_u32 v134, v134, v135                        // 

/* vwReg: glvw in which vw block? */
v_and_b32 v133, 0, v131                            // permute register between threads
v_lshrrev_b32 v133, 3, v133                        // permute register between threads

/* rReg : reminder of M_size % GlobalLoadVectorWidth */
v_and_b32 v135, 7, v131                            // v135 = v131 % 8
v_cmp_eq_u32 vcc_lo, v135, 0x1                     // wgMT%VW == 1
s_cbranch_vccnz label_0029                         // branch to shift d0 r=1
v_cmp_eq_u32 vcc_lo, v135, 0x2                     // wgMT%VW == 2
s_cbranch_vccnz label_0038                         // branch to shift d0 r=2
v_cmp_eq_u32 vcc_lo, v135, 0x3                     // wgMT%VW == 3
s_cbranch_vccnz label_0047                         // branch to shift d0 r=3
v_cmp_eq_u32 vcc_lo, v135, 0x4                     // wgMT%VW == 4
s_cbranch_vccnz label_0056                         // branch to shift d0 r=4
v_cmp_eq_u32 vcc_lo, v135, 0x5                     // wgMT%VW == 5
s_cbranch_vccnz label_0065                         // branch to shift d0 r=5
v_cmp_eq_u32 vcc_lo, v135, 0x6                     // wgMT%VW == 6
s_cbranch_vccnz label_0074                         // branch to shift d0 r=6
v_cmp_eq_u32 vcc_lo, v135, 0x7                     // wgMT%VW == 7
s_cbranch_vccnz label_0083                         // branch to shift d0 r=7
s_branch label_0092                                // no shifting

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_0029:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0030                         // branch to shift d0 r1 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0032                         // branch to shift d0 r1 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0034                         // branch to shift d0 r1 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0036                         // branch to shift d0 r1 mb3

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_0038:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0039                         // branch to shift d0 r2 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0041                         // branch to shift d0 r2 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0043                         // branch to shift d0 r2 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0045                         // branch to shift d0 r2 mb3

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_0047:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0048                         // branch to shift d0 r3 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0050                         // branch to shift d0 r3 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0052                         // branch to shift d0 r3 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0054                         // branch to shift d0 r3 mb3

/******************************************/
/* shift d0 r=4                           */
/******************************************/
label_0056:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0057                         // branch to shift d0 r4 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0059                         // branch to shift d0 r4 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0061                         // branch to shift d0 r4 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0063                         // branch to shift d0 r4 mb3

/******************************************/
/* shift d0 r=5                           */
/******************************************/
label_0065:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0066                         // branch to shift d0 r5 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0068                         // branch to shift d0 r5 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0070                         // branch to shift d0 r5 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0072                         // branch to shift d0 r5 mb3

/******************************************/
/* shift d0 r=6                           */
/******************************************/
label_0074:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0075                         // branch to shift d0 r6 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0077                         // branch to shift d0 r6 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0079                         // branch to shift d0 r6 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0081                         // branch to shift d0 r6 mb3

/******************************************/
/* shift d0 r=7                           */
/******************************************/
label_0083:
v_cmp_eq_u32 vcc_lo, v132, 0x0                     // 
s_cbranch_vccnz label_0084                         // branch to shift d0 r7 mb0
v_cmp_eq_u32 vcc_lo, v132, 0x2                     // 
s_cbranch_vccnz label_0086                         // branch to shift d0 r7 mb1
v_cmp_eq_u32 vcc_lo, v132, 0x4                     // 
s_cbranch_vccnz label_0088                         // branch to shift d0 r7 mb2
v_cmp_eq_u32 vcc_lo, v132, 0x6                     // 
s_cbranch_vccnz label_0090                         // branch to shift d0 r7 mb3

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_0030: // r1 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0031                         // branch to shift d0 r1 mb0 vw0

/******************************************/
/* shift d0 r=1 mb=1                      */
/******************************************/
label_0032: // r1 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0033                         // branch to shift d0 r1 mb1 vw0

/******************************************/
/* shift d0 r=1 mb=2                      */
/******************************************/
label_0034: // r1 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0035                         // branch to shift d0 r1 mb2 vw0

/******************************************/
/* shift d0 r=1 mb=3                      */
/******************************************/
label_0036: // r1 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0037                         // branch to shift d0 r1 mb3 vw0

/******************************************/
/* shift d0 r=2 mb=0                      */
/******************************************/
label_0039: // r2 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0040                         // branch to shift d0 r2 mb0 vw0

/******************************************/
/* shift d0 r=2 mb=1                      */
/******************************************/
label_0041: // r2 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0042                         // branch to shift d0 r2 mb1 vw0

/******************************************/
/* shift d0 r=2 mb=2                      */
/******************************************/
label_0043: // r2 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0044                         // branch to shift d0 r2 mb2 vw0

/******************************************/
/* shift d0 r=2 mb=3                      */
/******************************************/
label_0045: // r2 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0046                         // branch to shift d0 r2 mb3 vw0

/******************************************/
/* shift d0 r=3 mb=0                      */
/******************************************/
label_0048: // r3 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0049                         // branch to shift d0 r3 mb0 vw0

/******************************************/
/* shift d0 r=3 mb=1                      */
/******************************************/
label_0050: // r3 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0051                         // branch to shift d0 r3 mb1 vw0

/******************************************/
/* shift d0 r=3 mb=2                      */
/******************************************/
label_0052: // r3 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0053                         // branch to shift d0 r3 mb2 vw0

/******************************************/
/* shift d0 r=3 mb=3                      */
/******************************************/
label_0054: // r3 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0055                         // branch to shift d0 r3 mb3 vw0

/******************************************/
/* shift d0 r=4 mb=0                      */
/******************************************/
label_0057: // r4 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0058                         // branch to shift d0 r4 mb0 vw0

/******************************************/
/* shift d0 r=4 mb=1                      */
/******************************************/
label_0059: // r4 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0060                         // branch to shift d0 r4 mb1 vw0

/******************************************/
/* shift d0 r=4 mb=2                      */
/******************************************/
label_0061: // r4 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0062                         // branch to shift d0 r4 mb2 vw0

/******************************************/
/* shift d0 r=4 mb=3                      */
/******************************************/
label_0063: // r4 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0064                         // branch to shift d0 r4 mb3 vw0

/******************************************/
/* shift d0 r=5 mb=0                      */
/******************************************/
label_0066: // r5 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0067                         // branch to shift d0 r5 mb0 vw0

/******************************************/
/* shift d0 r=5 mb=1                      */
/******************************************/
label_0068: // r5 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0069                         // branch to shift d0 r5 mb1 vw0

/******************************************/
/* shift d0 r=5 mb=2                      */
/******************************************/
label_0070: // r5 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0071                         // branch to shift d0 r5 mb2 vw0

/******************************************/
/* shift d0 r=5 mb=3                      */
/******************************************/
label_0072: // r5 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0073                         // branch to shift d0 r5 mb3 vw0

/******************************************/
/* shift d0 r=6 mb=0                      */
/******************************************/
label_0075: // r6 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0076                         // branch to shift d0 r6 mb0 vw0

/******************************************/
/* shift d0 r=6 mb=1                      */
/******************************************/
label_0077: // r6 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0078                         // branch to shift d0 r6 mb1 vw0

/******************************************/
/* shift d0 r=6 mb=2                      */
/******************************************/
label_0079: // r6 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0080                         // branch to shift d0 r6 mb2 vw0

/******************************************/
/* shift d0 r=6 mb=3                      */
/******************************************/
label_0081: // r6 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0082                         // branch to shift d0 r6 mb3 vw0

/******************************************/
/* shift d0 r=7 mb=0                      */
/******************************************/
label_0084: // r7 mb0 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0085                         // branch to shift d0 r7 mb0 vw0

/******************************************/
/* shift d0 r=7 mb=1                      */
/******************************************/
label_0086: // r7 mb1 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0087                         // branch to shift d0 r7 mb1 vw0

/******************************************/
/* shift d0 r=7 mb=2                      */
/******************************************/
label_0088: // r7 mb2 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0089                         // branch to shift d0 r7 mb2 vw0

/******************************************/
/* shift d0 r=7 mb=3                      */
/******************************************/
label_0090: // r7 mb3 
v_cmp_eq_u32 vcc_lo, v133, 0x0                     // 
s_cbranch_vccnz label_0091                         // branch to shift d0 r7 mb3 vw0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_0031: // r1 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=1 mb=1 vw0                  */
/******************************************/
label_0033: // r1 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:28            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=1 mb=2 vw0                  */
/******************************************/
label_0035: // r1 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=1 mb=3 vw0                  */
/******************************************/
label_0037: // r1 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:28          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:28        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_0040: // r2 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=2 mb=1 vw0                  */
/******************************************/
label_0042: // r2 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:24            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=2 mb=2 vw0                  */
/******************************************/
label_0044: // r2 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=2 mb=3 vw0                  */
/******************************************/
label_0046: // r2 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:24          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:24        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_0049: // r3 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=3 mb=1 vw0                  */
/******************************************/
label_0051: // r3 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:20            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=3 mb=2 vw0                  */
/******************************************/
label_0053: // r3 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=3 mb=3 vw0                  */
/******************************************/
label_0055: // r3 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:20          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:20        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=4 mb=0 vw0                  */
/******************************************/
label_0058: // r4 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=4 mb=1 vw0                  */
/******************************************/
label_0060: // r4 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:16            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=4 mb=2 vw0                  */
/******************************************/
label_0062: // r4 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=4 mb=3 vw0                  */
/******************************************/
label_0064: // r4 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:16          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:16        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=5 mb=0 vw0                  */
/******************************************/
label_0067: // r5 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=5 mb=1 vw0                  */
/******************************************/
label_0069: // r5 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:12            // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=5 mb=2 vw0                  */
/******************************************/
label_0071: // r5 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=5 mb=3 vw0                  */
/******************************************/
label_0073: // r5 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:12          // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:12        // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=6 mb=0 vw0                  */
/******************************************/
label_0076: // r6 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=6 mb=1 vw0                  */
/******************************************/
label_0078: // r6 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:8             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=6 mb=2 vw0                  */
/******************************************/
label_0080: // r6 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=6 mb=3 vw0                  */
/******************************************/
label_0082: // r6 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:8           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:8         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=7 mb=0 vw0                  */
/******************************************/
label_0085: // r7 mb0 vw0 
s_mov_b32 s31, 0                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v0, v130, v0, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v1, v130, v1, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v2, v130, v2, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v3, v130, v3, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v4, v130, v4, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v5, v130, v5, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v6, v130, v6, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v7, v130, v7, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v32, v130, v32, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v33, v130, v33, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v34, v130, v34, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v35, v130, v35, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v36, v130, v36, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v37, v130, v37, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v38, v130, v38, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v39, v130, v39, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v64, v130, v64, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v65, v130, v65, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v66, v130, v66, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v67, v130, v67, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v68, v130, v68, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v69, v130, v69, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v70, v130, v70, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v71, v130, v71, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v96, v130, v96, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v97, v130, v97, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v98, v130, v98, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v99, v130, v99, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v100, v130, v100, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v101, v130, v101, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v102, v130, v102, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v103, v130, v103, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=7 mb=1 vw0                  */
/******************************************/
label_0087: // r7 mb1 vw0 
s_mov_b32 s31, 4                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v8, v130, v8, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v9, v130, v9, offset:4             // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v10, v130, v10, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v11, v130, v11, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v12, v130, v12, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v13, v130, v13, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v14, v130, v14, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v15, v130, v15, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v40, v130, v40, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v41, v130, v41, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v42, v130, v42, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v43, v130, v43, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v44, v130, v44, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v45, v130, v45, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v46, v130, v46, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v47, v130, v47, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v72, v130, v72, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v73, v130, v73, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v74, v130, v74, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v75, v130, v75, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v76, v130, v76, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v77, v130, v77, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v78, v130, v78, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v79, v130, v79, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v104, v130, v104, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v105, v130, v105, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v106, v130, v106, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v107, v130, v107, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v108, v130, v108, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v109, v130, v109, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v110, v130, v110, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v111, v130, v111, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=7 mb=2 vw0                  */
/******************************************/
label_0089: // r7 mb2 vw0 
s_mov_b32 s31, 8                                   // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v16, v130, v16, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v17, v130, v17, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v18, v130, v18, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v19, v130, v19, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v20, v130, v20, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v21, v130, v21, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v22, v130, v22, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v23, v130, v23, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v48, v130, v48, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v49, v130, v49, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v50, v130, v50, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v51, v130, v51, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v52, v130, v52, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v53, v130, v53, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v54, v130, v54, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v55, v130, v55, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v80, v130, v80, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v81, v130, v81, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v82, v130, v82, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v83, v130, v83, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v84, v130, v84, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v85, v130, v85, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v86, v130, v86, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v87, v130, v87, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v112, v130, v112, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v113, v130, v113, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v114, v130, v114, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v115, v130, v115, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v116, v130, v116, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v117, v130, v117, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v118, v130, v118, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v119, v130, v119, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting


/******************************************/
/* shift d0 r=7 mb=3 vw0                  */
/******************************************/
label_0091: // r7 mb3 vw0 
s_mov_b32 s31, 12                                  // 
_v_cmpx_eq_u32 s31, v134, s31                      // is thread in edge glvw region
v_and_b32 v130, 31, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v130, 2, v130                        // permute register between threads
ds_bpermute_b32 v24, v130, v24, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v25, v130, v25, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v26, v130, v26, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v27, v130, v27, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v28, v130, v28, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v29, v130, v29, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v30, v130, v30, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v31, v130, v31, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v56, v130, v56, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v57, v130, v57, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v58, v130, v58, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v59, v130, v59, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v60, v130, v60, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v61, v130, v61, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v62, v130, v62, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v63, v130, v63, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v88, v130, v88, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v89, v130, v89, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v90, v130, v90, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v91, v130, v91, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v92, v130, v92, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v93, v130, v93, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v94, v130, v94, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v95, v130, v95, offset:4           // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v120, v130, v120, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v121, v130, v121, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v122, v130, v122, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v123, v130, v123, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v124, v130, v124, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v125, v130, v125, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v126, v130, v126, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
ds_bpermute_b32 v127, v130, v127, offset:4         // permute edge values
s_waitcnt 0                                        // wait for swizzle operation
s_mov_b32 s31, 0xFFFFFFFF                          // to restore all threads active
s_or_saveexec_b32 vcc_lo, s31                      // all threads active
s_branch label_0092                                // done shifting

label_0092: // end shift0



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v134, 5, v[vgprSerial]               // v134 = v[vgprSerial] / 32
v_and_b32 v131, 31, v[vgprSerial]                  // v131 = v[vgprSerial] % 32
v_lshrrev_b32 v131, 4, v131                        // v131 = v131 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_lshrrev_b32 v135, 1, v134                        // v135 = v134 / 2
v_mul_lo_u32 v135, 0x10, v135                      // wave coordination offset 1
_v_add_u32 v131, v135, v131                        // coordination 1 = wave_id1 + tid1
v_mul_lo_u32 v132, v131, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v133, v131, s[sgprStrideD1J]          //  offset 1
v_and_b32 v135, 1, v134                            // v135 = v134 % 2
v_mul_lo_u32 v135, 0x10, v135                      // wave coordination offset 0
v_and_b32 v130, 15, v[vgprSerial]                  // v130 = v[vgprSerial] % 16
_v_add_lshl_u32 v130, v135, v130, 0                // coordination 0 = wave_id0 + tid0
s_mul_i32 s31, 128, s[sgprWorkGroup0]              // wgp0 * MT0
_v_add_u32 v130, s31, v130                         // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s31, 128, s[sgprWorkGroup1]              // wgp1 * MT1
_v_add_u32 v131, s31, v131                         // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1


/* not-LocalSplitU: global write */

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 GW_Beta_115                         // Branch if Beta is not zero

s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_106                        // jump if edges required
s_and_b32 s32, 127, s[sgprSizeJ]                   // s32 = s[sgprSizeJ] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_106                        // jump if edges required
GW_B0_E0_103:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=114 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (16,3,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (17,3,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (18,3,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (19,3,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (20,3,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (21,3,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (22,3,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (23,3,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (24,3,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (25,3,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (26,3,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (27,3,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
/* (d1,vc1,d0,vc0)=(16,0,3,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
/* (d1,vc1,d0,vc0)=(17,0,3,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
/* (d1,vc1,d0,vc0)=(18,0,3,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
/* (d1,vc1,d0,vc0)=(19,0,3,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
/* (d1,vc1,d0,vc0)=(20,0,3,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
/* (d1,vc1,d0,vc0)=(21,0,3,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
/* (d1,vc1,d0,vc0)=(22,0,3,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
/* (d1,vc1,d0,vc0)=(23,0,3,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
/* (d1,vc1,d0,vc0)=(24,0,3,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
/* (d1,vc1,d0,vc0)=(25,0,3,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
/* (d1,vc1,d0,vc0)=(26,0,3,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
/* (d1,vc1,d0,vc0)=(27,0,3,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
_v_add_lshl_u32 v136, v133, v130, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=130, coord0Vgpr=130

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (16, 3, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (17, 3, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (18, 3, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (19, 3, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (20, 3, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (21, 3, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (22, 3, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (23, 3, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (24, 3, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (25, 3, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (26, 3, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (27, 3, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+200], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+202], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+206], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+208], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+212], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+214], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+221], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+223], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+227], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+229], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+231], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+233], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+235], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+237], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+239], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+241], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+243], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+245], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+247], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+249], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+251], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+253], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+254], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v140, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v142, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v144, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v146, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v148, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v150, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v152, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v153, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v154, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v155, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v156, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v157, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v158, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v159, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v160, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v161, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v162, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v163, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v164, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v165, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v166, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v167, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v168, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v169, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v170, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v171, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v172, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v173, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v174, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v175, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v176, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v177, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v178, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v179, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v180, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v181, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v182, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v183, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v184, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v185, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v186, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v187, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v188, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v189, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v190, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v191, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v192, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v193, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v194, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v195, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v196, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v197, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v198, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v199, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v200, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v201, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v202, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v203, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v204, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v205, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v206, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v207, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v208, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v209, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v210, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v211, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v212, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v213, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v214, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v215, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v216, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v217, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v221, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v222, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v223, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v224, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v225, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v226, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v227, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v228, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v229, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v230, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v231, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v232, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v233, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v234, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v235, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v236, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v237, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v238, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v239, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v240, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v241, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v242, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v243, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v244, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v245, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v246, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v247, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v248, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v249, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v250, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v251, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v252, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v253, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v254, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Batch #1 (d1,d0,vc1,vc0) = */
/*    (28,2,0,0:vw1); (28,3,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (29,3,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (30,3,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1); (31,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
/* (d1,vc1,d0,vc0)=(28,0,3,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
/* (d1,vc1,d0,vc0)=(29,0,3,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
/* (d1,vc1,d0,vc0)=(30,0,3,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
/* (d1,vc1,d0,vc0)=(31,0,3,0) */

/* rC *= alpha batchElements=[(28, 2, 0, 0), (28, 3, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (29, 3, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (30, 3, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0), (31, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v140, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v142, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v144, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v146, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v148, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b16 v150, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_114                          // jump to end
GW_B0_E1_106:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=58 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v138, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v140, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v140, -1, v140, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v144, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v146, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v150, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v152, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, -1, v152, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v156, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, -1, v156, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v158, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v162, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v162, -1, v162, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v164, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v168, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v168, -1, v168, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v170, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v170, -1, v170, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v174, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v174, -1, v174, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v176, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, -1, v176, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v178, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v180, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, -1, v180, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v182, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, -1, v182, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v184, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v186, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, -1, v186, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v188, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, -1, v188, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v190, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v192, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, -1, v192, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v194, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, -1, v194, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v196, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v198, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, -1, v198, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v200, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, -1, v200, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v202, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v204, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, -1, v204, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v206, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, -1, v206, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v208, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v210, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, -1, v210, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v212, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, -1, v212, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v214, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v216, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, -1, v216, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v221, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, -1, v221, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v223, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v225, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, -1, v225, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v227, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, -1, v227, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v229, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v231, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, -1, v231, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v233, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, -1, v233, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v235, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v237, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, -1, v237, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v239, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, -1, v239, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v241, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v243, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, -1, v243, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v245, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, -1, v245, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v247, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v249, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, -1, v249, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v251, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, -1, v251, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v253, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, -1, v253, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+254], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v137, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v139, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v141, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v143, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v145, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v147, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v149, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v151, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v153, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v155, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v157, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v159, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v161, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v163, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v165, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v167, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v169, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v171, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v173, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v175, v174, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v177, v176, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v179, v178, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v181, v180, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v183, v182, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v185, v184, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v187, v186, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v189, v188, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v191, v190, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v193, v192, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v195, v194, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v197, v196, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v199, v198, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v201, v200, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v203, v202, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v205, v204, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v207, v206, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v209, v208, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v211, v210, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v213, v212, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v215, v214, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v217, v216, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v222, v221, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v224, v223, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v226, v225, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v228, v227, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v230, v229, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v232, v231, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v234, v233, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v236, v235, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v238, v237, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v240, v239, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v242, v241, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v244, v243, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v246, v245, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v248, v247, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v250, v249, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v252, v251, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v254, v253, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (16,3,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (17,3,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (18,3,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (19,3,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (20,3,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (21,3,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (22,3,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (23,3,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (24,3,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (25,3,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (26,3,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (27,3,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (28,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v138, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v140, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v140, -1, v140, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v144, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v146, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v150, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v152, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, -1, v152, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v156, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, -1, v156, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v158, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v162, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v162, -1, v162, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v164, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v168, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v168, -1, v168, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v170, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v170, -1, v170, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v174, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v174, -1, v174, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v176, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, -1, v176, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v178, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v180, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, -1, v180, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v182, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, -1, v182, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v184, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v186, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, -1, v186, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v188, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, -1, v188, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v190, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v192, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, -1, v192, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v194, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, -1, v194, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v196, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v198, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, -1, v198, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v200, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, -1, v200, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v202, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v204, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, -1, v204, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v206, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, -1, v206, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v208, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v210, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, -1, v210, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v212, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, -1, v212, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v214, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v216, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, -1, v216, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v221, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, -1, v221, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v223, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v225, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, -1, v225, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v227, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, -1, v227, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v229, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v231, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, -1, v231, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v233, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, -1, v233, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v235, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v237, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, -1, v237, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v239, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, -1, v239, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v241, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v243, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, -1, v243, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v245, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, -1, v245, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v247, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v249, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, -1, v249, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v251, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, -1, v251, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v253, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, -1, v253, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (16, 3, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (17, 3, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (18, 3, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (19, 3, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (20, 3, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (21, 3, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (22, 3, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (23, 3, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (24, 3, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (25, 3, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (26, 3, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (27, 3, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (28, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+254], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v137, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v139, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v141, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v143, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v145, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v147, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v149, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v151, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v153, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v155, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v157, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v159, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v161, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v163, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v165, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v167, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v169, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v171, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v173, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v175, v174, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v177, v176, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v179, v178, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v181, v180, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v183, v182, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v185, v184, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v187, v186, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v189, v188, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v191, v190, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v193, v192, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v195, v194, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v197, v196, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v199, v198, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v201, v200, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v203, v202, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v205, v204, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v207, v206, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v209, v208, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v211, v210, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v213, v212, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v215, v214, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v217, v216, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v222, v221, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v224, v223, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v226, v225, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v228, v227, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v230, v229, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v232, v231, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v234, v233, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v236, v235, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v238, v237, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v240, v239, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v242, v241, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v244, v243, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v246, v245, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v248, v247, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v250, v249, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v252, v251, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v254, v253, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (29,3,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (30,3,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1); (31,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v138, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v140, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v140, -1, v140, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v144, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v146, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v150, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v152, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, -1, v152, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v156, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, -1, v156, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v158, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (29, 3, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (30, 3, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0), (31, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
_buffer_store_b16 v137, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v139, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v141, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v143, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v145, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v147, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v149, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v151, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v153, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v155, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v157, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b16 v159, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_114                          // jump to end
GW_Beta_115:
s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_113                        // jump if edges required
s_and_b32 s32, 127, s[sgprSizeJ]                   // s32 = s[sgprSizeJ] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_113                        // jump if edges required
GW_B1_E0_110:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=56 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v137, v132, v130, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=130, coord0Vgpr=130
_buffer_load_d16_b16 v138, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_buffer_load_d16_b16 v140, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_buffer_load_d16_b16 v142, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_buffer_load_d16_b16 v144, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v146, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_buffer_load_d16_b16 v148, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_buffer_load_d16_b16 v150, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_buffer_load_d16_b16 v152, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v154, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_buffer_load_d16_b16 v156, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_buffer_load_d16_b16 v158, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_buffer_load_d16_b16 v160, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v162, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_buffer_load_d16_b16 v164, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_buffer_load_d16_b16 v166, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_buffer_load_d16_b16 v168, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v170, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_buffer_load_d16_b16 v172, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_buffer_load_d16_b16 v174, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
_buffer_load_d16_b16 v176, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v178, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_buffer_load_d16_b16 v180, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_buffer_load_d16_b16 v182, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
_buffer_load_d16_b16 v184, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v186, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_buffer_load_d16_b16 v188, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_buffer_load_d16_b16 v190, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
_buffer_load_d16_b16 v192, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v194, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_buffer_load_d16_b16 v196, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_buffer_load_d16_b16 v198, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
_buffer_load_d16_b16 v200, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 36                // scale StrideC *= numRows(18) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v202, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
_buffer_load_d16_b16 v204, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
_buffer_load_d16_b16 v206, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
_buffer_load_d16_b16 v208, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v210, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
_buffer_load_d16_b16 v212, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
_buffer_load_d16_b16 v214, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
_buffer_load_d16_b16 v216, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v221, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
_buffer_load_d16_b16 v223, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
_buffer_load_d16_b16 v225, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
_buffer_load_d16_b16 v227, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v229, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
_buffer_load_d16_b16 v231, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
_buffer_load_d16_b16 v233, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
_buffer_load_d16_b16 v235, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v237, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
_buffer_load_d16_b16 v239, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
_buffer_load_d16_b16 v241, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
_buffer_load_d16_b16 v243, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v245, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
_buffer_load_d16_b16 v247, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
_buffer_load_d16_b16 v249, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
_buffer_load_d16_b16 v251, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
_v_add_lshl_u32 v136, v133, v130, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=130, coord0Vgpr=130

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(55)                                // wait C (interleaved) 55 = 56 - 0 - 1
v_pk_mul_f16 v138, s[sgprBeta], v138               // v138 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+139], v138, v[vgprValuC+139] // sum*alpha + C*beta
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(54)                                // wait C (interleaved) 54 = 56 - 1 - 1
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(53)                                // wait C (interleaved) 53 = 56 - 2 - 1
v_pk_mul_f16 v142, s[sgprBeta], v142               // v142 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+143], v142, v[vgprValuC+143] // sum*alpha + C*beta
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(52)                                // wait C (interleaved) 52 = 56 - 3 - 1
v_pk_mul_f16 v144, s[sgprBeta], v144               // v144 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+145], v144, v[vgprValuC+145] // sum*alpha + C*beta
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(51)                                // wait C (interleaved) 51 = 56 - 4 - 1
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(50)                                // wait C (interleaved) 50 = 56 - 5 - 1
v_pk_mul_f16 v148, s[sgprBeta], v148               // v148 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+149], v148, v[vgprValuC+149] // sum*alpha + C*beta
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(49)                                // wait C (interleaved) 49 = 56 - 6 - 1
v_pk_mul_f16 v150, s[sgprBeta], v150               // v150 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+151], v150, v[vgprValuC+151] // sum*alpha + C*beta
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(48)                                // wait C (interleaved) 48 = 56 - 7 - 1
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 56 - 8 - 1
v_pk_mul_f16 v154, s[sgprBeta], v154               // v154 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+155], v154, v[vgprValuC+155] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v155, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 56 - 9 - 1
v_pk_mul_f16 v156, s[sgprBeta], v156               // v156 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+157], v156, v[vgprValuC+157] // sum*alpha + C*beta
_buffer_store_b16 v157, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 56 - 10 - 1
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 56 - 11 - 1
v_pk_mul_f16 v160, s[sgprBeta], v160               // v160 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+161], v160, v[vgprValuC+161] // sum*alpha + C*beta
_buffer_store_b16 v161, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 56 - 12 - 1
v_pk_mul_f16 v162, s[sgprBeta], v162               // v162 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+163], v162, v[vgprValuC+163] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v163, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 56 - 13 - 1
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 56 - 14 - 1
v_pk_mul_f16 v166, s[sgprBeta], v166               // v166 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+167], v166, v[vgprValuC+167] // sum*alpha + C*beta
_buffer_store_b16 v167, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 56 - 15 - 1
v_pk_mul_f16 v168, s[sgprBeta], v168               // v168 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+169], v168, v[vgprValuC+169] // sum*alpha + C*beta
_buffer_store_b16 v169, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 56 - 16 - 1
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=16 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v171, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 56 - 17 - 1
v_pk_mul_f16 v172, s[sgprBeta], v172               // v172 = C*beta ei=17 vi=0
v_pk_add_f16 v[vgprValuC+173], v172, v[vgprValuC+173] // sum*alpha + C*beta
_buffer_store_b16 v173, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 56 - 18 - 1
v_pk_mul_f16 v174, s[sgprBeta], v174               // v174 = C*beta ei=18 vi=0
v_pk_add_f16 v[vgprValuC+175], v174, v[vgprValuC+175] // sum*alpha + C*beta
_buffer_store_b16 v175, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 56 - 19 - 1
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=19 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 56 - 20 - 1
v_pk_mul_f16 v178, s[sgprBeta], v178               // v178 = C*beta ei=20 vi=0
v_pk_add_f16 v[vgprValuC+179], v178, v[vgprValuC+179] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v179, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 56 - 21 - 1
v_pk_mul_f16 v180, s[sgprBeta], v180               // v180 = C*beta ei=21 vi=0
v_pk_add_f16 v[vgprValuC+181], v180, v[vgprValuC+181] // sum*alpha + C*beta
_buffer_store_b16 v181, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 56 - 22 - 1
v_pk_mul_f16 v182, s[sgprBeta], v182               // v182 = C*beta ei=22 vi=0
v_pk_add_f16 v[vgprValuC+183], v182, v[vgprValuC+183] // sum*alpha + C*beta
_buffer_store_b16 v183, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 56 - 23 - 1
v_pk_mul_f16 v184, s[sgprBeta], v184               // v184 = C*beta ei=23 vi=0
v_pk_add_f16 v[vgprValuC+185], v184, v[vgprValuC+185] // sum*alpha + C*beta
_buffer_store_b16 v185, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 56 - 24 - 1
v_pk_mul_f16 v186, s[sgprBeta], v186               // v186 = C*beta ei=24 vi=0
v_pk_add_f16 v[vgprValuC+187], v186, v[vgprValuC+187] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v187, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 56 - 25 - 1
v_pk_mul_f16 v188, s[sgprBeta], v188               // v188 = C*beta ei=25 vi=0
v_pk_add_f16 v[vgprValuC+189], v188, v[vgprValuC+189] // sum*alpha + C*beta
_buffer_store_b16 v189, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 56 - 26 - 1
v_pk_mul_f16 v190, s[sgprBeta], v190               // v190 = C*beta ei=26 vi=0
v_pk_add_f16 v[vgprValuC+191], v190, v[vgprValuC+191] // sum*alpha + C*beta
_buffer_store_b16 v191, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 56 - 27 - 1
v_pk_mul_f16 v192, s[sgprBeta], v192               // v192 = C*beta ei=27 vi=0
v_pk_add_f16 v[vgprValuC+193], v192, v[vgprValuC+193] // sum*alpha + C*beta
_buffer_store_b16 v193, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 56 - 28 - 1
v_pk_mul_f16 v194, s[sgprBeta], v194               // v194 = C*beta ei=28 vi=0
v_pk_add_f16 v[vgprValuC+195], v194, v[vgprValuC+195] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v195, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 56 - 29 - 1
v_pk_mul_f16 v196, s[sgprBeta], v196               // v196 = C*beta ei=29 vi=0
v_pk_add_f16 v[vgprValuC+197], v196, v[vgprValuC+197] // sum*alpha + C*beta
_buffer_store_b16 v197, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 56 - 30 - 1
v_pk_mul_f16 v198, s[sgprBeta], v198               // v198 = C*beta ei=30 vi=0
v_pk_add_f16 v[vgprValuC+199], v198, v[vgprValuC+199] // sum*alpha + C*beta
_buffer_store_b16 v199, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 56 - 31 - 1
v_pk_mul_f16 v200, s[sgprBeta], v200               // v200 = C*beta ei=31 vi=0
v_pk_add_f16 v[vgprValuC+201], v200, v[vgprValuC+201] // sum*alpha + C*beta
_buffer_store_b16 v201, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 56 - 32 - 1
v_pk_mul_f16 v202, s[sgprBeta], v202               // v202 = C*beta ei=32 vi=0
v_pk_add_f16 v[vgprValuC+203], v202, v[vgprValuC+203] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v203, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 56 - 33 - 1
v_pk_mul_f16 v204, s[sgprBeta], v204               // v204 = C*beta ei=33 vi=0
v_pk_add_f16 v[vgprValuC+205], v204, v[vgprValuC+205] // sum*alpha + C*beta
_buffer_store_b16 v205, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 56 - 34 - 1
v_pk_mul_f16 v206, s[sgprBeta], v206               // v206 = C*beta ei=34 vi=0
v_pk_add_f16 v[vgprValuC+207], v206, v[vgprValuC+207] // sum*alpha + C*beta
_buffer_store_b16 v207, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 56 - 35 - 1
v_pk_mul_f16 v208, s[sgprBeta], v208               // v208 = C*beta ei=35 vi=0
v_pk_add_f16 v[vgprValuC+209], v208, v[vgprValuC+209] // sum*alpha + C*beta
_buffer_store_b16 v209, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 56 - 36 - 1
v_pk_mul_f16 v210, s[sgprBeta], v210               // v210 = C*beta ei=36 vi=0
v_pk_add_f16 v[vgprValuC+211], v210, v[vgprValuC+211] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v211, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 56 - 37 - 1
v_pk_mul_f16 v212, s[sgprBeta], v212               // v212 = C*beta ei=37 vi=0
v_pk_add_f16 v[vgprValuC+213], v212, v[vgprValuC+213] // sum*alpha + C*beta
_buffer_store_b16 v213, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 56 - 38 - 1
v_pk_mul_f16 v214, s[sgprBeta], v214               // v214 = C*beta ei=38 vi=0
v_pk_add_f16 v[vgprValuC+215], v214, v[vgprValuC+215] // sum*alpha + C*beta
_buffer_store_b16 v215, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 56 - 39 - 1
v_pk_mul_f16 v216, s[sgprBeta], v216               // v216 = C*beta ei=39 vi=0
v_pk_add_f16 v[vgprValuC+217], v216, v[vgprValuC+217] // sum*alpha + C*beta
_buffer_store_b16 v217, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 56 - 40 - 1
v_pk_mul_f16 v221, s[sgprBeta], v221               // v221 = C*beta ei=40 vi=0
v_pk_add_f16 v[vgprValuC+222], v221, v[vgprValuC+222] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v222, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 56 - 41 - 1
v_pk_mul_f16 v223, s[sgprBeta], v223               // v223 = C*beta ei=41 vi=0
v_pk_add_f16 v[vgprValuC+224], v223, v[vgprValuC+224] // sum*alpha + C*beta
_buffer_store_b16 v224, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 56 - 42 - 1
v_pk_mul_f16 v225, s[sgprBeta], v225               // v225 = C*beta ei=42 vi=0
v_pk_add_f16 v[vgprValuC+226], v225, v[vgprValuC+226] // sum*alpha + C*beta
_buffer_store_b16 v226, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 56 - 43 - 1
v_pk_mul_f16 v227, s[sgprBeta], v227               // v227 = C*beta ei=43 vi=0
v_pk_add_f16 v[vgprValuC+228], v227, v[vgprValuC+228] // sum*alpha + C*beta
_buffer_store_b16 v228, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 56 - 44 - 1
v_pk_mul_f16 v229, s[sgprBeta], v229               // v229 = C*beta ei=44 vi=0
v_pk_add_f16 v[vgprValuC+230], v229, v[vgprValuC+230] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v230, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 56 - 45 - 1
v_pk_mul_f16 v231, s[sgprBeta], v231               // v231 = C*beta ei=45 vi=0
v_pk_add_f16 v[vgprValuC+232], v231, v[vgprValuC+232] // sum*alpha + C*beta
_buffer_store_b16 v232, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 56 - 46 - 1
v_pk_mul_f16 v233, s[sgprBeta], v233               // v233 = C*beta ei=46 vi=0
v_pk_add_f16 v[vgprValuC+234], v233, v[vgprValuC+234] // sum*alpha + C*beta
_buffer_store_b16 v234, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 56 - 47 - 1
v_pk_mul_f16 v235, s[sgprBeta], v235               // v235 = C*beta ei=47 vi=0
v_pk_add_f16 v[vgprValuC+236], v235, v[vgprValuC+236] // sum*alpha + C*beta
_buffer_store_b16 v236, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 56 - 48 - 1
v_pk_mul_f16 v237, s[sgprBeta], v237               // v237 = C*beta ei=48 vi=0
v_pk_add_f16 v[vgprValuC+238], v237, v[vgprValuC+238] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v238, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 56 - 49 - 1
v_pk_mul_f16 v239, s[sgprBeta], v239               // v239 = C*beta ei=49 vi=0
v_pk_add_f16 v[vgprValuC+240], v239, v[vgprValuC+240] // sum*alpha + C*beta
_buffer_store_b16 v240, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 56 - 50 - 1
v_pk_mul_f16 v241, s[sgprBeta], v241               // v241 = C*beta ei=50 vi=0
v_pk_add_f16 v[vgprValuC+242], v241, v[vgprValuC+242] // sum*alpha + C*beta
_buffer_store_b16 v242, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 56 - 51 - 1
v_pk_mul_f16 v243, s[sgprBeta], v243               // v243 = C*beta ei=51 vi=0
v_pk_add_f16 v[vgprValuC+244], v243, v[vgprValuC+244] // sum*alpha + C*beta
_buffer_store_b16 v244, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 56 - 52 - 1
v_pk_mul_f16 v245, s[sgprBeta], v245               // v245 = C*beta ei=52 vi=0
v_pk_add_f16 v[vgprValuC+246], v245, v[vgprValuC+246] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v246, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 56 - 53 - 1
v_pk_mul_f16 v247, s[sgprBeta], v247               // v247 = C*beta ei=53 vi=0
v_pk_add_f16 v[vgprValuC+248], v247, v[vgprValuC+248] // sum*alpha + C*beta
_buffer_store_b16 v248, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 56 - 54 - 1
v_pk_mul_f16 v249, s[sgprBeta], v249               // v249 = C*beta ei=54 vi=0
v_pk_add_f16 v[vgprValuC+250], v249, v[vgprValuC+250] // sum*alpha + C*beta
_buffer_store_b16 v250, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 56 - 55 - 1
v_pk_mul_f16 v251, s[sgprBeta], v251               // v251 = C*beta ei=55 vi=0
v_pk_add_f16 v[vgprValuC+252], v251, v[vgprValuC+252] // sum*alpha + C*beta
_buffer_store_b16 v252, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (16,3,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (17,3,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (18,3,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (19,3,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (20,3,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (21,3,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (22,3,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (23,3,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (24,3,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (25,3,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (26,3,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (27,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v138, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
_buffer_load_d16_b16 v140, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
_buffer_load_d16_b16 v142, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
_buffer_load_d16_b16 v144, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v146, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
_buffer_load_d16_b16 v148, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
_buffer_load_d16_b16 v150, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
_buffer_load_d16_b16 v152, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 36                // scale StrideC *= numRows(18) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v154, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
_buffer_load_d16_b16 v156, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
_buffer_load_d16_b16 v158, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(16,0,3,0) */
_buffer_load_d16_b16 v160, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v162, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
_buffer_load_d16_b16 v164, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
_buffer_load_d16_b16 v166, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(17,0,3,0) */
_buffer_load_d16_b16 v168, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v170, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
_buffer_load_d16_b16 v172, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
_buffer_load_d16_b16 v174, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(18,0,3,0) */
_buffer_load_d16_b16 v176, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v178, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
_buffer_load_d16_b16 v180, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
_buffer_load_d16_b16 v182, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(19,0,3,0) */
_buffer_load_d16_b16 v184, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v186, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
_buffer_load_d16_b16 v188, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
_buffer_load_d16_b16 v190, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(20,0,3,0) */
_buffer_load_d16_b16 v192, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v194, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
_buffer_load_d16_b16 v196, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
_buffer_load_d16_b16 v198, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(21,0,3,0) */
_buffer_load_d16_b16 v200, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v202, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
_buffer_load_d16_b16 v204, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
_buffer_load_d16_b16 v206, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(22,0,3,0) */
_buffer_load_d16_b16 v208, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v210, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
_buffer_load_d16_b16 v212, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
_buffer_load_d16_b16 v214, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(23,0,3,0) */
_buffer_load_d16_b16 v216, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 36                // scale StrideC *= numRows(18) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v221, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
_buffer_load_d16_b16 v223, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
_buffer_load_d16_b16 v225, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(24,0,3,0) */
_buffer_load_d16_b16 v227, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v229, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
_buffer_load_d16_b16 v231, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
_buffer_load_d16_b16 v233, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(25,0,3,0) */
_buffer_load_d16_b16 v235, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v237, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
_buffer_load_d16_b16 v239, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
_buffer_load_d16_b16 v241, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(26,0,3,0) */
_buffer_load_d16_b16 v243, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v245, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
_buffer_load_d16_b16 v247, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
_buffer_load_d16_b16 v249, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(27,0,3,0) */
_buffer_load_d16_b16 v251, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc

/* rC *= alpha batchElements=[(14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (16, 3, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (17, 3, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (18, 3, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (19, 3, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (20, 3, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (21, 3, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (22, 3, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (23, 3, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (24, 3, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (25, 3, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (26, 3, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (27, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(55)                                // wait C (interleaved) 55 = 56 - 0 - 1
v_pk_mul_f16 v138, s[sgprBeta], v138               // v138 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+139], v138, v[vgprValuC+139] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(54)                                // wait C (interleaved) 54 = 56 - 1 - 1
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(53)                                // wait C (interleaved) 53 = 56 - 2 - 1
v_pk_mul_f16 v142, s[sgprBeta], v142               // v142 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+143], v142, v[vgprValuC+143] // sum*alpha + C*beta
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(52)                                // wait C (interleaved) 52 = 56 - 3 - 1
v_pk_mul_f16 v144, s[sgprBeta], v144               // v144 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+145], v144, v[vgprValuC+145] // sum*alpha + C*beta
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(51)                                // wait C (interleaved) 51 = 56 - 4 - 1
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(50)                                // wait C (interleaved) 50 = 56 - 5 - 1
v_pk_mul_f16 v148, s[sgprBeta], v148               // v148 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+149], v148, v[vgprValuC+149] // sum*alpha + C*beta
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(49)                                // wait C (interleaved) 49 = 56 - 6 - 1
v_pk_mul_f16 v150, s[sgprBeta], v150               // v150 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+151], v150, v[vgprValuC+151] // sum*alpha + C*beta
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(48)                                // wait C (interleaved) 48 = 56 - 7 - 1
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 56 - 8 - 1
v_pk_mul_f16 v154, s[sgprBeta], v154               // v154 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+155], v154, v[vgprValuC+155] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v155, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 56 - 9 - 1
v_pk_mul_f16 v156, s[sgprBeta], v156               // v156 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+157], v156, v[vgprValuC+157] // sum*alpha + C*beta
_buffer_store_b16 v157, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 56 - 10 - 1
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 56 - 11 - 1
v_pk_mul_f16 v160, s[sgprBeta], v160               // v160 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+161], v160, v[vgprValuC+161] // sum*alpha + C*beta
_buffer_store_b16 v161, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 56 - 12 - 1
v_pk_mul_f16 v162, s[sgprBeta], v162               // v162 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+163], v162, v[vgprValuC+163] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v163, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 56 - 13 - 1
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 56 - 14 - 1
v_pk_mul_f16 v166, s[sgprBeta], v166               // v166 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+167], v166, v[vgprValuC+167] // sum*alpha + C*beta
_buffer_store_b16 v167, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 56 - 15 - 1
v_pk_mul_f16 v168, s[sgprBeta], v168               // v168 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+169], v168, v[vgprValuC+169] // sum*alpha + C*beta
_buffer_store_b16 v169, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 56 - 16 - 1
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=16 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v171, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 56 - 17 - 1
v_pk_mul_f16 v172, s[sgprBeta], v172               // v172 = C*beta ei=17 vi=0
v_pk_add_f16 v[vgprValuC+173], v172, v[vgprValuC+173] // sum*alpha + C*beta
_buffer_store_b16 v173, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 56 - 18 - 1
v_pk_mul_f16 v174, s[sgprBeta], v174               // v174 = C*beta ei=18 vi=0
v_pk_add_f16 v[vgprValuC+175], v174, v[vgprValuC+175] // sum*alpha + C*beta
_buffer_store_b16 v175, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 56 - 19 - 1
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=19 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 56 - 20 - 1
v_pk_mul_f16 v178, s[sgprBeta], v178               // v178 = C*beta ei=20 vi=0
v_pk_add_f16 v[vgprValuC+179], v178, v[vgprValuC+179] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v179, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 56 - 21 - 1
v_pk_mul_f16 v180, s[sgprBeta], v180               // v180 = C*beta ei=21 vi=0
v_pk_add_f16 v[vgprValuC+181], v180, v[vgprValuC+181] // sum*alpha + C*beta
_buffer_store_b16 v181, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 56 - 22 - 1
v_pk_mul_f16 v182, s[sgprBeta], v182               // v182 = C*beta ei=22 vi=0
v_pk_add_f16 v[vgprValuC+183], v182, v[vgprValuC+183] // sum*alpha + C*beta
_buffer_store_b16 v183, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 56 - 23 - 1
v_pk_mul_f16 v184, s[sgprBeta], v184               // v184 = C*beta ei=23 vi=0
v_pk_add_f16 v[vgprValuC+185], v184, v[vgprValuC+185] // sum*alpha + C*beta
_buffer_store_b16 v185, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 56 - 24 - 1
v_pk_mul_f16 v186, s[sgprBeta], v186               // v186 = C*beta ei=24 vi=0
v_pk_add_f16 v[vgprValuC+187], v186, v[vgprValuC+187] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v187, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 56 - 25 - 1
v_pk_mul_f16 v188, s[sgprBeta], v188               // v188 = C*beta ei=25 vi=0
v_pk_add_f16 v[vgprValuC+189], v188, v[vgprValuC+189] // sum*alpha + C*beta
_buffer_store_b16 v189, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 56 - 26 - 1
v_pk_mul_f16 v190, s[sgprBeta], v190               // v190 = C*beta ei=26 vi=0
v_pk_add_f16 v[vgprValuC+191], v190, v[vgprValuC+191] // sum*alpha + C*beta
_buffer_store_b16 v191, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 56 - 27 - 1
v_pk_mul_f16 v192, s[sgprBeta], v192               // v192 = C*beta ei=27 vi=0
v_pk_add_f16 v[vgprValuC+193], v192, v[vgprValuC+193] // sum*alpha + C*beta
_buffer_store_b16 v193, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 56 - 28 - 1
v_pk_mul_f16 v194, s[sgprBeta], v194               // v194 = C*beta ei=28 vi=0
v_pk_add_f16 v[vgprValuC+195], v194, v[vgprValuC+195] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v195, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 56 - 29 - 1
v_pk_mul_f16 v196, s[sgprBeta], v196               // v196 = C*beta ei=29 vi=0
v_pk_add_f16 v[vgprValuC+197], v196, v[vgprValuC+197] // sum*alpha + C*beta
_buffer_store_b16 v197, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 56 - 30 - 1
v_pk_mul_f16 v198, s[sgprBeta], v198               // v198 = C*beta ei=30 vi=0
v_pk_add_f16 v[vgprValuC+199], v198, v[vgprValuC+199] // sum*alpha + C*beta
_buffer_store_b16 v199, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 56 - 31 - 1
v_pk_mul_f16 v200, s[sgprBeta], v200               // v200 = C*beta ei=31 vi=0
v_pk_add_f16 v[vgprValuC+201], v200, v[vgprValuC+201] // sum*alpha + C*beta
_buffer_store_b16 v201, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 56 - 32 - 1
v_pk_mul_f16 v202, s[sgprBeta], v202               // v202 = C*beta ei=32 vi=0
v_pk_add_f16 v[vgprValuC+203], v202, v[vgprValuC+203] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v203, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 56 - 33 - 1
v_pk_mul_f16 v204, s[sgprBeta], v204               // v204 = C*beta ei=33 vi=0
v_pk_add_f16 v[vgprValuC+205], v204, v[vgprValuC+205] // sum*alpha + C*beta
_buffer_store_b16 v205, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 56 - 34 - 1
v_pk_mul_f16 v206, s[sgprBeta], v206               // v206 = C*beta ei=34 vi=0
v_pk_add_f16 v[vgprValuC+207], v206, v[vgprValuC+207] // sum*alpha + C*beta
_buffer_store_b16 v207, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 56 - 35 - 1
v_pk_mul_f16 v208, s[sgprBeta], v208               // v208 = C*beta ei=35 vi=0
v_pk_add_f16 v[vgprValuC+209], v208, v[vgprValuC+209] // sum*alpha + C*beta
_buffer_store_b16 v209, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 56 - 36 - 1
v_pk_mul_f16 v210, s[sgprBeta], v210               // v210 = C*beta ei=36 vi=0
v_pk_add_f16 v[vgprValuC+211], v210, v[vgprValuC+211] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v211, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 56 - 37 - 1
v_pk_mul_f16 v212, s[sgprBeta], v212               // v212 = C*beta ei=37 vi=0
v_pk_add_f16 v[vgprValuC+213], v212, v[vgprValuC+213] // sum*alpha + C*beta
_buffer_store_b16 v213, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 56 - 38 - 1
v_pk_mul_f16 v214, s[sgprBeta], v214               // v214 = C*beta ei=38 vi=0
v_pk_add_f16 v[vgprValuC+215], v214, v[vgprValuC+215] // sum*alpha + C*beta
_buffer_store_b16 v215, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 56 - 39 - 1
v_pk_mul_f16 v216, s[sgprBeta], v216               // v216 = C*beta ei=39 vi=0
v_pk_add_f16 v[vgprValuC+217], v216, v[vgprValuC+217] // sum*alpha + C*beta
_buffer_store_b16 v217, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 56 - 40 - 1
v_pk_mul_f16 v221, s[sgprBeta], v221               // v221 = C*beta ei=40 vi=0
v_pk_add_f16 v[vgprValuC+222], v221, v[vgprValuC+222] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 36                // scale StrideD *= numRows(18) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v222, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 56 - 41 - 1
v_pk_mul_f16 v223, s[sgprBeta], v223               // v223 = C*beta ei=41 vi=0
v_pk_add_f16 v[vgprValuC+224], v223, v[vgprValuC+224] // sum*alpha + C*beta
_buffer_store_b16 v224, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 56 - 42 - 1
v_pk_mul_f16 v225, s[sgprBeta], v225               // v225 = C*beta ei=42 vi=0
v_pk_add_f16 v[vgprValuC+226], v225, v[vgprValuC+226] // sum*alpha + C*beta
_buffer_store_b16 v226, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 56 - 43 - 1
v_pk_mul_f16 v227, s[sgprBeta], v227               // v227 = C*beta ei=43 vi=0
v_pk_add_f16 v[vgprValuC+228], v227, v[vgprValuC+228] // sum*alpha + C*beta
_buffer_store_b16 v228, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 56 - 44 - 1
v_pk_mul_f16 v229, s[sgprBeta], v229               // v229 = C*beta ei=44 vi=0
v_pk_add_f16 v[vgprValuC+230], v229, v[vgprValuC+230] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v230, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 56 - 45 - 1
v_pk_mul_f16 v231, s[sgprBeta], v231               // v231 = C*beta ei=45 vi=0
v_pk_add_f16 v[vgprValuC+232], v231, v[vgprValuC+232] // sum*alpha + C*beta
_buffer_store_b16 v232, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 56 - 46 - 1
v_pk_mul_f16 v233, s[sgprBeta], v233               // v233 = C*beta ei=46 vi=0
v_pk_add_f16 v[vgprValuC+234], v233, v[vgprValuC+234] // sum*alpha + C*beta
_buffer_store_b16 v234, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 56 - 47 - 1
v_pk_mul_f16 v235, s[sgprBeta], v235               // v235 = C*beta ei=47 vi=0
v_pk_add_f16 v[vgprValuC+236], v235, v[vgprValuC+236] // sum*alpha + C*beta
_buffer_store_b16 v236, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 56 - 48 - 1
v_pk_mul_f16 v237, s[sgprBeta], v237               // v237 = C*beta ei=48 vi=0
v_pk_add_f16 v[vgprValuC+238], v237, v[vgprValuC+238] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v238, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 56 - 49 - 1
v_pk_mul_f16 v239, s[sgprBeta], v239               // v239 = C*beta ei=49 vi=0
v_pk_add_f16 v[vgprValuC+240], v239, v[vgprValuC+240] // sum*alpha + C*beta
_buffer_store_b16 v240, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 56 - 50 - 1
v_pk_mul_f16 v241, s[sgprBeta], v241               // v241 = C*beta ei=50 vi=0
v_pk_add_f16 v[vgprValuC+242], v241, v[vgprValuC+242] // sum*alpha + C*beta
_buffer_store_b16 v242, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 56 - 51 - 1
v_pk_mul_f16 v243, s[sgprBeta], v243               // v243 = C*beta ei=51 vi=0
v_pk_add_f16 v[vgprValuC+244], v243, v[vgprValuC+244] // sum*alpha + C*beta
_buffer_store_b16 v244, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 56 - 52 - 1
v_pk_mul_f16 v245, s[sgprBeta], v245               // v245 = C*beta ei=52 vi=0
v_pk_add_f16 v[vgprValuC+246], v245, v[vgprValuC+246] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v246, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 56 - 53 - 1
v_pk_mul_f16 v247, s[sgprBeta], v247               // v247 = C*beta ei=53 vi=0
v_pk_add_f16 v[vgprValuC+248], v247, v[vgprValuC+248] // sum*alpha + C*beta
_buffer_store_b16 v248, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 56 - 54 - 1
v_pk_mul_f16 v249, s[sgprBeta], v249               // v249 = C*beta ei=54 vi=0
v_pk_add_f16 v[vgprValuC+250], v249, v[vgprValuC+250] // sum*alpha + C*beta
_buffer_store_b16 v250, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 56 - 55 - 1
v_pk_mul_f16 v251, s[sgprBeta], v251               // v251 = C*beta ei=55 vi=0
v_pk_add_f16 v[vgprValuC+252], v251, v[vgprValuC+252] // sum*alpha + C*beta
_buffer_store_b16 v252, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (28,3,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (29,3,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (30,3,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1); (31,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v138, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
_buffer_load_d16_b16 v140, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
_buffer_load_d16_b16 v142, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(28,0,3,0) */
_buffer_load_d16_b16 v144, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v146, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
_buffer_load_d16_b16 v148, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
_buffer_load_d16_b16 v150, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(29,0,3,0) */
_buffer_load_d16_b16 v152, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v154, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
_buffer_load_d16_b16 v156, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
_buffer_load_d16_b16 v158, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(30,0,3,0) */
_buffer_load_d16_b16 v160, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s32, s[sgprStrideC1J], 4                 // scale StrideC *= numRows(2) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_d16_b16 v162, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
_buffer_load_d16_b16 v164, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
_buffer_load_d16_b16 v166, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(31,0,3,0) */
_buffer_load_d16_b16 v168, v137, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc

/* rC *= alpha batchElements=[(28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (28, 3, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (29, 3, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (30, 3, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0), (31, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 - 1
v_pk_mul_f16 v138, s[sgprBeta], v138               // v138 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+139], v138, v[vgprValuC+139] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v139, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 - 1
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 - 1
v_pk_mul_f16 v142, s[sgprBeta], v142               // v142 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+143], v142, v[vgprValuC+143] // sum*alpha + C*beta
_buffer_store_b16 v143, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 - 1
v_pk_mul_f16 v144, s[sgprBeta], v144               // v144 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+145], v144, v[vgprValuC+145] // sum*alpha + C*beta
_buffer_store_b16 v145, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 - 1
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v147, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 - 1
v_pk_mul_f16 v148, s[sgprBeta], v148               // v148 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+149], v148, v[vgprValuC+149] // sum*alpha + C*beta
_buffer_store_b16 v149, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 - 1
v_pk_mul_f16 v150, s[sgprBeta], v150               // v150 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+151], v150, v[vgprValuC+151] // sum*alpha + C*beta
_buffer_store_b16 v151, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 - 1
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 - 1
v_pk_mul_f16 v154, s[sgprBeta], v154               // v154 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+155], v154, v[vgprValuC+155] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v155, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 - 1
v_pk_mul_f16 v156, s[sgprBeta], v156               // v156 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+157], v156, v[vgprValuC+157] // sum*alpha + C*beta
_buffer_store_b16 v157, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 - 1
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 - 1
v_pk_mul_f16 v160, s[sgprBeta], v160               // v160 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+161], v160, v[vgprValuC+161] // sum*alpha + C*beta
_buffer_store_b16 v161, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 - 1
v_pk_mul_f16 v162, s[sgprBeta], v162               // v162 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+163], v162, v[vgprValuC+163] // sum*alpha + C*beta
s_mul_i32 s32, s[sgprStrideD1J], 4                 // scale StrideD *= numRows(2) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s32       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b16 v163, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 - 1
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 - 1
v_pk_mul_f16 v166, s[sgprBeta], v166               // v166 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+167], v166, v[vgprValuC+167] // sum*alpha + C*beta
_buffer_store_b16 v167, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 - 1
v_pk_mul_f16 v168, s[sgprBeta], v168               // v168 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+169], v168, v[vgprValuC+169] // sum*alpha + C*beta
_buffer_store_b16 v169, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_114                          // jump to end
GW_B1_E1_113:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=38 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v139, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v140, v139, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v139, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v145, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v146, v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v151, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v152, v151, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v151, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v157, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v158, v157, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v157, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v163, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v164, v163, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v163, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v169, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v170, v169, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v169, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v175, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v176, v175, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v175, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v178, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v179, v178, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v178, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v181, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v182, v181, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v181, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v184, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v185, v184, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v184, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v187, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v188, v187, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v187, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v190, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v191, v190, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v190, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v193, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v194, v193, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v193, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v196, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v197, v196, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v196, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v199, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v200, v199, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v199, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v202, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v203, v202, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v202, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v205, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v206, v205, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v205, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v208, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v209, v208, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v208, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v211, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v212, v211, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v211, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v214, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v215, v214, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v214, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v217, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v221, v217, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v217, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v223, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v224, v223, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v223, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v226, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v227, v226, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v226, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v229, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v230, v229, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v229, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v232, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v233, v232, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v232, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v235, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v236, v235, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v235, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v238, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v239, v238, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v238, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v241, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v242, v241, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v241, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v244, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v245, v244, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v244, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v247, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v248, v247, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v247, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v250, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v251, v250, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v250, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+231], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+237], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+243], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+249], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait C
s_waitcnt_vscnt null, 0                            // writes

/* apply mask, calc new C and issue writes */
v_pk_mul_f16 v137, s[sgprBeta], v137               // v137 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+138], v137, v[vgprValuC+138] // sum*alpha + C*beta
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v143, s[sgprBeta], v143               // v143 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+144], v143, v[vgprValuC+144] // sum*alpha + C*beta
_buffer_store_b16 v144, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
_buffer_store_b16 v147, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v149, s[sgprBeta], v149               // v149 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+150], v149, v[vgprValuC+150] // sum*alpha + C*beta
_buffer_store_b16 v150, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v155, s[sgprBeta], v155               // v155 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+156], v155, v[vgprValuC+156] // sum*alpha + C*beta
_buffer_store_b16 v156, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v157, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v161, s[sgprBeta], v161               // v161 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+162], v161, v[vgprValuC+162] // sum*alpha + C*beta
_buffer_store_b16 v162, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v163, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v167, s[sgprBeta], v167               // v167 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+168], v167, v[vgprValuC+168] // sum*alpha + C*beta
_buffer_store_b16 v168, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
_buffer_store_b16 v171, v169, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v173, s[sgprBeta], v173               // v173 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+174], v173, v[vgprValuC+174] // sum*alpha + C*beta
_buffer_store_b16 v174, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v175, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v179, s[sgprBeta], v179               // v179 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+180], v179, v[vgprValuC+180] // sum*alpha + C*beta
_buffer_store_b16 v180, v178, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v182, s[sgprBeta], v182               // v182 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+183], v182, v[vgprValuC+183] // sum*alpha + C*beta
_buffer_store_b16 v183, v181, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v185, s[sgprBeta], v185               // v185 = C*beta ei=16 vi=0
v_pk_add_f16 v[vgprValuC+186], v185, v[vgprValuC+186] // sum*alpha + C*beta
_buffer_store_b16 v186, v184, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v188, s[sgprBeta], v188               // v188 = C*beta ei=17 vi=0
v_pk_add_f16 v[vgprValuC+189], v188, v[vgprValuC+189] // sum*alpha + C*beta
_buffer_store_b16 v189, v187, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v191, s[sgprBeta], v191               // v191 = C*beta ei=18 vi=0
v_pk_add_f16 v[vgprValuC+192], v191, v[vgprValuC+192] // sum*alpha + C*beta
_buffer_store_b16 v192, v190, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v194, s[sgprBeta], v194               // v194 = C*beta ei=19 vi=0
v_pk_add_f16 v[vgprValuC+195], v194, v[vgprValuC+195] // sum*alpha + C*beta
_buffer_store_b16 v195, v193, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v197, s[sgprBeta], v197               // v197 = C*beta ei=20 vi=0
v_pk_add_f16 v[vgprValuC+198], v197, v[vgprValuC+198] // sum*alpha + C*beta
_buffer_store_b16 v198, v196, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v200, s[sgprBeta], v200               // v200 = C*beta ei=21 vi=0
v_pk_add_f16 v[vgprValuC+201], v200, v[vgprValuC+201] // sum*alpha + C*beta
_buffer_store_b16 v201, v199, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v203, s[sgprBeta], v203               // v203 = C*beta ei=22 vi=0
v_pk_add_f16 v[vgprValuC+204], v203, v[vgprValuC+204] // sum*alpha + C*beta
_buffer_store_b16 v204, v202, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v206, s[sgprBeta], v206               // v206 = C*beta ei=23 vi=0
v_pk_add_f16 v[vgprValuC+207], v206, v[vgprValuC+207] // sum*alpha + C*beta
_buffer_store_b16 v207, v205, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v209, s[sgprBeta], v209               // v209 = C*beta ei=24 vi=0
v_pk_add_f16 v[vgprValuC+210], v209, v[vgprValuC+210] // sum*alpha + C*beta
_buffer_store_b16 v210, v208, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v212, s[sgprBeta], v212               // v212 = C*beta ei=25 vi=0
v_pk_add_f16 v[vgprValuC+213], v212, v[vgprValuC+213] // sum*alpha + C*beta
_buffer_store_b16 v213, v211, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v215, s[sgprBeta], v215               // v215 = C*beta ei=26 vi=0
v_pk_add_f16 v[vgprValuC+216], v215, v[vgprValuC+216] // sum*alpha + C*beta
_buffer_store_b16 v216, v214, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v221, s[sgprBeta], v221               // v221 = C*beta ei=27 vi=0
v_pk_add_f16 v[vgprValuC+222], v221, v[vgprValuC+222] // sum*alpha + C*beta
_buffer_store_b16 v222, v217, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v224, s[sgprBeta], v224               // v224 = C*beta ei=28 vi=0
v_pk_add_f16 v[vgprValuC+225], v224, v[vgprValuC+225] // sum*alpha + C*beta
_buffer_store_b16 v225, v223, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v227, s[sgprBeta], v227               // v227 = C*beta ei=29 vi=0
v_pk_add_f16 v[vgprValuC+228], v227, v[vgprValuC+228] // sum*alpha + C*beta
_buffer_store_b16 v228, v226, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v230, s[sgprBeta], v230               // v230 = C*beta ei=30 vi=0
v_pk_add_f16 v[vgprValuC+231], v230, v[vgprValuC+231] // sum*alpha + C*beta
_buffer_store_b16 v231, v229, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v233, s[sgprBeta], v233               // v233 = C*beta ei=31 vi=0
v_pk_add_f16 v[vgprValuC+234], v233, v[vgprValuC+234] // sum*alpha + C*beta
_buffer_store_b16 v234, v232, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v236, s[sgprBeta], v236               // v236 = C*beta ei=32 vi=0
v_pk_add_f16 v[vgprValuC+237], v236, v[vgprValuC+237] // sum*alpha + C*beta
_buffer_store_b16 v237, v235, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v239, s[sgprBeta], v239               // v239 = C*beta ei=33 vi=0
v_pk_add_f16 v[vgprValuC+240], v239, v[vgprValuC+240] // sum*alpha + C*beta
_buffer_store_b16 v240, v238, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v242, s[sgprBeta], v242               // v242 = C*beta ei=34 vi=0
v_pk_add_f16 v[vgprValuC+243], v242, v[vgprValuC+243] // sum*alpha + C*beta
_buffer_store_b16 v243, v241, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v245, s[sgprBeta], v245               // v245 = C*beta ei=35 vi=0
v_pk_add_f16 v[vgprValuC+246], v245, v[vgprValuC+246] // sum*alpha + C*beta
_buffer_store_b16 v246, v244, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v248, s[sgprBeta], v248               // v248 = C*beta ei=36 vi=0
v_pk_add_f16 v[vgprValuC+249], v248, v[vgprValuC+249] // sum*alpha + C*beta
_buffer_store_b16 v249, v247, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v251, s[sgprBeta], v251               // v251 = C*beta ei=37 vi=0
v_pk_add_f16 v[vgprValuC+252], v251, v[vgprValuC+252] // sum*alpha + C*beta
_buffer_store_b16 v252, v250, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (16,3,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (17,3,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (18,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v139, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v140, v139, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v139, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v145, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v146, v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v151, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v152, v151, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v151, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v157, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v158, v157, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v157, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v163, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v164, v163, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v163, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v169, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v170, v169, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v169, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v175, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v176, v175, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v175, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v178, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v179, v178, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v178, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v181, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v182, v181, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v181, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v184, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v185, v184, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v184, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v187, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v188, v187, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v187, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v190, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v191, v190, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v190, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v193, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v194, v193, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v193, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v196, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v197, v196, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v196, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v199, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v200, v199, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v199, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v202, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v203, v202, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v202, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v205, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v206, v205, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v205, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v208, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v209, v208, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v208, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v211, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v212, v211, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v211, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v214, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v215, v214, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v214, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v217, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v221, v217, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v217, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v223, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v224, v223, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v223, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v226, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v227, v226, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v226, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v229, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v230, v229, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v229, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v232, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v233, v232, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v232, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v235, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v236, v235, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v235, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v238, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v239, v238, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v238, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v241, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v242, v241, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v241, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v244, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v245, v244, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v244, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v247, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v248, v247, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v247, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v250, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v251, v250, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v250, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (16, 3, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (17, 3, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (18, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+231], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+237], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+243], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+249], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait C
s_waitcnt_vscnt null, 0                            // writes

/* apply mask, calc new C and issue writes */
v_pk_mul_f16 v137, s[sgprBeta], v137               // v137 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+138], v137, v[vgprValuC+138] // sum*alpha + C*beta
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v143, s[sgprBeta], v143               // v143 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+144], v143, v[vgprValuC+144] // sum*alpha + C*beta
_buffer_store_b16 v144, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
_buffer_store_b16 v147, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v149, s[sgprBeta], v149               // v149 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+150], v149, v[vgprValuC+150] // sum*alpha + C*beta
_buffer_store_b16 v150, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v155, s[sgprBeta], v155               // v155 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+156], v155, v[vgprValuC+156] // sum*alpha + C*beta
_buffer_store_b16 v156, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v157, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v161, s[sgprBeta], v161               // v161 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+162], v161, v[vgprValuC+162] // sum*alpha + C*beta
_buffer_store_b16 v162, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v163, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v167, s[sgprBeta], v167               // v167 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+168], v167, v[vgprValuC+168] // sum*alpha + C*beta
_buffer_store_b16 v168, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
_buffer_store_b16 v171, v169, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v173, s[sgprBeta], v173               // v173 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+174], v173, v[vgprValuC+174] // sum*alpha + C*beta
_buffer_store_b16 v174, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v175, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v179, s[sgprBeta], v179               // v179 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+180], v179, v[vgprValuC+180] // sum*alpha + C*beta
_buffer_store_b16 v180, v178, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v182, s[sgprBeta], v182               // v182 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+183], v182, v[vgprValuC+183] // sum*alpha + C*beta
_buffer_store_b16 v183, v181, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v185, s[sgprBeta], v185               // v185 = C*beta ei=16 vi=0
v_pk_add_f16 v[vgprValuC+186], v185, v[vgprValuC+186] // sum*alpha + C*beta
_buffer_store_b16 v186, v184, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v188, s[sgprBeta], v188               // v188 = C*beta ei=17 vi=0
v_pk_add_f16 v[vgprValuC+189], v188, v[vgprValuC+189] // sum*alpha + C*beta
_buffer_store_b16 v189, v187, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v191, s[sgprBeta], v191               // v191 = C*beta ei=18 vi=0
v_pk_add_f16 v[vgprValuC+192], v191, v[vgprValuC+192] // sum*alpha + C*beta
_buffer_store_b16 v192, v190, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v194, s[sgprBeta], v194               // v194 = C*beta ei=19 vi=0
v_pk_add_f16 v[vgprValuC+195], v194, v[vgprValuC+195] // sum*alpha + C*beta
_buffer_store_b16 v195, v193, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v197, s[sgprBeta], v197               // v197 = C*beta ei=20 vi=0
v_pk_add_f16 v[vgprValuC+198], v197, v[vgprValuC+198] // sum*alpha + C*beta
_buffer_store_b16 v198, v196, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v200, s[sgprBeta], v200               // v200 = C*beta ei=21 vi=0
v_pk_add_f16 v[vgprValuC+201], v200, v[vgprValuC+201] // sum*alpha + C*beta
_buffer_store_b16 v201, v199, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v203, s[sgprBeta], v203               // v203 = C*beta ei=22 vi=0
v_pk_add_f16 v[vgprValuC+204], v203, v[vgprValuC+204] // sum*alpha + C*beta
_buffer_store_b16 v204, v202, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v206, s[sgprBeta], v206               // v206 = C*beta ei=23 vi=0
v_pk_add_f16 v[vgprValuC+207], v206, v[vgprValuC+207] // sum*alpha + C*beta
_buffer_store_b16 v207, v205, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v209, s[sgprBeta], v209               // v209 = C*beta ei=24 vi=0
v_pk_add_f16 v[vgprValuC+210], v209, v[vgprValuC+210] // sum*alpha + C*beta
_buffer_store_b16 v210, v208, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v212, s[sgprBeta], v212               // v212 = C*beta ei=25 vi=0
v_pk_add_f16 v[vgprValuC+213], v212, v[vgprValuC+213] // sum*alpha + C*beta
_buffer_store_b16 v213, v211, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v215, s[sgprBeta], v215               // v215 = C*beta ei=26 vi=0
v_pk_add_f16 v[vgprValuC+216], v215, v[vgprValuC+216] // sum*alpha + C*beta
_buffer_store_b16 v216, v214, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v221, s[sgprBeta], v221               // v221 = C*beta ei=27 vi=0
v_pk_add_f16 v[vgprValuC+222], v221, v[vgprValuC+222] // sum*alpha + C*beta
_buffer_store_b16 v222, v217, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v224, s[sgprBeta], v224               // v224 = C*beta ei=28 vi=0
v_pk_add_f16 v[vgprValuC+225], v224, v[vgprValuC+225] // sum*alpha + C*beta
_buffer_store_b16 v225, v223, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v227, s[sgprBeta], v227               // v227 = C*beta ei=29 vi=0
v_pk_add_f16 v[vgprValuC+228], v227, v[vgprValuC+228] // sum*alpha + C*beta
_buffer_store_b16 v228, v226, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v230, s[sgprBeta], v230               // v230 = C*beta ei=30 vi=0
v_pk_add_f16 v[vgprValuC+231], v230, v[vgprValuC+231] // sum*alpha + C*beta
_buffer_store_b16 v231, v229, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v233, s[sgprBeta], v233               // v233 = C*beta ei=31 vi=0
v_pk_add_f16 v[vgprValuC+234], v233, v[vgprValuC+234] // sum*alpha + C*beta
_buffer_store_b16 v234, v232, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v236, s[sgprBeta], v236               // v236 = C*beta ei=32 vi=0
v_pk_add_f16 v[vgprValuC+237], v236, v[vgprValuC+237] // sum*alpha + C*beta
_buffer_store_b16 v237, v235, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v239, s[sgprBeta], v239               // v239 = C*beta ei=33 vi=0
v_pk_add_f16 v[vgprValuC+240], v239, v[vgprValuC+240] // sum*alpha + C*beta
_buffer_store_b16 v240, v238, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v242, s[sgprBeta], v242               // v242 = C*beta ei=34 vi=0
v_pk_add_f16 v[vgprValuC+243], v242, v[vgprValuC+243] // sum*alpha + C*beta
_buffer_store_b16 v243, v241, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v245, s[sgprBeta], v245               // v245 = C*beta ei=35 vi=0
v_pk_add_f16 v[vgprValuC+246], v245, v[vgprValuC+246] // sum*alpha + C*beta
_buffer_store_b16 v246, v244, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v248, s[sgprBeta], v248               // v248 = C*beta ei=36 vi=0
v_pk_add_f16 v[vgprValuC+249], v248, v[vgprValuC+249] // sum*alpha + C*beta
_buffer_store_b16 v249, v247, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v251, s[sgprBeta], v251               // v251 = C*beta ei=37 vi=0
v_pk_add_f16 v[vgprValuC+252], v251, v[vgprValuC+252] // sum*alpha + C*beta
_buffer_store_b16 v252, v250, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (19,3,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (20,3,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (21,3,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (22,3,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (23,3,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (24,3,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (25,3,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (26,3,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (27,3,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v139, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v140, v139, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v139, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v145, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v146, v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v151, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v152, v151, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v151, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v157, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v158, v157, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v157, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v163, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v164, v163, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v163, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v169, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v170, v169, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v169, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v175, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v176, v175, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v175, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v178, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v179, v178, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v178, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, -1, v178, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v181, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v182, v181, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v181, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, -1, v181, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v184, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v185, v184, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v184, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, -1, v184, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v187, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v188, v187, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v187, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v187, -1, v187, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v190, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v191, v190, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v190, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, -1, v190, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v193, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v194, v193, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v193, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v193, -1, v193, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 18               // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 18                // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 18                // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v196, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v197, v196, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v196, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, -1, v196, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v199, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v200, v199, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v199, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, -1, v199, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v202, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v203, v202, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v202, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, -1, v202, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v205, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v206, v205, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v205, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, -1, v205, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v208, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v209, v208, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v208, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, -1, v208, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v211, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v212, v211, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v211, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, -1, v211, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v214, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v215, v214, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v214, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, -1, v214, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v217, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v221, v217, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v217, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, -1, v217, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v223, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v224, v223, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v223, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, -1, v223, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v226, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v227, v226, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v226, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, -1, v226, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v229, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v230, v229, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v229, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, -1, v229, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v232, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v233, v232, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v232, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, -1, v232, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v235, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v236, v235, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v235, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, -1, v235, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v238, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v239, v238, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v238, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, -1, v238, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v241, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v242, v241, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v241, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, -1, v241, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v244, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v245, v244, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v244, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, -1, v244, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v247, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v248, v247, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v247, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, -1, v247, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v250, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v251, v250, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v250, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, -1, v250, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (19, 3, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (20, 3, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (21, 3, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (22, 3, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (23, 3, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (24, 3, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (25, 3, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (26, 3, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (27, 3, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+231], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+237], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+243], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+249], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait C
s_waitcnt_vscnt null, 0                            // writes

/* apply mask, calc new C and issue writes */
v_pk_mul_f16 v137, s[sgprBeta], v137               // v137 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+138], v137, v[vgprValuC+138] // sum*alpha + C*beta
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v143, s[sgprBeta], v143               // v143 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+144], v143, v[vgprValuC+144] // sum*alpha + C*beta
_buffer_store_b16 v144, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
_buffer_store_b16 v147, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v149, s[sgprBeta], v149               // v149 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+150], v149, v[vgprValuC+150] // sum*alpha + C*beta
_buffer_store_b16 v150, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v155, s[sgprBeta], v155               // v155 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+156], v155, v[vgprValuC+156] // sum*alpha + C*beta
_buffer_store_b16 v156, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v157, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v161, s[sgprBeta], v161               // v161 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+162], v161, v[vgprValuC+162] // sum*alpha + C*beta
_buffer_store_b16 v162, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v163, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v167, s[sgprBeta], v167               // v167 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+168], v167, v[vgprValuC+168] // sum*alpha + C*beta
_buffer_store_b16 v168, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
_buffer_store_b16 v171, v169, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v173, s[sgprBeta], v173               // v173 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+174], v173, v[vgprValuC+174] // sum*alpha + C*beta
_buffer_store_b16 v174, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v175, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v179, s[sgprBeta], v179               // v179 = C*beta ei=14 vi=0
v_pk_add_f16 v[vgprValuC+180], v179, v[vgprValuC+180] // sum*alpha + C*beta
_buffer_store_b16 v180, v178, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v182, s[sgprBeta], v182               // v182 = C*beta ei=15 vi=0
v_pk_add_f16 v[vgprValuC+183], v182, v[vgprValuC+183] // sum*alpha + C*beta
_buffer_store_b16 v183, v181, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v185, s[sgprBeta], v185               // v185 = C*beta ei=16 vi=0
v_pk_add_f16 v[vgprValuC+186], v185, v[vgprValuC+186] // sum*alpha + C*beta
_buffer_store_b16 v186, v184, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v188, s[sgprBeta], v188               // v188 = C*beta ei=17 vi=0
v_pk_add_f16 v[vgprValuC+189], v188, v[vgprValuC+189] // sum*alpha + C*beta
_buffer_store_b16 v189, v187, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v191, s[sgprBeta], v191               // v191 = C*beta ei=18 vi=0
v_pk_add_f16 v[vgprValuC+192], v191, v[vgprValuC+192] // sum*alpha + C*beta
_buffer_store_b16 v192, v190, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v194, s[sgprBeta], v194               // v194 = C*beta ei=19 vi=0
v_pk_add_f16 v[vgprValuC+195], v194, v[vgprValuC+195] // sum*alpha + C*beta
_buffer_store_b16 v195, v193, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v197, s[sgprBeta], v197               // v197 = C*beta ei=20 vi=0
v_pk_add_f16 v[vgprValuC+198], v197, v[vgprValuC+198] // sum*alpha + C*beta
_buffer_store_b16 v198, v196, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v200, s[sgprBeta], v200               // v200 = C*beta ei=21 vi=0
v_pk_add_f16 v[vgprValuC+201], v200, v[vgprValuC+201] // sum*alpha + C*beta
_buffer_store_b16 v201, v199, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v203, s[sgprBeta], v203               // v203 = C*beta ei=22 vi=0
v_pk_add_f16 v[vgprValuC+204], v203, v[vgprValuC+204] // sum*alpha + C*beta
_buffer_store_b16 v204, v202, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v206, s[sgprBeta], v206               // v206 = C*beta ei=23 vi=0
v_pk_add_f16 v[vgprValuC+207], v206, v[vgprValuC+207] // sum*alpha + C*beta
_buffer_store_b16 v207, v205, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v209, s[sgprBeta], v209               // v209 = C*beta ei=24 vi=0
v_pk_add_f16 v[vgprValuC+210], v209, v[vgprValuC+210] // sum*alpha + C*beta
_buffer_store_b16 v210, v208, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v212, s[sgprBeta], v212               // v212 = C*beta ei=25 vi=0
v_pk_add_f16 v[vgprValuC+213], v212, v[vgprValuC+213] // sum*alpha + C*beta
_buffer_store_b16 v213, v211, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v215, s[sgprBeta], v215               // v215 = C*beta ei=26 vi=0
v_pk_add_f16 v[vgprValuC+216], v215, v[vgprValuC+216] // sum*alpha + C*beta
_buffer_store_b16 v216, v214, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v221, s[sgprBeta], v221               // v221 = C*beta ei=27 vi=0
v_pk_add_f16 v[vgprValuC+222], v221, v[vgprValuC+222] // sum*alpha + C*beta
_buffer_store_b16 v222, v217, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v224, s[sgprBeta], v224               // v224 = C*beta ei=28 vi=0
v_pk_add_f16 v[vgprValuC+225], v224, v[vgprValuC+225] // sum*alpha + C*beta
_buffer_store_b16 v225, v223, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v227, s[sgprBeta], v227               // v227 = C*beta ei=29 vi=0
v_pk_add_f16 v[vgprValuC+228], v227, v[vgprValuC+228] // sum*alpha + C*beta
_buffer_store_b16 v228, v226, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v230, s[sgprBeta], v230               // v230 = C*beta ei=30 vi=0
v_pk_add_f16 v[vgprValuC+231], v230, v[vgprValuC+231] // sum*alpha + C*beta
_buffer_store_b16 v231, v229, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v233, s[sgprBeta], v233               // v233 = C*beta ei=31 vi=0
v_pk_add_f16 v[vgprValuC+234], v233, v[vgprValuC+234] // sum*alpha + C*beta
_buffer_store_b16 v234, v232, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v236, s[sgprBeta], v236               // v236 = C*beta ei=32 vi=0
v_pk_add_f16 v[vgprValuC+237], v236, v[vgprValuC+237] // sum*alpha + C*beta
_buffer_store_b16 v237, v235, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v239, s[sgprBeta], v239               // v239 = C*beta ei=33 vi=0
v_pk_add_f16 v[vgprValuC+240], v239, v[vgprValuC+240] // sum*alpha + C*beta
_buffer_store_b16 v240, v238, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v242, s[sgprBeta], v242               // v242 = C*beta ei=34 vi=0
v_pk_add_f16 v[vgprValuC+243], v242, v[vgprValuC+243] // sum*alpha + C*beta
_buffer_store_b16 v243, v241, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v245, s[sgprBeta], v245               // v245 = C*beta ei=35 vi=0
v_pk_add_f16 v[vgprValuC+246], v245, v[vgprValuC+246] // sum*alpha + C*beta
_buffer_store_b16 v246, v244, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v248, s[sgprBeta], v248               // v248 = C*beta ei=36 vi=0
v_pk_add_f16 v[vgprValuC+249], v248, v[vgprValuC+249] // sum*alpha + C*beta
_buffer_store_b16 v249, v247, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v251, s[sgprBeta], v251               // v251 = C*beta ei=37 vi=0
v_pk_add_f16 v[vgprValuC+252], v251, v[vgprValuC+252] // sum*alpha + C*beta
_buffer_store_b16 v252, v250, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (28,2,0,0:vw1); (28,3,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (29,3,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (30,3,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1); (31,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v136, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v139, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v140, v139, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v139, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v142, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v145, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v146, v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v148, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v151, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v152, v151, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v151, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v154, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v157, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v158, v157, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v157, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v160, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v163, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v164, v163, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v163, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
_v_add_co_u32 v131, vcc_lo, v131, 2                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s32, s[sgprStrideC1J], 2                 // scale stride
_v_add_u32 v132, v132, s32                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s32, s[sgprStrideD1J], 2                 // scale stride
_v_add_u32 v133, v133, s32                         // Move coutRowPtr to next row
v_cmp_lt_u32 s32, v130, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v166, v132, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v133, v130, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
_v_add_co_u32 v134, vcc_lo, v130, 32               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v169, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v170, v169, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v169, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v169, -1, v169, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
_v_add_co_u32 v134, vcc_lo, v130, 64               // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v172, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v172, -1, v172, s34                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,3,0) */
s_mov_b32 s32, 96                                  // coordOffset0 d0=3 vc0=0
_v_add_co_u32 v134, vcc_lo, v130, s32              // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s32, v134, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s34, v131, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s34, s32, s34                            // in0 && in1
_v_add_lshl_u32 v175, v132, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDC clip if OOB. offset
_buffer_load_d16_b16 v176, v175, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v175, v133, v134, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v175, -1, v175, s34                  // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(28, 2, 0, 0), (28, 3, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (29, 3, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (30, 3, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0), (31, 3, 0, 0)] */
v_pk_mul_f16 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_pk_mul_f16 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait C
s_waitcnt_vscnt null, 0                            // writes

/* apply mask, calc new C and issue writes */
v_pk_mul_f16 v137, s[sgprBeta], v137               // v137 = C*beta ei=0 vi=0
v_pk_add_f16 v[vgprValuC+138], v137, v[vgprValuC+138] // sum*alpha + C*beta
_buffer_store_b16 v138, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v140, s[sgprBeta], v140               // v140 = C*beta ei=1 vi=0
v_pk_add_f16 v[vgprValuC+141], v140, v[vgprValuC+141] // sum*alpha + C*beta
_buffer_store_b16 v141, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v143, s[sgprBeta], v143               // v143 = C*beta ei=2 vi=0
v_pk_add_f16 v[vgprValuC+144], v143, v[vgprValuC+144] // sum*alpha + C*beta
_buffer_store_b16 v144, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v146, s[sgprBeta], v146               // v146 = C*beta ei=3 vi=0
v_pk_add_f16 v[vgprValuC+147], v146, v[vgprValuC+147] // sum*alpha + C*beta
_buffer_store_b16 v147, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v149, s[sgprBeta], v149               // v149 = C*beta ei=4 vi=0
v_pk_add_f16 v[vgprValuC+150], v149, v[vgprValuC+150] // sum*alpha + C*beta
_buffer_store_b16 v150, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v152, s[sgprBeta], v152               // v152 = C*beta ei=5 vi=0
v_pk_add_f16 v[vgprValuC+153], v152, v[vgprValuC+153] // sum*alpha + C*beta
_buffer_store_b16 v153, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v155, s[sgprBeta], v155               // v155 = C*beta ei=6 vi=0
v_pk_add_f16 v[vgprValuC+156], v155, v[vgprValuC+156] // sum*alpha + C*beta
_buffer_store_b16 v156, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v158, s[sgprBeta], v158               // v158 = C*beta ei=7 vi=0
v_pk_add_f16 v[vgprValuC+159], v158, v[vgprValuC+159] // sum*alpha + C*beta
_buffer_store_b16 v159, v157, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v161, s[sgprBeta], v161               // v161 = C*beta ei=8 vi=0
v_pk_add_f16 v[vgprValuC+162], v161, v[vgprValuC+162] // sum*alpha + C*beta
_buffer_store_b16 v162, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v164, s[sgprBeta], v164               // v164 = C*beta ei=9 vi=0
v_pk_add_f16 v[vgprValuC+165], v164, v[vgprValuC+165] // sum*alpha + C*beta
_buffer_store_b16 v165, v163, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v167, s[sgprBeta], v167               // v167 = C*beta ei=10 vi=0
v_pk_add_f16 v[vgprValuC+168], v167, v[vgprValuC+168] // sum*alpha + C*beta
_buffer_store_b16 v168, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v170, s[sgprBeta], v170               // v170 = C*beta ei=11 vi=0
v_pk_add_f16 v[vgprValuC+171], v170, v[vgprValuC+171] // sum*alpha + C*beta
_buffer_store_b16 v171, v169, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v173, s[sgprBeta], v173               // v173 = C*beta ei=12 vi=0
v_pk_add_f16 v[vgprValuC+174], v173, v[vgprValuC+174] // sum*alpha + C*beta
_buffer_store_b16 v174, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_pk_mul_f16 v176, s[sgprBeta], v176               // v176 = C*beta ei=13 vi=0
v_pk_add_f16 v[vgprValuC+177], v176, v[vgprValuC+177] // sum*alpha + C*beta
_buffer_store_b16 v177, v175, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_114                          // jump to end
label_GW_End_114:

label_0119:  /// KernelEnd
s_endpgm                                           // Kernel End


