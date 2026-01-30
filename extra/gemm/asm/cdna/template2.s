.text
.section	.text.
.global	gemm
.p2align	8
.type	gemm,@function

gemm:
INSTRUCTIONS

.section .rodata,"a",@progbits
.p2align 6, 0x0
.amdhsa_kernel gemm
  .amdhsa_group_segment_fixed_size 133120
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 48
  .amdhsa_next_free_vgpr 512
  .amdhsa_next_free_sgpr 96
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_preload_length 0
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_accum_offset 256
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_tg_split 0
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_ieee_mode 1
  .amdhsa_fp16_overflow 0
  .amdhsa_dx10_clamp 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .name:           Gemm info
        .offset:         0
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           kernel info0
        .offset:         4
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           kernel info1
        .offset:         8
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           numWG
        .offset:         12
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree0
        .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree1
        .offset:         20
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree2
        .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesSum0
        .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .address_space:  generic
        .name:           D
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .value_type:     bf16
      - .address_space:  generic
        .name:           C
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .value_type:     bf16
      - .address_space:  generic
        .name:           A
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .value_type:     bf16
      - .address_space:  generic
        .name:           B
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .value_type:     bf16
      - .address_space:  generic
        .name:           AddressWS
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  generic
        .name:           AddressFlags
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
        .value_type:     bf16
      - .name:           strideD0
        .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideD1
        .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideC0
        .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideC1
        .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideA0
        .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideA1
        .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideB0
        .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideB1
        .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           alpha
        .offset:         112
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           beta
        .offset:         116
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           ItersPerTile
        .offset:         120
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           MagicNumberItersPerTile
        .offset:         124
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           MagicShiftItersPerTile
        .offset:         128
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           TotalIters
        .offset:         132
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SKItersPerWG
        .offset:         136
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           skGrid
        .offset:         140
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           skTiles
        .offset:         144
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .address_space:  generic
        .name:           AddressScaleAlphaVec
        .offset:         148
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  generic
        .name:           bias
        .offset:         156
        .size:           8
        .value_kind:     global_buffer
        .value_type:     void
      - .name:           biasType
        .offset:         164
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           StrideBias
        .offset:         168
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           activationAlpha
        .offset:         172
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           activationBeta
        .offset:         176
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           activationType
        .offset:         180
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
    .group_segment_fixed_size: 133120
    .kernarg_segment_align: 8
    .kernarg_segment_size: 184
    .max_flat_workgroup_size: 256
    .name:           gemm
    .private_segment_fixed_size: 0
    .sgpr_count:     95
    .sgpr_spill_count: 0
    .symbol:         gemm.kd
    .vgpr_count:     249
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 1
...
.end_amdgpu_metadata
