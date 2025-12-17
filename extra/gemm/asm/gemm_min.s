.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.section	.text.
.global	gemm
.p2align	8
.type	gemm,@function

gemm:
  s_nop 0
  s_endpgm

    .section	.rodata,"a",@progbits
    .p2align	6, 0x0
    .amdhsa_kernel gemm
        .amdhsa_group_segment_fixed_size 65536
        .amdhsa_private_segment_fixed_size 0
        .amdhsa_kernarg_size 24
        .amdhsa_user_sgpr_count 2
        .amdhsa_user_sgpr_dispatch_ptr 0
        .amdhsa_user_sgpr_queue_ptr 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_user_sgpr_dispatch_id 0
        .amdhsa_user_sgpr_kernarg_preload_length 0
        .amdhsa_user_sgpr_kernarg_preload_offset 0
        .amdhsa_user_sgpr_private_segment_size 0
        .amdhsa_uses_dynamic_stack 0
        .amdhsa_enable_private_segment 0
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 0
        .amdhsa_system_sgpr_workgroup_info 0
        .amdhsa_system_vgpr_workitem_id 0
        .amdhsa_next_free_vgpr 484
        .amdhsa_next_free_sgpr 24
        .amdhsa_accum_offset 228
        .amdhsa_reserve_vcc 1
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 3
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_dx10_clamp 1
        .amdhsa_ieee_mode 1
        .amdhsa_fp16_overflow 0
        .amdhsa_tg_split 0
        .amdhsa_exception_fp_ieee_invalid_op 0
        .amdhsa_exception_fp_denorm_src 0
        .amdhsa_exception_fp_ieee_div_zero 0
        .amdhsa_exception_fp_ieee_overflow 0
        .amdhsa_exception_fp_ieee_underflow 0
        .amdhsa_exception_fp_ieee_inexact 0
        .amdhsa_exception_int_div_zero 0
    .end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           A.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .name:           B.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .name:           C.coerce
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm
    .private_segment_fixed_size: 0
    .sgpr_count:     23
    .sgpr_spill_count: 0
    .symbol:         gemm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     484
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
