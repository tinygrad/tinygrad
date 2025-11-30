.text
.globl kernel
.p2align 8   // TODO: do you need more?
.type kernel,@function

kernel:
  s_clause 0x1                                               // 000000001600: BF850001
  s_load_b128 s[4:7], s[0:1], 0x0                            // 000000001604: F4004100 F8000000
  s_wait_kmcnt 0x0                                           // 000000001614: BFC70000
  s_load_b32 s0, s[6:7], 0x0                                 // 000000001620: F4000000 F8000000
  s_wait_kmcnt 0x0                                           // 000000001628: BFC70000
  s_mov_b32 s1, 1.0
  s_add_f32 s0, s1, s0                                       // 00000000162C: A0000002
  s_delay_alu instid0(SALU_CYCLE_3)                          // 000000001630: BF87000B
  v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0              // 000000001634: CA100080 00000000
  global_store_b32 v0, v1, s[4:5]                            // 00000000163C: EE068004 00800000 00000000
  s_nop 0                                                    // 000000001648: BF800000
  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000164C: BFB60003
  s_endpgm                                                   // 000000001650: BFB00000
.rodata
.amdhsa_kernel kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
