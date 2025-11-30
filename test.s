.text
.globl kernel
.p2align 8 // TODO: need more?
.type kernel,@function

kernel:
  s_load_b64 s[0:1], s[0:1], 0x0
  v_mov_b32_e32 v0 0
  v_mov_b32_e32 v1 6.0
  s_wait_kmcnt 0
  global_store_b32 v0, v1, s[0:1]
  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
  s_endpgm

.rodata
.amdhsa_kernel kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
