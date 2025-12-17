.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.section	.text.
.global	gemm
.p2align	8
.type	gemm,@function

gemm:
  s_nop 0
  s_endpgm
