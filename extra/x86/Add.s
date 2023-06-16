.section .text 
.globl _add
_add:
  enter $8, $0
  mov $0, -8(%rsp)

.L0:
    mov -8(%rsp), %r8

    movd 0(%rsi, %r8, 4), %xmm0
    addss 0(%rdx, %r8, 4), %xmm0
    movd %xmm0, 0(%rdi, %r8, 4)

    mov -8(%rsp), %r15
    inc %r15
    mov %r15, -8(%rsp)
    cmp $3, %r15
    jle .L0

    mov %r15, %rax
    leave
    ret
