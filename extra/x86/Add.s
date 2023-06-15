.section .text 
.globl _add
_add:
  enter $8, $0
  mov $0, -8(%rsp)

.L0:
    mov -8(%rsp), %r8

    movl 0(%rsi, %r8, 4), %eax
    addl 0(%rdx, %r8, 4), %eax
    movl %eax, 0(%rdi, %r8, 4)

    mov -8(%rsp), %r15
    inc %r15
    mov %r15, -8(%rsp)
    cmp $3, %r15
    jle .L0

    mov %r15, %rax
    leave
    ret
