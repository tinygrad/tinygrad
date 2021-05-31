restart:
li s0, 3 
lui s1, 0x2000
/*li s3, 0x100*/

addr_c:
sll s0,s0,1
and s0,s0,0xFF
beq s0,zero,restart

/*addi s0,s0,0x1
blt s0,s3,addr_18
li s0, 2 

addr_18:
li s2, 0x2

addr_1c:
bge s2, s0, addr_38
mv a0, s0
mv a1, s2
jal ra, mod_or_something
beq a0, zero, addr_40
addi s2, s2, 0x1
j addr_1c*/

addr_38:
sw s0, 0x0(s1)
jal ra, sleep_fixed

addr_40:
j addr_c


/*mod_or_something:
li t0,0x1

addr_48:
sub a0,a0,a1
bge a0,t0,addr_48

ret*/

sleep_fixed:
lui t0,0x100
addi t0,t0,-0x5e0
addr_5c:
addi t0,t0,-0x1
bne t0,zero,addr_5c
ret

