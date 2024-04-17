The A64 instruction set defines the following instruction bit patterns as reserved:

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|29|3|op0|Must be `0`
|25|4||Must be `0`|
|21|4|op1 (hi)|Must be `1`|
|16|5|op1 (lo)||
|0|16|

AMX instructions fit into this space like so:

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|29|3|op0|Must be `0`
|25|4||Must be `0`|
|21|4|op1 (hi)|Must be `1`|
|16|5|op1 (lo)|Must be `0`|
|12|4||Must be `1`|
|10|2||Must be `0`|
|5|5|Instruction|`0` through `22` used|
|0|5|Operand|Either 5-bit immediate or 5-bit GPR index, depending on instruction. GPR 31 denotes xzr.|

Inline assembly syntax can be used to emit AMX instructions from C code. The easy case is when the operand is a 5-bit immediate:
```c
#define AMX_OP_IMM5(op, imm5) \
    __asm(".word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
```

The harder case is the operand being a 5-bit GPR index. The C compiler will choose the particular GPR however it sees fit, which will be one of `x0` through `x31`, but inline assembly syntax receives this choice as a string rather than an integer. However, _prepending_ the _string_ `0` to the choice gives the _strings_ `0x0` through `0x31`, which just happen to be parsable as hexadecimal numbers. They can be used as such, with some math to convert from base 16 back to base 10:
```c
#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")
```

Instruction `17` is used for setup and clearing, and is the sole instruction which uses a 5-bit immediate; all other instructions use a 5-bit GPR index. In Apple's [Accelerate](https://developer.apple.com/documentation/accelerate?language=objc), instruction `17` is apparently always prefixed by three nops. The same can be done in inline assembly:
```c
#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
```

With these macros, the full AMX instruction set can be accessed:
```c
#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)  AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)
```

 
