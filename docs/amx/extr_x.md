## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`extrx`|`x[i] = y[i]`|None|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `8`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|28|36|Ignored|
|27|1|Must be `1`|Otherwise decodes as [`extrh`](extr_h.md)|
|26|1|Must be `0`|Otherwise decodes as [`extrh`](extr_h.md)|
|23|3|Ignored|
|20|3|Y register index|
|19|1|Ignored|
|16|3|X register index|
|0|16|Ignored|

## Description

Copies an entire register from Y to X.

## Emulation code

See [extr.c](extr.c).

A representative sample is:
```c
void emulate_AMX_EXTRX(amx_state* state, uint64_t operand) {
    if (operand & EXTR_HV) {
        ...
    } else if (operand & EXTR_BETWEEN_XY) {
        memcpy(state->x + ((operand >> 16) & 7),
               state->y + ((operand >> 20) & 7), 64);
    } else {
        ...
    }
}
```
