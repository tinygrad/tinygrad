## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`extry`|`y[i] = x[i]`|None|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `9`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|28|36|Ignored|
|27|1|Must be `1`|Otherwise decodes as [`extrv`](extr_v.md)|
|26|1|Must be `0`|Otherwise decodes as [`extrv`](extr_v.md)|
|23|3|Ignored|
|20|3|X register index|
|9|11|Ignored|
|6|3|Y register index|
|0|6|Ignored|

## Description

Copies an entire register from X to Y.

## Emulation code

See [extr.c](extr.c).

A representative sample is:
```c
void emulate_AMX_EXTRY(amx_state* state, uint64_t operand) {
    if (operand & EXTR_HV) {
        ...
    } else if (operand & EXTR_BETWEEN_XY) {
        memcpy(state->y + ((operand >>  6) & 7),
               state->x + ((operand >> 20) & 7), 64);
    } else {
        ...
    }
}
```
