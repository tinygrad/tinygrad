## Quick summary

|Instruction|General theme|Optional special features|
|---|---|---|
|`ldx`|<code>&nbsp;&nbsp;&nbsp;x[i] = memory[i]</code>|Load pair|
|`ldy`|<code>&nbsp;&nbsp;&nbsp;y[i] = memory[i]</code>|Load pair|
|`ldz`<br/>`ldzi`|`z[_][i] = memory[i]`|Load pair, interleaved Z|
|`stx`|<code>memory[i] =&nbsp;&nbsp;&nbsp;&nbsp;x[i]</code>|Store pair|
|`sty`|<code>memory[i] =&nbsp;&nbsp;&nbsp;&nbsp;y[i]</code>|Store pair|
|`stz`<br/>`stzi`|`memory[i] = z[_][i]`|Store pair, interleaved Z|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|`0` for `ldx`<br/>`1` for `ldy`<br/>`2` for `stx`<br/>`3` for `sty`<br/>`4` for `ldz`<br/>`5` for `stz`<br/>`6` for `ldzi`<br/>`7` for `stzi`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

For `ldx` / `ldy`:

|Bit|Width|Meaning|
|---:|---:|---|
|63|1|Ignored|
|62|1|Load multiple registers (`1`) or single register (`0`)|
|61|1|On M1/M2: Ignored (loads are always to consecutive registers)<br/>On M3: Load to non-consecutive registers (`1`) or to consecutive registers (`0`)|
|60|1|On M1: Ignored ("multiple" always means two registers)<br/>On M2/M3: "Multiple" means four registers (`1`) or two registers (`0`)|
|59|1|Ignored|
|56|3|X / Y register index|
|0|56|Pointer|

For `stx` / `sty`:

|Bit|Width|Meaning|
|---:|---:|---|
|63|1|Ignored|
|62|1|Store pair of registers (`1`) or single register (`0`)|
|59|3|Ignored|
|56|3|X / Y register index|
|0|56|Pointer|

For `ldz` / `stz`:

|Bit|Width|Meaning|
|---:|---:|---|
|63|1|Ignored|
|62|1|Load / store pair of registers (`1`) or single register (`0`)|
|56|6|Z row|
|0|56|Pointer|

For `ldzi` / `stzi`:

|Bit|Width|Meaning|
|---:|---:|---|
|62|2|Ignored|
|57|5|Z row (high 5 bits thereof)|
|56|1|Right hand half (`1`) or left hand half (`0`) of Z register pair|
|0|56|Pointer|

## Description

Move 64 bytes of data between memory (does not have to be aligned) and an AMX register, or move 128 bytes of data between memory (must be aligned to 128 bytes) and an adjacent pair of AMX registers. On M2/M3, can also move 256 bytes of data from memory to four consecutive X or Y registers. On M3, can move 128 or 256 bytes of data from memory to non-consecutive X or Y registers: if bit 61 is set, 128 bytes are moved to registers `n` and `(n+4)%8`, or 256 bytes are moved to registers `n`, `(n+2)%8`, `(n+4)%8`, `(n+6)%8`.

The `ldzi` and `stzi` instructions manipulate _half_ of a _pair_ of Z registers. Viewing the 64 bytes of memory and the 64 bytes of every Z register as vectors of i32 / u32 / f32, the mapping between memory and Z is:

<table>
<tr><th>Memory</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr>
<tr><th>Z0</th><td>0 L</td><td>2 L</td><td>4 L</td><td>6 L</td><td>8 L</td><td>10 L</td><td>12 L</td><td>14 L</td><td>0 R</td><td>2 R</td><td>4 R</td><td>6 R</td><td>8 R</td><td>10 R</td><td>12 R</td><td>14 R</td></tr>
<tr><th>Z1</th><td>1 L</td><td>3 L</td><td>5 L</td><td>7 L</td><td>9 L</td><td>11 L</td><td>13 L</td><td>15 L</td><td>1 R</td><td>3 R</td><td>5 R</td><td>7 R</td><td>9 R</td><td>11 R</td><td>13 R</td><td>15 R</td></tr>
</table>

In other words, the even Z register contains the even lanes from memory, and the odd Z register contains the odd lanes from memory. By a happy coincidence, this matches up with the "interleaved pair" lane arrangements of mixed-width [`mac16`](mac16.md) / [`fma16`](fma.md) / [`fms16`](fms.md) instructions, and with the "interleaved pair" lane arrangements of other instructions when in a (16, 16, 32) arrangement.

## Emulation code

See [ldst.c](ldst.c).

A representative sample is:
```c
void emulate_AMX_LDX(amx_state* state, uint64_t operand) {
    ld_common(state->x, operand, 7);
}

void ld_common(amx_reg* regs, uint64_t operand, uint32_t regmask) {
    uint32_t rn = (operand >> 56) & regmask;
    const uint8_t* src = (uint8_t*)((operand << 8) >> 8);
    memcpy(regs + rn, src, 64);
    if (operand & LDST_MULTIPLE) {
        uint32_t rs = 1;
        if ((AMX_VER >= AMX_VER_M3) && (operand & LDST_NON_CONSECUTIVE) && (regmask <= 15)) {
            rs = (operand & LDST_MULTIPLE_MEANS_FOUR) ? 2 : 4;
        }
        memcpy(regs + ((rn + rs) & regmask), src + 64, 64);
        if ((AMX_VER >= AMX_VER_M2) && (operand & LDST_MULTIPLE_MEANS_FOUR) && (regmask <= 15)) {
            memcpy(regs + ((rn + rs*2) & regmask), src + 128, 64);
            memcpy(regs + ((rn + rs*3) & regmask), src + 192, 64);
        }
    }
}
```
