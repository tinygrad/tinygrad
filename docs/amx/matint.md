## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`matint`&nbsp;(47≠4)|<code>z[j][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[j])</code>|9 bit X or Y|Indexed X or Y, shuffle X, shuffle Y,<br/>right shift, `sqrdmlah`, `popcnt`|
|`matint`&nbsp;(47=4)|<code>z[j][i]&nbsp;&nbsp;=&nbsp;f(z[j][i])</code>|9 bit X or Y|Right shift, saturation|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `20`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|(47=4) 63|1|Z is signed (`1`) or unsigned (`0`)|
|(47≠4) 63|1|X is signed (`1`) or unsigned (`0`)|
|58|5|Right shift amount|Ignored when ALU mode in {5, 6, 9}|
|57|1|Ignored|
|55|2|Must be zero|No-op otherwise|
|(53=1) 54|1|ALU mode 8 (`1`) or mode 0 (`0`)|
|(53=0) 54|1|Must be zero|No-op otherwise|
|53|1|[Indexed load](RegisterFile.md#indexed-loads) (`1`) or regular load (`0`)|
|(53=1) 52|1|Ignored|
|(53=1) 49|3|Register to index into|
|(53=1) 48|1|Indices are 4 bits (`1`) or 2 bits (`0`)|
|(53=1) 47|1|Indexed load of Y (`1`) or of X (`0`)|
|(53=0) 47|6|ALU mode|
|46|1|Ignored|
|42|4|Lane width mode|Meaning dependent upon ALU mode|
|41|1|Ignored|
|38|3|X or Y enable mode|See bit 25|
|32|6|X or Y enable value|Meaning dependent upon associated mode|
|31|1|Ignored|
|(47=4) 30|1|Saturate Z (`1`) or truncate Z (`0`)|
|(47=4) 29|1|Right shift is rounding (`1`) or truncating (`0`)|
|(47≠4) 29|2|[X shuffle](RegisterFile.md#shuffles)|
|27|2|[Y shuffle](RegisterFile.md#shuffles)|
|(47=4) 26|1|Z saturation is signed (`1`) or unsigned (`0`)|
|(47≠4) 26|1|Y is signed (`1`) or unsigned (`0`)|
|25|1|Enable mode is Y (`1`) or is X (`0`)|
|22|3|Ignored|Would be middle bits of Z row|
|20|2|Z row|High bits ignored in some lane width modes|
|19|1|Ignored|
|10|9|X offset (in bytes)|
|9|1|Ignored|
|0|9|Y offset (in bytes)|

ALU modes:
|Integer operation|47|Notes|
|---|---|---|
|`z+((x*y)>>s)`|`0`|
|`z-((x*y)>>s)`|`1`|
|`z+((x+y)>>s)`|`2`|Particular write enable mode can skip `x` or `y`|
|`z-((x+y)>>s)`|`3`|Particular write enable mode can skip `x` or `y`|
|`z>>s` or `sat(z>>s)`|`4`|Shift can be rounding, saturation is optional|
|`sat(z+((x*y*2)>>16))`|`5`|Shift is rounding, saturation is signed|
|`sat(z-((x*y*2)>>16))`|`6`|Shift is rounding, saturation is signed|
|no-op|`7`|
|`z+((x*y)>>s)`|`8`|Same as `0`, but different lane width modes|
|`z+popcnt(~(x^y))`|`9`|See [XNOR-Net](https://arxiv.org/abs/1603.05279)|
|no-op|anything else|

When ALU mode < 4, lane width modes:
|X,Y|Z|42|
|---|---|---|
|i16 or u16|i32 or u32 (all rows, interleaved pairs)|`3`|
|i16 or u16|i16 or u16 (one row from each two)|anything else|

When ALU mode = 4, lane width modes:
|Z |Z saturation|42|
|---|---|---|
|i32 or u32 (one row from each four)|i16 or u16|`3`|
|i32 or u32 (one row from each four)|i32 or u32|`4`|
|i32 or u32 (one row from each four)|i8 or u8|`10`|
|i16 or u16 (one row from each two)|i8 or u8|`11`|
|i16 or u16 (one row from each two)|i16 or u16|anything else|

When ALU mode in {5, 6}, lane width modes:

|X,Y|Z|42|
|---|---|---|
|i16 or u16|i16 or u16 (one row from each two)|anything|

When ALU mode = 8, lane width modes:
|X|Y|Z|42|
|---|---|---|---|
|i8 or u8|i8 or u8 (only every fourth lane is used, said lanes are used four times each)|i32 or u32 (all rows)|`10`|
|i8 or u8|i16 or u16 (only even lanes used, said lanes are used four times each)|i32 or u32 (all rows)|`12` (M3 only)|
|i8 or u8|i8 or u8 (only even lanes used, said lanes are used twice each)|i16 or u16 (all rows)|anything else|

When ALU mode = 9, lane width modes:

|X,Y|Z|42|
|---|---|---|
|i16 or u16|i32 or u32 (all rows, interleaved pairs)|`3`|
|i32 or u32|i32 or u32 (one row from each four)|`4`|
|i16 or u16|i16 or u16 (one row from each two)|anything else|

X or Y enable modes:
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0`), or odd lanes only (`1`), or even lanes only (`2`), or enable all lanes but override the ALU operation to `0` (`3`) or enable all lanes but override their value to `0` (`4` or `5`) or no lanes enabled (anything else) |
|`1`|Only enable lane #N|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|
|`4`|Only enable the first N lanes (no lanes when N is zero)|
|`5`|Only enable the last N lanes (no lanes when N is zero)|
|`6`|No lanes enabled|
|`7`|No lanes enabled|

## Description

When 47=4, performs an in-place reduction of a 2D grid of Z values, where reduction means right shift (either rounding or truncating), optionally followed by saturation (to i8/u8/i16/u16/i32/u32). Z values are 16 bit or 32 bit integers.

When 47≠4, performs some ALU operation in an outer-product manner between an X vector, a Y vector, and a 2D grid of Z values, accumulating onto Z. Various combinations of line widths are permitted.

When 47=8, FMA is performed with 8-bit multiplicands, accumlating onto 16-bit or 32-bit Z, using a slightly unusual (for AMX) data layout. For 32-bit Z, each 4 by 4 block of bytes ends up looking like:

<table><tr><td/><td>X<sub>0</sub></td><td>X<sub>1</sub></td><td>X<sub>2</sub></td><td>X<sub>3</sub></td></tr>
<tr><td>Y<sub>0</sub></td><td colspan="4">Z<sub>0,0:3</sub> += X<sub>0</sub> × Y<sub>0</sub></tr>
<tr><td>Y<sub>1</sub></td><td colspan="4">Z<sub>1,0:3</sub> += X<sub>1</sub> × Y<sub>0</sub></td>
<tr><td>Y<sub>2</sub></td><td colspan="4">Z<sub>2,0:3</sub> += X<sub>2</sub> × Y<sub>0</sub></tr>
<tr><td>Y<sub>3</sub></td><td colspan="4">Z<sub>3,0:3</sub> += X<sub>3</sub> × Y<sub>0</sub></td>
</table>

For 16-bit Z, each 2 by 2 block of bytes ends up looking like:

<table><tr><td/><td>X<sub>0</sub></td><td>X<sub>1</sub></td></tr>
<tr><td>Y<sub>0</sub></td><td colspan="2">Z<sub>0,0:1</sub> += X<sub>0</sub> × Y<sub>0</sub></tr>
<tr><td>Y<sub>1</sub></td><td colspan="2">Z<sub>1,0:1</sub> += X<sub>1</sub> × Y<sub>0</sub></td>
</table>

When 47=8, M3 also supports 8-bit X, 16-bit Y, 32-bit Z, with each 4 by 4 block of bytes looking like:

<table><tr><td/><td>X<sub>0</sub></td><td>X<sub>1</sub></td><td>X<sub>2</sub></td><td>X<sub>3</sub></td></tr>
<tr><td>Y<sub>0</sub></td><td colspan="4">Z<sub>0,0:3</sub> += X<sub>0</sub> × Y<sub>0:1</sub></tr>
<tr><td>Y<sub>1</sub></td><td colspan="4">Z<sub>1,0:3</sub> += X<sub>1</sub> × Y<sub>0:1</sub></td>
<tr><td>Y<sub>2</sub></td><td colspan="4">Z<sub>2,0:3</sub> += X<sub>2</sub> × Y<sub>0:1</sub></tr>
<tr><td>Y<sub>3</sub></td><td colspan="4">Z<sub>3,0:3</sub> += X<sub>3</sub> × Y<sub>0:1</sub></td>
</table>

## Emulation code

See [matint.c](matint.c), and [vecint.c](vecint.c) for the shared ALU.

A representative sample is:
```c
void emulate_AMX_MATINT(amx_state* state, uint64_t operand) {
    if ((operand >> 55) & 3) {
        return;
    }
    if ((operand & (1ull << 54)) && !(operand & MATINT_INDEXED_LOAD)) {
        return;
    }

    uint64_t z_row = (operand >> 20) & 3;
    int32_t omask = (((operand >> 32) & 0x1ff) == 3) ? 0 : -1;
    int alumode = (operand & MATINT_INDEXED_LOAD) ? (operand & (1ull << 54)) >> 51 : (operand >> 47) & 0x3f;
    uint32_t shift = (operand >> 58) & 0x1f;

    uint32_t xbits = 0, ybits = 0, zbits, satbits;
    if (alumode == 4) {
        switch ((operand >> 42) & 0xf) {
        case  3: zbits = 32; satbits = 16; break;
        case  4: zbits = 32; satbits = 32; break;
        case 10: zbits = 32; satbits =  8; break;
        case 11: zbits = 16; satbits =  8; break;
        default: zbits = 16; satbits = 16; break;
        }
    } else if (alumode == 5 || alumode == 6) {
        xbits = ybits = 16; zbits = 16;
        shift = 15;
    } else if (alumode == 8) {
        switch ((operand >> 42) & 0xf) {
        case 10: xbits = ybits = 8; zbits = 32; break;
        case 12: xbits = 8; if (AMX_VER >= AMX_VER_M3) { ybits = 16; zbits = 32; } else { ybits = 8; zbits = 16; } break;
        default: xbits = ybits = 8; zbits = 16; break;
        }
    } else if (alumode == 9) {
        switch ((operand >> 42) & 0xf) {
        case  3: xbits = ybits = 16; zbits = 32; break;
        case  4: xbits = ybits = 32; zbits = 32; break;
        default: xbits = ybits = 16; zbits = 16; break;
        }
        shift = 64 - xbits; // Not actually used as a shift
    } else {
        switch ((operand >> 42) & 0xf) {
        case  3: xbits = ybits = 16; zbits = 32; break;
        default: xbits = ybits = 16; zbits = 16; break;
        }
    }
    uint32_t xbytes = xbits / 8;
    uint32_t ybytes = ybits / 8;
    uint32_t zbytes = zbits / 8;
    
    if (alumode == 4) {
        ...
        return;
    } else if (alumode == 7 || alumode >= 10) {
        return;
    }

    uint8_t x[64];
    uint8_t y[64];
    load_xy_reg(x, state->x, (operand >> 10) & 0x1FF);
    load_xy_reg(y, state->y, operand & 0x1FF);
    if (operand & MATINT_INDEXED_LOAD) {
        uint32_t src_reg = (operand >> 49) & 7;
        uint32_t ibits = (operand & MATINT_INDEXED_LOAD_4BIT) ? 4 : 2;
        if (operand & MATINT_INDEXED_LOAD_Y) {
            load_xy_reg_indexed(y, state->y[src_reg].u8, ibits, ybits);
        } else {
            load_xy_reg_indexed(x, state->x[src_reg].u8, ibits, xbits);
        }
    }
    xy_shuffle(x, (operand >> 29) & 3, xbytes);
    xy_shuffle(y, (operand >> 27) & 3, ybytes);

    uint64_t x_enable, y_enable;
    if (operand & MATINT_ENABLE_MASK_IS_Y) {
        x_enable = ~(uint64_t)0;
        y_enable = parse_writemask(operand >> 32, ybytes, 9);
    } else {
        x_enable = parse_writemask(operand >> 32, xbytes, 9);
        y_enable = ~(uint64_t)0;
    }
    if (((operand >> (32+6)) & 7) == 0) {
        uint32_t val = (operand >> 32) & 0x3F;
        if (val == 4 || val == 5) {
            memset((operand & MATINT_ENABLE_MASK_IS_Y) ? y : x, 0, 64);
        }
    }

    uint32_t xsignext = (operand & MATINT_SIGNED_X) ? (64 - xbits) : 0;
    uint32_t ysignext = (operand & MATINT_SIGNED_Y) ? (64 - ybits) : 0;
    uint32_t zsignext = 64 - zbits;
    uint32_t zmask = (zbytes / xbytes) - 1;
    uint32_t step = xbytes == 1 ? zbytes : xbytes;
    for (uint32_t j = 0; j < 64; j += step) {
        if (!((y_enable >> j) & 1)) continue;
        for (uint32_t i = 0; i < 64; i += xbytes) {
            if (!((x_enable >> i) & 1)) continue;
            int64_t xv = load_int(x + i, xbytes, xsignext);
            int64_t yv = load_int(y + j, ybytes, ysignext);
            void* z = &state->z[bit_select(bit_select(j, z_row, xbytes - 1), i / xbytes, zmask)].u8[i & -zbytes];
            int64_t zv = load_int(z, zbytes, zsignext);
            int64_t result = vecint_alu(xv, yv, zv, alumode, shift) & omask;
            store_int(z, zbytes, result);
        }
    }
}

int64_t vecint_alu(int64_t x, int64_t y, int64_t z, int alumode, uint32_t shift) {
    int64_t val = x * y;
    if (alumode == 5 || alumode == 6) {
        val += 1ull << (shift - 1);
    } else if (alumode == 2 || alumode == 3) {
        val = x + y;
    } else if (alumode == 9) {
        return z + __builtin_popcountll((~(x ^ y)) << shift);
    }
    val >>= shift;
    if (alumode == 1 || alumode == 3 || alumode == 6) {
        val = -val;
    }
    val += z;
    if (alumode == 5 || alumode == 6) {
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
    }
    return val;
}
```
