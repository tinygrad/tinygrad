## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`matfp`|<code>z[j][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[j])</code>|9 bit X, 9 bit Y|Indexed X or Y, shuffle X, shuffle Y,<br/>positive selection|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `21`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|63|1|Ignored|
|57|6|Y enable value|Meaning dependent upon associated mode, see bit 23|
|54|3|Must be zero|No-op otherwise|
|53|1|[Indexed load](RegisterFile.md#indexed-loads) (`1`) or regular load (`0`)|
|(53=1) 52|1|Ignored|
|(53=1) 49|3|Register to index into|
|(53=1) 48|1|Indices are 4 bits (`1`) or 2 bits (`0`)|
|(53=1) 47|1|Indexed load of Y (`1`) or of X (`0`)|
|(53=0) 47|6|ALU mode|
|46|1|Ignored|
|42|4|Lane width mode||
|41|1|Ignored|
|38|3|X enable mode|
|37|1|Ignored|
|32|5|X enable value|Meaning dependent upon associated mode|
|31|1|Ignored|
|29|2|[X shuffle](RegisterFile.md#shuffles)|
|27|2|[Y shuffle](RegisterFile.md#shuffles)|
|26|1|Ignored|
|23|3|Y enable mode|Not high bits of Z row|
|20|3|Z row|High bits ignored in some lane width modes|
|19|1|Ignored|
|10|9|X offset (in bytes)|
|9|1|Ignored|
|0|9|Y offset (in bytes)|

ALU modes:
|Floating-point operation|47|Notes|
|---|---|---|
|`z + x*y`|`0`|
|`z - x*y`|`1`|
|`x <= 0 ? 0 : y`|`4`|Z input not used|
|no-op|anything else|

Lane width modes:
|X,Y|Z|42|Notes|
|---|---|---|---|
|bf16|bf16 (one row from each two)|`0`|M2 only|
|bf16|f32 (all rows, interleaved pairs)|`1`|M2 only|
|f16|f32 (all rows, interleaved pairs)|`3`|
|f32|f32 (one row from each four)|`4`|
|f64|f64 (one row from each eight)|`7`|
|f16|f16 (one row from each two)|anything else|

X/Y enable modes:
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0`), or odd lanes only (`1`), or even lanes only (`2`), or enable all lanes but override the ALU operation to `0.0` (`3`) or enable all lanes but override their value to `0.0` (`4` or `5`) or no lanes enabled (anything else) |
|`1`|Only enable lane #N|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|
|`4`|Only enable the first N lanes (no lanes when N is zero)|
|`5`|Only enable the last N lanes (no lanes when N is zero)|
|`6`|No lanes enabled|
|`7`|No lanes enabled|

## Description

Performs a fused-multiply-add (or other ALU operation) outer-product between an X vector, a Y vector, and a 2D grid of Z values, accumulating onto Z. All three of X and Y and Z have the same element type, either f16 or f32 or f64 (or bf16 on M2). Alternatively, when X and Y are both f16 (or bf16 on M2), Z can have type f32, in which case the entire 64x64 byte grid of Z is used, with even lanes of X going into even Z registers and odd lanes of X going into odd Z registers (see [Mixed lane widths](RegisterFile.md#mixed-lane-widths)).

## Emulation code

See [matfp.c](matfp.c), and [vecfp.c](vecfp.c) for the shared ALU. Note the code in [test.c](test.c) to set the DN bit of `fpcr`.

A representative sample is:
```c
void emulate_AMX_MATFP(amx_state* state, uint64_t operand) {
    if ((operand >> 54) & 7) {
        return;
    }
    operand &=~ (1ull << 37);
    operand &=~ (1ull << 63);

    int alumode = (operand & MATFP_INDEXED_LOAD) ? 0 : (operand >> 47) & 0x3f;
    if (alumode == 2 || alumode == 3 || alumode >= 5) {
        return;
    }

    uint32_t xybits, zbits, bf16 = 0;
    switch ((operand >> 42) & 0xf) {
    case  0: xybits = 16; if (AMX_VER >= AMX_VER_M2) { zbits = 16; bf16 = 1; } else { zbits = 16; } break;
    case  1: xybits = 16; if (AMX_VER >= AMX_VER_M2) { zbits = 32; bf16 = 1; } else { zbits = 16; } break;
    case  3: xybits = 16; zbits = 32; break;
    case  4: xybits = 32; zbits = 32; break;
    case  7: xybits = 64; zbits = 64; break;
    default: xybits = 16; zbits = 16; break;
    }
    uint32_t xybytes = xybits / 8;

    amx_reg x;
    amx_reg y;
    load_xy_reg(&x, state->x, (operand >> 10) & 0x1FF);
    load_xy_reg(&y, state->y, operand & 0x1FF);
    if (operand & MATFP_INDEXED_LOAD) {
        uint32_t src_reg = (operand >> 49) & 7;
        uint32_t ibits = (operand & MATFP_INDEXED_LOAD_4BIT) ? 4 : 2;
        if (operand & MATFP_INDEXED_LOAD_Y) {
            load_xy_reg_indexed(y.u8, state->y[src_reg].u8, ibits, xybits);
        } else {
            load_xy_reg_indexed(x.u8, state->x[src_reg].u8, ibits, xybits);
        }
    }
    xy_shuffle(x.u8, (operand >> 29) & 3, xybytes);
    xy_shuffle(y.u8, (operand >> 27) & 3, xybytes);

    uint64_t x_enable = parse_writemask(operand >> 32, xybytes, 9);
    uint64_t y_enable = parse_writemask((((operand >> 23) & 7) << 6) | (operand >> 58), xybytes, 9);
    int32_t omask = -1;
    if (((operand >> (32+6)) & 7) == 0) {
        uint32_t val = (operand >> 32) & 0x3F;
        if (val == 3) {
            omask = 0;
        } else if (val == 4 || val == 5) {
            memset(&x, 0, 64);
        }
    }
    if (((operand >> 23) & 7) == 0) {
        uint32_t val = (operand >> 58) & 0x3F;
        if (val == 3) {
            omask = 0;
        } else if (val == 4 || val == 5) {
            memset(&y, 0, 64);
        }
    }

    uint64_t z_row = (operand >> 20) & 7;
    if (zbits == 16) {
        if (bf16) {
            ...
        } else {
            for (uint32_t j = 0; j < 32; j += 1) {
                if (!((y_enable >> (j*xybytes)) & 1)) continue;
                for (uint32_t i = 0; i < 32; i += 1) {
                    if (!((x_enable >> (i*xybytes)) & 1)) continue;
                    _Float16* z = &state->z[bit_select(j*2, z_row, 1)].f16[i];
                    *z = omask ? vecfp_alu16(x.f16[i], y.f16[j], *z, alumode) : 0;
                }
            }
        }
    } else {
        ...
    }
}

_Float16 vecfp_alu16(_Float16 x, _Float16 y, _Float16 z, int alumode) {
    switch (alumode) {
    case 0: __asm("fmadd %h0, %h1, %h2, %h3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 1: __asm("fmsub %h0, %h1, %h2, %h3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 4: z = (x <= (_Float16)0) ? (_Float16)0 : y; break;
    }
    return z;
}
```

## Performance (M1 Max)

Note that a fused-multiply-add counts as _two_ floating-point operations. A measurement of 1.0 GFLOPS would mean 10<sup>9</sup> floating-point operations per second. The measurements are done without any load or store instructions; real-world workloads will need loads and stores, and thus will achieve lower numbers.

X and Y being `f16[32]`, each Z accumulator being `f16[32][32]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|1444.6 GFLOPS|2945.9 GFLOPS|2668.9 GFLOPS|4296.1 GFLOPS|4692.2 GFLOPS|5082.6 GFLOPS|
|2 per thread|2944.7 GFLOPS|5856.6 GFLOPS|4857.7 GFLOPS|6150.3 GFLOPS|5565.6 GFLOPS|6186.8 GFLOPS|

X and Y being `f16[32]`, each Z accumulator being `f32[32][32]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|1466.1 GFLOPS|2926.1 GFLOPS|2665.2 GFLOPS|2856.6 GFLOPS|2835.5 GFLOPS|2874.2 GFLOPS|

X and Y being `f32[16]`, each Z accumulator being `f32[16][16]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|367.0 GFLOPS|725.8 GFLOPS|923.2 GFLOPS|1053.6 GFLOPS|1572.3 GFLOPS|1418.4 GFLOPS|
|2 per thread|735.4 GFLOPS|1474.6 GFLOPS|1321.7 GFLOPS|1790.2 GFLOPS|2708.2 GFLOPS|2673.2 GFLOPS|
|3 per thread|1095.0 GFLOPS|2215.9 GFLOPS|2010.7 GFLOPS|2469.2 GFLOPS|2865.7 GFLOPS|2861.8 GFLOPS|
|4 per thread|1478.4 GFLOPS|2955.3 GFLOPS|2771.5 GFLOPS|2786.5 GFLOPS|2903.5 GFLOPS|2963.8 GFLOPS|

X and Y being `f64[8]`, each Z accumulator being `f64[8][8]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|91.6 GFLOPS|184.9 GFLOPS|210.1 GFLOPS|329.7 GFLOPS|411.7 GFLOPS|408.6 GFLOPS|
|2 per thread|184.7 GFLOPS|369.7 GFLOPS|334.1 GFLOPS|491.7 GFLOPS|720.2 GFLOPS|712.6 GFLOPS|
|3 per thread|276.2 GFLOPS|553.2 GFLOPS|546.5 GFLOPS|652.4 GFLOPS|757.9 GFLOPS|685.1 GFLOPS|
|4 per thread|364.6 GFLOPS|736.5 GFLOPS|702.4 GFLOPS|770.0 GFLOPS|767.5 GFLOPS|754.2 GFLOPS|
|5 per thread|368.3 GFLOPS|731.0 GFLOPS|596.8 GFLOPS|763.6 GFLOPS|793.2 GFLOPS|710.5 GFLOPS|
|6 per thread|369.5 GFLOPS|737.3 GFLOPS|776.0 GFLOPS|768.2 GFLOPS|794.7 GFLOPS|787.6 GFLOPS|
|7 per thread|369.1 GFLOPS|738.6 GFLOPS|606.1 GFLOPS|708.7 GFLOPS|787.0 GFLOPS|792.2 GFLOPS|
|8 per thread|369.6 GFLOPS|736.8 GFLOPS|686.6 GFLOPS|773.6 GFLOPS|779.9 GFLOPS|790.0 GFLOPS|
