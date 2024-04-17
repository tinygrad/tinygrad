## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`fma64`&nbsp;(63=0)<br/>`fma32`&nbsp;(63=0)<br/>`fma16`&nbsp;(63=0)|`z[j][i] += x[i] * y[j]`|7 bit X, 7 bit Y|X/Y/Z input disable|
|`fma64`&nbsp;(63=1)<br/>`fma32`&nbsp;(63=1)<br/>`fma16`&nbsp;(63=1)|`z[_][i] += x[i] * y[i]`|7 bit|X/Y/Z input disable|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|`10` for `fma64`<br/>`12` for `fma32`<br/>`15` for `fma16`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|63|1|Vector mode (`1`) or matrix mode (`0`)|
|62|1|Z is f32 (`1`) or Z is instruction width (`0`)|Only used by `fma16` in matrix mode, ignored otherwise|
|61|1|X is f16 (`1`) or X is instruction width (`0`)|Only used by `fma32`, ignored otherwise|
|60|1|Y is f16 (`1`) or Y is instruction width (`0`)|Only used by `fma32`, ignored otherwise|
|48|12|Ignored|
|46|2|X enable mode||
|41|5|X enable value|Meaning dependent upon associated mode|
|39|2|Ignored|
|37|2|Y enable mode|Ignored in vector mode|
|32|5|Y enable value|Ignored in vector mode<br/>Meaning dependent upon associated mode|
|30|2|Ignored|
|29|1|Skip X input (`1`) or use X input (`0`)|
|28|1|Skip Y input (`1`) or use Y input (`0`)|
|27|1|Skip Z input (`1`) or use Z input (`0`)|
|26|1|Ignored|
|20|6|Z row|High bits ignored in matrix mode|
|19|1|Ignored|
|10|9|X offset (in bytes)|
|9|1|Ignored|
|0|9|Y offset (in bytes)|

Combinations of bits 27-29 result in various floating-point ALU operations:

|Operation|29 (X)|28 (Y)|27 (Z)|
|---|---|---|---|
|`x*y+z`|`0`|`0`|`0`|
|<code>x*y&nbsp;&nbsp;</code>|`0`|`0`|`1`|
|<code>x&nbsp;&nbsp;+z</code>|`0`|`1`|`0`|
|<code>x&nbsp;&nbsp;&nbsp;&nbsp;</code>|`0`|`1`|`1`|
|<code>&nbsp;&nbsp;y+z</code>|`1`|`0`|`0`|
|<code>&nbsp;&nbsp;y&nbsp;&nbsp;</code>|`1`|`0`|`1`|
|<code>&nbsp;&nbsp;&nbsp;&nbsp;z</code>|`1`|`1`|`0`|
|`0`|`1`|`1`|`1`|

Combinations of the instruction and bits 60-63 result in various widths for X / Y / Z:

|Mode|X|Y|Z|63 (M)|62 (Z)|61 (X)|60 (Y)|Op|
|---|---|---|---|---|---|---|---|---|
|Matrix|f16|f16|f16 (one row from each two)|`0`|`0`|||`fma16`|
|Matrix|f16|f16|f32 (all rows, interleaved pairs)|`0`|`1`|||`fma16`|
|Matrix|f32|f32|f32 (one row from each four)|`0`||`0`|`0`|`fma32`|
|Matrix|f32|f16 (even lanes)|f32 (one row from each four)|`0`||`0`|`1`|`fma32`|
|Matrix|f16 (even lanes)|f32|f32 (one row from each four)|`0`||`1`|`0`|`fma32`|
|Matrix|f16 (even lanes)|f16 (even lanes)|f32 (one row from each four)|`0`||`1`|`1`|`fma32`|
|Matrix|f64|f64|f64 (one row from each eight)|`0`||||`fma64`|
|Vector|f16|f16|f16 (one row)|`1`||||`fma16`|
|Vector|f32|f32|f32 (one row)|`1`||`0`|`0`|`fma32`|
|Vector|f32|f16 (even lanes)|f32 (one row)|`1`||`0`|`1`|`fma32`|
|Vector|f16 (even lanes)|f32|f32 (one row)|`1`||`1`|`0`|`fma32`|
|Vector|f16 (even lanes)|f16 (even lanes)|f32 (one row)|`1`||`1`|`1`|`fma32`|
|Vector|f64|f64|f64 (one row)|`1`||||`fma64`|

X/Y enable modes:
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0`), or odd lanes only (`1`), or even lanes only (`2`), or no lanes (anything else) |
|`1`|Only enable lane #N|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|

## Description

In vector mode, performs a pointwise fused-multiply-add (or simplification thereof) operation between an X vector, a Y vector, and a Z vector, accumulating onto the Z vector. All three vectors have the same element type, either f16 or f32 or f64. Alternatively, when Z has type f32, X or Y (or both) can have type f16, though only the even lanes are used.

In matrix mode, performs a fused-multiply-add (or simplification thereof) outer-product between an X vector, a Y vector, and a 2D grid of Z values, accumulating onto Z. All three of X and Y and Z have the same element type, either f16 or f32 or f64. Alternatively, when Z has type f32, X or Y (or both) can have type f16, though only the even lanes are used. As a final alternative, when Z has type f32 and both X/Y have type f16, then all lanes of X and Y can be used in combination with the entire 64x64 byte grid of Z, with even lanes of X going into even Z registers and odd lanes of X going into odd Z registers (see [Mixed lane widths](RegisterFile.md#mixed-lane-widths)).

## Emulation code

See [fma.c](fma.c). Note the code in [test.c](test.c) to set the DN bit of `fpcr`.

A representative sample is:
```c
void emulate_AMX_FMA64(amx_state* state, uint64_t operand) {
    uint64_t y_offset = operand & 0x1FF;
    uint64_t x_offset = (operand >> 10) & 0x1FF;
    uint64_t z_row = (operand >> 20) & 63;
    uint64_t x_enable = parse_writemask(operand >> 41, 8, 7);
    uint64_t y_enable = parse_writemask(operand >> 32, 8, 7);

    double x[8];
    double y[8];
    load_xy_reg(x, state->x, x_offset);
    load_xy_reg(y, state->y, y_offset);

    for (int i = 0; i < 8; i++) {
        if (!((x_enable >> (i * 8)) & 1)) continue;
        if (operand & FMA_VECTOR_PRODUCT) {
            double* z = &state->z[z_row].f64[i];
            *z = fma64_alu(x[i], y[i], *z, operand);
        } else {
            for (int j = 0; j < 8; j++) {
                if (!((y_enable >> (j * 8)) & 1)) continue;
                double* z = &state->z[(j * 8) + (z_row & 7)].f64[i];
                *z = fma64_alu(x[i], y[j], *z, operand);
            }
        }
    }
}

double fma64_alu(double x, double y, double z, uint64_t operand) {
    switch ((operand >> 27) & 7) {
    case 1: return x * y;
    case 2: return z + x;
    case 3: return x;
    case 4: return z + y;
    case 5: return y;
    case 6: return z;
    case 7: return 0.;
    }
    double result;
    __asm("fmadd %d0, %d1, %d2, %d3" : "=w"(result) : "w"(x), "w"(y), "w"(z));
    return result;
}
```

## Performance (M1 Max)

Note that a fused-multiply-add counts as _two_ floating-point operations. A measurement of 1.0 GFLOPS would mean 10<sup>9</sup> floating-point operations per second. The measurements are done without any load or store instructions; real-world workloads will need loads and stores, and thus will achieve lower numbers.

`fma16` in matrix mode, each Z accumulator being `f16[32][32]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|1453.0 GFLOPS|2958.4 GFLOPS|2705.5 GFLOPS|3553.5 GFLOPS|4609.2 GFLOPS|5268.5 GFLOPS|
|2 per thread|2958.9 GFLOPS|5915.7 GFLOPS|4862.3 GFLOPS|5355.6 GFLOPS|5546.6 GFLOPS|6263.4 GFLOPS|

`fma16` in matrix mode, each Z accumulator being `f32[32][32]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|1459.7 GFLOPS|2948.3 GFLOPS|2842.5 GFLOPS|2626.4 GFLOPS|2892.2 GFLOPS|2909.9 GFLOPS|

`fma32` in matrix mode, each Z accumulator being `f32[16][16]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|367.6 GFLOPS|739.1 GFLOPS|866.1 GFLOPS|1108.5 GFLOPS|1388.4 GFLOPS|1512.4 GFLOPS|
|2 per thread|736.4 GFLOPS|1478.1 GFLOPS|1335.4 GFLOPS|1796.2 GFLOPS|2606.3 GFLOPS|2470.7 GFLOPS|
|3 per thread|1108.4 GFLOPS|2217.9 GFLOPS|1878.5 GFLOPS|2507.2 GFLOPS|2564.2 GFLOPS|2798.4 GFLOPS|
|4 per thread|1475.0 GFLOPS|2956.7 GFLOPS|2429.8 GFLOPS|3077.4 GFLOPS|2894.6 GFLOPS|3118.7 GFLOPS|

`fma64` in matrix mode, each Z accumulator being `f64[8][8]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|92.1 GFLOPS|184.3 GFLOPS|214.9 GFLOPS|311.7 GFLOPS|416.9 GFLOPS|381.8 GFLOPS|
|2 per thread|183.6 GFLOPS|366.6 GFLOPS|369.3 GFLOPS|516.6 GFLOPS|697.7 GFLOPS|641.6 GFLOPS|
|3 per thread|275.6 GFLOPS|553.5 GFLOPS|548.2 GFLOPS|650.5 GFLOPS|768.4 GFLOPS|758.4 GFLOPS|
|4 per thread|369.1 GFLOPS|738.1 GFLOPS|603.6 GFLOPS|706.5 GFLOPS|780.0 GFLOPS|756.3 GFLOPS|
|5 per thread|368.4 GFLOPS|736.8 GFLOPS|604.6 GFLOPS|725.2 GFLOPS|777.6 GFLOPS|797.7 GFLOPS|
|6 per thread|368.6 GFLOPS|738.5 GFLOPS|603.6 GFLOPS|689.8 GFLOPS|775.0 GFLOPS|776.6 GFLOPS|
|7 per thread|368.4 GFLOPS|738.0 GFLOPS|604.5 GFLOPS|713.3 GFLOPS|792.6 GFLOPS|739.2 GFLOPS|
|8 per thread|369.2 GFLOPS|739.5 GFLOPS|602.7 GFLOPS|733.3 GFLOPS|742.5 GFLOPS|768.9 GFLOPS|

`fma16` in vector mode, each Z accumulator being `f16[32]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|45.8 GFLOPS|92.6 GFLOPS|105.2 GFLOPS|141.2 GFLOPS|173.0 GFLOPS|226.9 GFLOPS|
|2 per thread|90.6 GFLOPS|181.0 GFLOPS|199.6 GFLOPS|276.9 GFLOPS|335.5 GFLOPS|352.0 GFLOPS|
|3 per thread|138.8 GFLOPS|276.9 GFLOPS|330.9 GFLOPS|381.4 GFLOPS|463.1 GFLOPS|453.1 GFLOPS|
|4 per thread|184.7 GFLOPS|366.0 GFLOPS|428.0 GFLOPS|514.6 GFLOPS|649.0 GFLOPS|574.5 GFLOPS|
|5 per thread|230.9 GFLOPS|462.5 GFLOPS|469.5 GFLOPS|629.4 GFLOPS|671.7 GFLOPS|683.1 GFLOPS|
|6 per thread|271.3 GFLOPS|553.7 GFLOPS|536.7 GFLOPS|713.7 GFLOPS|765.5 GFLOPS|768.7 GFLOPS|
|7 per thread|322.7 GFLOPS|647.3 GFLOPS|597.3 GFLOPS|762.7 GFLOPS|768.0 GFLOPS|738.8 GFLOPS|
|8 per thread|366.7 GFLOPS|729.0 GFLOPS|655.7 GFLOPS|760.8 GFLOPS|742.7 GFLOPS|806.2 GFLOPS|
|9 per thread|342.6 GFLOPS|688.9 GFLOPS|622.8 GFLOPS|742.7 GFLOPS|750.7 GFLOPS|802.1 GFLOPS|
|10 per thread|362.7 GFLOPS|716.3 GFLOPS|642.6 GFLOPS|785.9 GFLOPS|768.7 GFLOPS|793.7 GFLOPS|
|11 per thread|358.3 GFLOPS|716.1 GFLOPS|660.6 GFLOPS|798.9 GFLOPS|798.8 GFLOPS|822.2 GFLOPS|
|12 per thread|361.7 GFLOPS|730.8 GFLOPS|662.4 GFLOPS|783.0 GFLOPS|752.6 GFLOPS|823.2 GFLOPS|
|13 per thread|368.3 GFLOPS|735.9 GFLOPS|669.9 GFLOPS|804.4 GFLOPS|794.5 GFLOPS|778.2 GFLOPS|
|14 per thread|367.6 GFLOPS|733.3 GFLOPS|653.3 GFLOPS|802.9 GFLOPS|781.6 GFLOPS|790.2 GFLOPS|
|15 per thread|360.2 GFLOPS|725.4 GFLOPS|661.7 GFLOPS|785.4 GFLOPS|759.2 GFLOPS|822.9 GFLOPS|
|16 per thread|370.6 GFLOPS|733.4 GFLOPS|674.5 GFLOPS|797.6 GFLOPS|805.8 GFLOPS|818.7 GFLOPS|

`fma32` in vector mode, each Z accumulator being `f32[16]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|23.0 GFLOPS|46.3 GFLOPS|53.2 GFLOPS|70.8 GFLOPS|88.1 GFLOPS|104.5 GFLOPS|
|2 per thread|46.3 GFLOPS|92.4 GFLOPS|106.1 GFLOPS|134.2 GFLOPS|188.9 GFLOPS|178.0 GFLOPS|
|3 per thread|68.7 GFLOPS|137.5 GFLOPS|153.6 GFLOPS|199.1 GFLOPS|239.1 GFLOPS|255.8 GFLOPS|
|4 per thread|92.5 GFLOPS|184.4 GFLOPS|196.8 GFLOPS|264.6 GFLOPS|285.1 GFLOPS|282.8 GFLOPS|
|5 per thread|114.9 GFLOPS|230.0 GFLOPS|225.3 GFLOPS|299.6 GFLOPS|354.0 GFLOPS|367.2 GFLOPS|
|6 per thread|134.3 GFLOPS|276.0 GFLOPS|257.3 GFLOPS|344.1 GFLOPS|399.2 GFLOPS|380.9 GFLOPS|
|7 per thread|161.3 GFLOPS|322.7 GFLOPS|297.6 GFLOPS|372.5 GFLOPS|389.1 GFLOPS|401.8 GFLOPS|
|8 per thread|183.1 GFLOPS|370.2 GFLOPS|331.2 GFLOPS|381.4 GFLOPS|393.3 GFLOPS|389.6 GFLOPS|
|9 per thread|170.6 GFLOPS|351.8 GFLOPS|323.3 GFLOPS|390.9 GFLOPS|395.6 GFLOPS|410.6 GFLOPS|
|10 per thread|180.7 GFLOPS|357.5 GFLOPS|328.3 GFLOPS|395.6 GFLOPS|397.2 GFLOPS|399.6 GFLOPS|
|11 per thread|185.0 GFLOPS|366.8 GFLOPS|330.3 GFLOPS|400.0 GFLOPS|399.6 GFLOPS|385.4 GFLOPS|
|12 per thread|183.2 GFLOPS|369.7 GFLOPS|332.3 GFLOPS|399.2 GFLOPS|379.9 GFLOPS|403.0 GFLOPS|
|13 per thread|184.1 GFLOPS|370.0 GFLOPS|401.0 GFLOPS|389.6 GFLOPS|383.9 GFLOPS|408.4 GFLOPS|
|14 per thread|184.3 GFLOPS|368.2 GFLOPS|326.5 GFLOPS|400.6 GFLOPS|394.8 GFLOPS|406.5 GFLOPS|
|15 per thread|181.8 GFLOPS|369.4 GFLOPS|335.7 GFLOPS|404.6 GFLOPS|404.5 GFLOPS|397.5 GFLOPS|
|16 per thread|183.0 GFLOPS|368.0 GFLOPS|334.9 GFLOPS|402.2 GFLOPS|399.4 GFLOPS|393.5 GFLOPS|

`fma64` in vector mode, each Z accumulator being `f64[8]`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|11.5 GFLOPS|23.1 GFLOPS|26.7 GFLOPS|35.9 GFLOPS|44.2 GFLOPS|48.7 GFLOPS|
|2 per thread|23.2 GFLOPS|46.3 GFLOPS|53.4 GFLOPS|70.5 GFLOPS|89.5 GFLOPS|102.6 GFLOPS|
|3 per thread|34.7 GFLOPS|69.5 GFLOPS|79.9 GFLOPS|104.6 GFLOPS|120.6 GFLOPS|108.4 GFLOPS|
|4 per thread|45.8 GFLOPS|92.4 GFLOPS|104.3 GFLOPS|127.4 GFLOPS|156.8 GFLOPS|146.7 GFLOPS|
|5 per thread|57.5 GFLOPS|114.7 GFLOPS|119.6 GFLOPS|159.0 GFLOPS|180.2 GFLOPS|172.7 GFLOPS|
|6 per thread|69.2 GFLOPS|138.7 GFLOPS|135.7 GFLOPS|179.9 GFLOPS|188.9 GFLOPS|200.9 GFLOPS|
|7 per thread|80.9 GFLOPS|159.8 GFLOPS|151.4 GFLOPS|181.4 GFLOPS|197.2 GFLOPS|199.8 GFLOPS|
|8 per thread|91.7 GFLOPS|183.5 GFLOPS|170.0 GFLOPS|199.0 GFLOPS|209.2 GFLOPS|196.0 GFLOPS|
|9 per thread|85.3 GFLOPS|169.9 GFLOPS|159.9 GFLOPS|198.0 GFLOPS|199.3 GFLOPS|202.9 GFLOPS|
|10 per thread|91.2 GFLOPS|180.4 GFLOPS|165.2 GFLOPS|199.5 GFLOPS|198.6 GFLOPS|206.1 GFLOPS|
|11 per thread|92.1 GFLOPS|184.1 GFLOPS|167.1 GFLOPS|200.7 GFLOPS|200.7 GFLOPS|198.6 GFLOPS|
|12 per thread|92.5 GFLOPS|185.1 GFLOPS|168.8 GFLOPS|198.5 GFLOPS|201.3 GFLOPS|203.1 GFLOPS|
|13 per thread|92.3 GFLOPS|183.2 GFLOPS|168.5 GFLOPS|200.6 GFLOPS|201.4 GFLOPS|208.8 GFLOPS|
|14 per thread|92.7 GFLOPS|185.1 GFLOPS|199.0 GFLOPS|201.3 GFLOPS|200.4 GFLOPS|209.8 GFLOPS|
|15 per thread|92.6 GFLOPS|184.8 GFLOPS|168.2 GFLOPS|201.7 GFLOPS|200.1 GFLOPS|208.0 GFLOPS|
|16 per thread|92.2 GFLOPS|185.1 GFLOPS|168.1 GFLOPS|202.2 GFLOPS|201.3 GFLOPS|205.3 GFLOPS|
