## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`vecfp`|<code>z[_][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[i])</code>|9 bit|Indexed X or Y, shuffle X, shuffle Y,<br/>broadcast Y element,<br/>positive selection, `min`, `max`|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `19`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|57|7|Ignored|
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
|(31=1)&nbsp;35|6|Ignored|
|(31=1)&nbsp;32|3|Broadcast mode|
|(31=0)&nbsp;38|3|Write enable or broadcast mode|
|(31=0)&nbsp;37|1|Ignored|
|(31=0)&nbsp;32|5|Write enable value or broadcast lane index|Meaning dependent upon associated mode|
|31|1|Perform operation for multiple vectors (`1`)<br/>or just one vector (`0`)|M2 only (always reads as `0` on M1)|
|29|2|[X shuffle](RegisterFile.md#shuffles)|
|27|2|[Y shuffle](RegisterFile.md#shuffles)|
|26|1|Ignored|
|(31=1)&nbsp;25|1|"Multiple" means four vectors (`1`)<br/>or two vectors (`0`)|Top two bits of Z row ignored if operating on four vectors|
|20|6|Z row|Low bits ignored in some lane width modes<br/>When 31=1, top bit or top two bits ignored|
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
|`min(x, z)`|`5`|Y input not used|
|`max(x, z)`|`7`|Y input not used|
|`x * y`|`10`|M2 only. Z input not used|
|`z + x`|`11`|M2 only. Y input not used|
|`z + y`|`12`|M2 only. X input not used|
|no-op|anything else|

Lane width modes:
|X,Y|Z|42|Notes|
|---|---|---|---|
|bf16|bf16 (one row)|`0`|M2 only|
|bf16|f32 (two rows, interleaved pair)|`1`|M2 only|
|f16|f32 (two rows, interleaved pair)|`3`|
|f32|f32 (one row)|`4`|
|f64|f64 (one row)|`7`|
|f16|f16 (one row)|anything else|

Write enable or broadcast modes when 31=0:
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0`), or odd lanes only (`1`), or even lanes only (`2`), or enable all lanes but override the ALU output to `0.0` (`3`) or enable all lanes but override X values to `0.0` (`4`) or enable all lanes but override Y values to `0.0` (`5`) or no lanes enabled (anything else) |
|`1`|Enable all lanes, but broadcast Y lane #N to all lanes of Y|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|
|`4`|Only enable the first N lanes (no lanes when N is zero)|
|`5`|Only enable the last N lanes (no lanes when N is zero)|
|`6`|No lanes enabled|
|`7`|No lanes enabled|

Broadcast modes when 31=1:
|Mode|X inputs|Y inputs|Other effects|
|---:|---|---|---|
|`0`|Consecutive registers|Consecutive registers|
|`1`|Ignored|Ignored|Override ALU output to `0.0`|
|`2`|Use same register for every iteration|Consecutive registers|
|`3`|Consecutive registers|Use same register for every iteration|
|`4`|Override values to `0.0`|Consecutive registers|
|`5`|Consecutive registers|Override values to `0.0`|
|`6`|Use same register for every iteration,<br/>and broadcast lane #0 to all lanes|Consecutive registers|
|`7`|Consecutive registers|Use same register for every iteration,<br/>and broadcast lane #0 to all lanes|

## Description

Performs a pointwise fused-multiply-add (or other ALU operation) between an X vector, a Y vector, and a Z vector, accumulating onto the Z vector. All three vectors have the same element type, either f16 or f32 or f64 (or bf16 on M2). Alternatively, when X and Y are both f16 (or bf16 on M2), Z can have type f32, in which case two rows of Z are used (see [Mixed lane widths](RegisterFile.md#mixed-lane-widths)).

On M2, the whole operation can optionally be repeated multiple times, by setting bit 31. Bit 25 controls the repetition count; either two times or four times. By default, consecutive X or Y registers are used as the source operands, but broadcast mode settings can cause the same vector (or lane therein) to be used multiple times. If repeated twice, the top bit of Z row is ignored, and Z row is incremented by 32 for the 2<sup>nd</sup> iteration. If repeated four times, the top two bits of Z row are ignored, and Z row is incremented by 16 on each iteration.

## Emulation code

See [vecfp.c](vecfp.c). Note the code in [test.c](test.c) to set the DN bit of `fpcr`.

A representative sample is:
```c
void emulate_AMX_VECFP(amx_state* state, uint64_t operand) {
    if ((operand >> 54) & 7) {
        return;
    }
    operand &=~ (1ull << 37);

    int alumode = (operand & VECFP_INDEXED_LOAD) ? 0 : (operand >> 47) & 0x3f;
    switch (alumode) {
    case 0: case 1: case 4: case 5: case 7:
        break;
    case 10: case 11: case 12:
        if (AMX_VER >= AMX_VER_M2) {
            break;
        } else {
            return;
        }
    default:
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

    uint64_t z_row = operand >> 20;
    uint64_t z_step = 64;
    uint64_t x_step = 64;
    uint64_t y_step = 64;
    int32_t ximask = -1;
    if ((AMX_VER >= AMX_VER_M2) && (operand & (1ull << 31))) {
        uint64_t bmode = (operand >> 32) & 0x7;
        operand &=~ (0x1ffull << 32);
        switch (bmode) {
        case 1: operand |= 3ull << 32; break; // override ALU operation to 0
        case 2: x_step = 0; break; // same x vector for all operations
        case 3: y_step = 0; break; // same y vector for all operations
        case 4: operand |= 4ull << 32; break; // override x operand to zero
        case 5: operand |= 5ull << 32; break; // override y operand to zero
        case 6: x_step = 0; ximask = 0; break; // use lane 0 of x vector 0 for all operations
        case 7: y_step = 0; operand |= 1ull << 38; break; // use lane 0 of y vector 0 for all operations
        }
        z_step = z_row & 32 ? 16 : 32;
    }

    uint64_t x_offset = operand >> 10;
    uint64_t y_offset = operand;
    for (z_row &= z_step - 1; z_row <= 63; z_row += z_step) {
        amx_reg x;
        amx_reg y;
        load_xy_reg(&x, state->x, x_offset & 0x1FF); x_offset += x_step;
        load_xy_reg(&y, state->y, y_offset & 0x1FF); y_offset += y_step;
        if (operand & VECFP_INDEXED_LOAD) {
            uint32_t src_reg = (operand >> 49) & 7;
            uint32_t ibits = (operand & VECFP_INDEXED_LOAD_4BIT) ? 4 : 2;
            if (operand & VECFP_INDEXED_LOAD_Y) {
                load_xy_reg_indexed(y.u8, state->y[src_reg].u8, ibits, xybits);
                y_offset -= y_step - y_step * ibits / xybits;
            } else {
                load_xy_reg_indexed(x.u8, state->x[src_reg].u8, ibits, xybits);
                x_offset -= x_step - x_step * ibits / xybits;
            }
        }
        xy_shuffle(x.u8, (operand >> 29) & 3, xybytes);
        xy_shuffle(y.u8, (operand >> 27) & 3, xybytes);

        uint64_t x_enable = parse_writemask(operand >> 32, xybytes, 9);
        bool broadcast_y = ((operand >> (32+6)) & 7) == 1;
        int32_t omask = -1;
        if (broadcast_y) {
            x_enable = ~(uint64_t)0;
        } else if (((operand >> (32+6)) & 7) == 0) {
            uint32_t val = (operand >> 32) & 0x3F;
            if (val == 3) {
                omask = 0;
            } else if (val == 4) {
                memset(&x, 0, 64);
            } else if (val == 5) {
                memset(&y, 0, 64);
            }
        }

        if (zbits == 16) {
            if (bf16) {
                ...
            } else {
                for (uint32_t i = 0; i < 32; i += 1) {
                    if (!((x_enable >> (i*xybytes)) & 1)) continue;
                    uint32_t j = broadcast_y ? ((operand >> 32) & 0x1f) : i;
                    _Float16* z = &state->z[z_row].f16[i];
                    *z = omask ? vecfp_alu16(x.f16[i & ximask], y.f16[j], *z, alumode) : 0;
                }
            }
        } else {
            ...
        }
    }
}

_Float16 vecfp_alu16(_Float16 x, _Float16 y, _Float16 z, int alumode) {
    switch (alumode) {
    case 0: __asm("fmadd %h0, %h1, %h2, %h3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 1: __asm("fmsub %h0, %h1, %h2, %h3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 4: z = (x <= (_Float16)0) ? (_Float16)0 : y; break;
    case 5: __asm("fmin %h0, %h1, %h2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 7: __asm("fmax %h0, %h1, %h2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 10: z = x * y; break;
    case 11: z = z + x; break;
    case 12: z = z + y; break;
    }
    return z;
}
```

## Performance (M1 Max)

Note that a fused-multiply-add counts as _two_ floating-point operations. A measurement of 1.0 GFLOPS would mean 10<sup>9</sup> floating-point operations per second. The measurements are done without any load or store instructions; real-world workloads will need loads and stores, and thus will achieve lower numbers.

X and Y being `f16[32]`, each Z accumulator being `f16[32]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|45.4 GFLOPS|92.4 GFLOPS|105.2 GFLOPS|140.1 GFLOPS|177.4 GFLOPS|182.1 GFLOPS|
|2 per thread|90.0 GFLOPS|174.4 GFLOPS|210.3 GFLOPS|301.1 GFLOPS|384.2 GFLOPS|378.0 GFLOPS|
|3 per thread|135.2 GFLOPS|275.3 GFLOPS|311.2 GFLOPS|409.9 GFLOPS|500.4 GFLOPS|463.3 GFLOPS|
|4 per thread|183.7 GFLOPS|369.3 GFLOPS|417.2 GFLOPS|549.9 GFLOPS|595.2 GFLOPS|570.3 GFLOPS|
|5 per thread|230.2 GFLOPS|458.5 GFLOPS|434.6 GFLOPS|605.1 GFLOPS|678.5 GFLOPS|636.2 GFLOPS|
|6 per thread|272.3 GFLOPS|546.9 GFLOPS|522.3 GFLOPS|710.3 GFLOPS|743.6 GFLOPS|758.3 GFLOPS|
|7 per thread|321.9 GFLOPS|646.6 GFLOPS|579.7 GFLOPS|757.0 GFLOPS|777.5 GFLOPS|787.9 GFLOPS|
|8 per thread|369.5 GFLOPS|737.6 GFLOPS|667.2 GFLOPS|796.3 GFLOPS|792.8 GFLOPS|744.4 GFLOPS|
|9 per thread|339.9 GFLOPS|685.4 GFLOPS|625.4 GFLOPS|786.4 GFLOPS|803.6 GFLOPS|812.0 GFLOPS|
|10 per thread|362.0 GFLOPS|703.8 GFLOPS|642.9 GFLOPS|755.8 GFLOPS|742.5 GFLOPS|811.2 GFLOPS|
|11 per thread|361.1 GFLOPS|731.6 GFLOPS|704.4 GFLOPS|789.6 GFLOPS|787.2 GFLOPS|822.5 GFLOPS|
|12 per thread|359.3 GFLOPS|728.4 GFLOPS|649.4 GFLOPS|777.5 GFLOPS|785.2 GFLOPS|830.3 GFLOPS|
|13 per thread|370.4 GFLOPS|729.3 GFLOPS|655.7 GFLOPS|794.0 GFLOPS|816.7 GFLOPS|792.1 GFLOPS|
|14 per thread|365.9 GFLOPS|731.8 GFLOPS|649.5 GFLOPS|789.4 GFLOPS|782.8 GFLOPS|806.1 GFLOPS|
|15 per thread|364.8 GFLOPS|697.6 GFLOPS|657.2 GFLOPS|802.0 GFLOPS|776.0 GFLOPS|821.9 GFLOPS|
|16 per thread|369.1 GFLOPS|733.9 GFLOPS|662.7 GFLOPS|790.0 GFLOPS|789.8 GFLOPS|809.1 GFLOPS|

X and Y being `f16[32]`, each Z accumulator being `f32[2][16]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|44.6 GFLOPS|91.9 GFLOPS|95.6 GFLOPS|139.6 GFLOPS|171.6 GFLOPS|189.6 GFLOPS|
|2 per thread|91.4 GFLOPS|183.1 GFLOPS|226.3 GFLOPS|242.7 GFLOPS|273.5 GFLOPS|256.7 GFLOPS|
|3 per thread|137.9 GFLOPS|276.6 GFLOPS|285.3 GFLOPS|391.2 GFLOPS|461.9 GFLOPS|407.9 GFLOPS|
|4 per thread|184.1 GFLOPS|367.3 GFLOPS|403.4 GFLOPS|448.1 GFLOPS|470.6 GFLOPS|460.9 GFLOPS|
|5 per thread|171.7 GFLOPS|345.7 GFLOPS|365.0 GFLOPS|406.9 GFLOPS|461.2 GFLOPS|423.5 GFLOPS|
|6 per thread|184.6 GFLOPS|369.3 GFLOPS|406.4 GFLOPS|433.2 GFLOPS|465.7 GFLOPS|456.0 GFLOPS|
|7 per thread|182.0 GFLOPS|366.9 GFLOPS|377.2 GFLOPS|420.8 GFLOPS|463.3 GFLOPS|434.6 GFLOPS|
|8 per thread|178.9 GFLOPS|363.8 GFLOPS|389.1 GFLOPS|436.8 GFLOPS|474.7 GFLOPS|449.5 GFLOPS|
|9 per thread|185.1 GFLOPS|369.9 GFLOPS|376.3 GFLOPS|442.7 GFLOPS|465.5 GFLOPS|424.9 GFLOPS|
|10 per thread|178.1 GFLOPS|365.0 GFLOPS|352.8 GFLOPS|418.3 GFLOPS|445.1 GFLOPS|429.2 GFLOPS|
|11 per thread|182.2 GFLOPS|362.9 GFLOPS|417.9 GFLOPS|435.3 GFLOPS|457.6 GFLOPS|455.3 GFLOPS|
|12 per thread|179.0 GFLOPS|362.1 GFLOPS|395.5 GFLOPS|442.8 GFLOPS|452.1 GFLOPS|440.0 GFLOPS|
|13 per thread|184.2 GFLOPS|368.3 GFLOPS|368.7 GFLOPS|433.7 GFLOPS|465.8 GFLOPS|434.9 GFLOPS|
|14 per thread|184.6 GFLOPS|369.9 GFLOPS|391.5 GFLOPS|445.7 GFLOPS|477.9 GFLOPS|459.5 GFLOPS|
|15 per thread|182.7 GFLOPS|368.3 GFLOPS|382.6 GFLOPS|436.2 GFLOPS|470.8 GFLOPS|442.0 GFLOPS|
|16 per thread|182.0 GFLOPS|369.2 GFLOPS|391.2 GFLOPS|435.9 GFLOPS|460.0 GFLOPS|449.9 GFLOPS|

X and Y being `f32[16]`, each Z accumulator being `f32[16]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|22.8 GFLOPS|46.2 GFLOPS|52.5 GFLOPS|72.2 GFLOPS|85.5 GFLOPS|101.1 GFLOPS|
|2 per thread|45.9 GFLOPS|91.5 GFLOPS|99.8 GFLOPS|151.0 GFLOPS|176.2 GFLOPS|193.5 GFLOPS|
|3 per thread|68.8 GFLOPS|137.7 GFLOPS|147.9 GFLOPS|198.9 GFLOPS|226.3 GFLOPS|232.9 GFLOPS|
|4 per thread|91.6 GFLOPS|183.2 GFLOPS|192.7 GFLOPS|268.0 GFLOPS|295.6 GFLOPS|285.9 GFLOPS|
|5 per thread|113.9 GFLOPS|229.0 GFLOPS|235.0 GFLOPS|313.3 GFLOPS|358.5 GFLOPS|361.0 GFLOPS|
|6 per thread|138.4 GFLOPS|276.6 GFLOPS|320.7 GFLOPS|357.2 GFLOPS|380.4 GFLOPS|377.3 GFLOPS|
|7 per thread|156.8 GFLOPS|318.0 GFLOPS|299.3 GFLOPS|378.7 GFLOPS|387.0 GFLOPS|380.0 GFLOPS|
|8 per thread|184.6 GFLOPS|369.2 GFLOPS|330.5 GFLOPS|404.1 GFLOPS|424.1 GFLOPS|401.2 GFLOPS|
|9 per thread|172.3 GFLOPS|346.6 GFLOPS|318.7 GFLOPS|389.4 GFLOPS|382.3 GFLOPS|404.6 GFLOPS|
|10 per thread|180.5 GFLOPS|359.7 GFLOPS|327.1 GFLOPS|400.3 GFLOPS|391.0 GFLOPS|401.5 GFLOPS|
|11 per thread|176.7 GFLOPS|355.4 GFLOPS|323.6 GFLOPS|381.9 GFLOPS|380.1 GFLOPS|393.9 GFLOPS|
|12 per thread|181.7 GFLOPS|367.4 GFLOPS|330.8 GFLOPS|400.9 GFLOPS|388.6 GFLOPS|412.3 GFLOPS|
|13 per thread|180.7 GFLOPS|363.7 GFLOPS|340.4 GFLOPS|398.4 GFLOPS|386.9 GFLOPS|409.7 GFLOPS|
|14 per thread|183.2 GFLOPS|367.5 GFLOPS|334.0 GFLOPS|397.1 GFLOPS|414.1 GFLOPS|410.0 GFLOPS|
|15 per thread|185.1 GFLOPS|369.0 GFLOPS|332.6 GFLOPS|367.8 GFLOPS|395.1 GFLOPS|416.4 GFLOPS|
|16 per thread|184.5 GFLOPS|368.5 GFLOPS|333.3 GFLOPS|398.5 GFLOPS|395.9 GFLOPS|411.7 GFLOPS|

X and Y being `f64[8]`, each Z accumulator being `f64[8]`, ALU operation being `z + x*y` or `z - x*y`:

|Z Accumulators|1 Thread|2 Threads|3 Threads|4 Threads|5 Threads|6 Threads|
|---:|---:|---:|---:|---:|---:|---:|
|1 per thread|11.5 GFLOPS|23.1 GFLOPS|26.5 GFLOPS|34.5 GFLOPS|50.2 GFLOPS|49.3 GFLOPS|
|2 per thread|23.1 GFLOPS|46.2 GFLOPS|54.1 GFLOPS|71.2 GFLOPS|93.8 GFLOPS|97.8 GFLOPS|
|3 per thread|34.6 GFLOPS|69.3 GFLOPS|85.3 GFLOPS|104.0 GFLOPS|121.7 GFLOPS|118.4 GFLOPS|
|4 per thread|46.2 GFLOPS|92.3 GFLOPS|116.1 GFLOPS|137.0 GFLOPS|163.0 GFLOPS|146.3 GFLOPS|
|5 per thread|57.7 GFLOPS|115.5 GFLOPS|127.1 GFLOPS|157.4 GFLOPS|176.2 GFLOPS|171.2 GFLOPS|
|6 per thread|68.8 GFLOPS|138.5 GFLOPS|133.5 GFLOPS|178.3 GFLOPS|189.1 GFLOPS|185.1 GFLOPS|
|7 per thread|80.7 GFLOPS|161.8 GFLOPS|150.3 GFLOPS|190.6 GFLOPS|193.9 GFLOPS|200.1 GFLOPS|
|8 per thread|92.4 GFLOPS|184.9 GFLOPS|166.9 GFLOPS|200.4 GFLOPS|203.5 GFLOPS|210.6 GFLOPS|
|9 per thread|85.2 GFLOPS|171.2 GFLOPS|158.2 GFLOPS|194.7 GFLOPS|199.1 GFLOPS|202.7 GFLOPS|
|10 per thread|91.1 GFLOPS|182.2 GFLOPS|162.5 GFLOPS|194.0 GFLOPS|196.1 GFLOPS|203.5 GFLOPS|
|11 per thread|91.7 GFLOPS|182.9 GFLOPS|164.8 GFLOPS|200.6 GFLOPS|195.7 GFLOPS|193.7 GFLOPS|
|12 per thread|91.7 GFLOPS|184.5 GFLOPS|165.8 GFLOPS|198.8 GFLOPS|198.0 GFLOPS|205.5 GFLOPS|
|13 per thread|92.4 GFLOPS|184.6 GFLOPS|166.7 GFLOPS|201.5 GFLOPS|204.2 GFLOPS|206.4 GFLOPS|
|14 per thread|92.7 GFLOPS|184.9 GFLOPS|165.4 GFLOPS|197.8 GFLOPS|198.0 GFLOPS|209.4 GFLOPS|
|15 per thread|92.0 GFLOPS|184.4 GFLOPS|167.2 GFLOPS|201.2 GFLOPS|212.0 GFLOPS|195.4 GFLOPS|
|16 per thread|92.3 GFLOPS|184.7 GFLOPS|166.4 GFLOPS|199.8 GFLOPS|198.8 GFLOPS|203.2 GFLOPS|
