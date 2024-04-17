## Quick summary

|Instruction|General theme|Notes|
|---|---|---|
|`genlut`&nbsp;(53≤6)|Generate indices for indexed load|For use by `matfp` / `matint` / `vecfp` / `vecint` / `genlut`&nbsp;(53≥7)|
|`genlut`&nbsp;(53≥7)|Perform indexed load|Can write to any of `x` or `y` or `z`|


## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `22`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|63|1|Ignored|
|60|3|Table register index|
|59|1|Table is from Y (`1`) or from X (`0`)|
|57|2|Ignored|
|53|4|Mode|Data type, lane width, index width, whether generating or looking up|
|31|22|Ignored|
|(53=1)&nbsp;30|1|Data type is bf16 (`1`) or f16 (`0`)|M2 only (always reads as `0` on M1)|
|27|3|Ignored|
|26|1|Destination is Z (`1`) or is X or Y (`0`)|Ignored in generate modes; they always write to X or Y| 
|(26=0) 25|1|Destination is Y (`1`) or is X (`0`)|
|(26=0) 23|2|Ignored|
|(26=1) 23|3|Destination register index (hi)|
|20|3|Destination register index (lo)|
|11|9|Ignored|
|10|1|Source is from Y (`1`) or from X (`0`)|
|9|1|Ignored|
|0|9|Source offset (in bytes)|

Mode bits:
|Data type (DT)|Index type (IT)|Lane count (LC)|Direction|53|Notes|
|---:|---:|---:|---|---:|---|
|f32|u4|16|Generate|`0`|
|bf16 or f16|u5|32|Generate|`1`|M2 required for bf16. See bit 30|
|f64|u4|8|Generate|`2`|High bit of index always generated as `0`|
|i32|u4|16|Generate|`3`|
|i16|u5|32|Generate|`4`|
|u32|u4|16|Generate|`5`|
|u16|u5|32|Generate|`6`|
|any 32-bit|u2|16|Lookup|`7`|
|any 16-bit|u2|32|Lookup|`8`|
|any 8-bit|u2|64|Lookup|`9`|
|any 64-bit|u4|8|Lookup|`10`|High bit of index ignored|
|any 32-bit|u4|16|Lookup|`11`|
|any 16-bit|u4|32|Lookup|`12`|
|any 8-bit|u4|64|Lookup|`13`|
|any 16-bit|u5|32|Lookup|`14`|
|any 8-bit|u5|64|Lookup|`15`|

Note that DT times LC is always 512 bits.

## Description

In _lookup_ modes, a densely-packed `IT[LC]` vector is read from the source (between 4 and 40 bytes, depending on `IT` and `LC`), each lane is expanded from `IT` to `DT` by treating it as a lane index into the table register (which has type `DT[LC]`), and then the resultant 64 bytes are written to an X or Y or Z register.

In _generate_ modes, a `DT[LC]` vector is read from the source (64 bytes), and a densely-packed `IT[LC]` vector is written to the low (between 4 and 40) bytes of an X or Y register, with the remaining bytes of the X or Y register cleared to zero. The index values are obtained by searching the `DT[LC]` table register: for each lane of the source, the minimum `v` is found such that `table[v] > source_lane`, and then the index value is `v - 1`. If no such `v` is found, the index value is `-1` (just as if `v` was `0`). Note that `-1` is represented as an unsigned integer with all bits set (except in the f64 case, where the high bit of the index is forced to zero). If `table` happens to be sorted in ascending order, this results in `v` when `table[v] <= source_lane < table[v+1]` and `-1` when `source_lane` is either less than or greater than all of the table elements.

By using _generate_ followed by _lookup_ (into a different table), unary functions can be approximated in a piecewise linear (or piecewise polynomial) fashion: the _generate_ determines which piece of the function we're in, and the _lookup_ fetches the relevant coefficients for that piece.

## Emulation code

See [genlut.c](genlut.c).

A representative sample is:
```c
void emulate_AMX_GENLUT(amx_state* state, uint64_t operand) {
    uint64_t mode = (operand >> 53) & 0xf;
    const amx_reg* source = (operand & GENLUT_SOURCE_Y) ? state->y : state->x;
    const amx_reg* table = (operand & GENLUT_TABLE_Y) ? state->y : state->x;
    table += (operand >> 60) & 7;
    amx_reg xy;
    load_xy_reg(&xy, source, operand & 0x1FF);
    uint32_t ibits, ebits;
    switch (mode) {
    case  0: ibits = 4; ebits = 32; break; // generate from f32
    case  1: ibits = 5; ebits = 16; break; // generate from f16 (or bf16 on M2)
    case  2: ibits = 4; ebits = 64; break; // generate from f64
    case  3: ibits = 4; ebits = 32; break; // generate from i32
    case  4: ibits = 5; ebits = 16; break; // generate from i16
    case  5: ibits = 4; ebits = 32; break; // generate from u32
    case  6: ibits = 5; ebits = 16; break; // generate from u16
    case  7: ibits = 2; ebits = 32; break; // lookup
    case  8: ibits = 2; ebits = 16; break; // lookup
    case  9: ibits = 2; ebits =  8; break; // lookup
    case 10: ibits = 4; ebits = 64; break; // lookup
    case 11: ibits = 4; ebits = 32; break; // lookup
    case 12: ibits = 4; ebits = 16; break; // lookup
    case 13: ibits = 4; ebits =  8; break; // lookup
    case 14: ibits = 5; ebits = 16; break; // lookup
    case 15: ibits = 5; ebits =  8; break; // lookup
    }
    if (mode <= 6) {
        uint8_t vs[32]; // 8 bits per element, subsequently packed to ibits per element
        find_first_greater_than(vs, mode, &xy, table, operand);
        pack_bits(xy.u8, vs, ibits, ebits);
        operand &=~ GENLUT_DEST_Z;
    } else {
        load_xy_reg_indexed(xy.u8, table->u8, ibits, ebits);
    }
    amx_reg* dest;
    uint64_t doff = (operand >> 20);
    if (operand & GENLUT_DEST_Z) {
        dest = state->z;
        doff &= 63;
    } else {
        dest = (operand & GENLUT_DEST_Y) ? state->y : state->x;
        doff &= 7;
    }
    memcpy(dest + doff, &xy, 64);
}

void find_first_greater_than(uint8_t* vs, uint32_t mode, const amx_reg* xy, const amx_reg* table, uint64_t operand) {
    switch (mode) {
    case 0:
        for (uint32_t i = 0; i < 16; ++i) {
            uint32_t v = 0;
            for (; v < 16; ++v) { if (table->f32[v] > xy->f32[i]) break; }
            vs[i] = v - 1;
        }
        break;
    ...
    }
}

void pack_bits(uint8_t* dst, const uint8_t* bytes, uint32_t ibits, uint32_t ebits) {
    uint8_t* end = dst + 64;
    uint64_t imask = ebits == 64 ? 7 : (1ull << ibits) - 1;
    for (uint32_t etotal = 0; etotal < 64; etotal += ebits) {
        uint64_t packed = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            packed |= (bytes[i] & imask) << (i * ibits);
        }
        memcpy(dst, &packed, 8);
        dst += ibits;
        bytes += 8;
    }
    memset(dst, 0, end - dst);
}

void load_xy_reg_indexed(uint8_t* dst, const uint8_t* table, uint32_t ibits, uint32_t ebits) {
    uint8_t tmp[40];
    memcpy(tmp, dst, 40); // As we modify dst in-place

    uint32_t ebytes = ebits / 8;
    uint32_t imask = (1u << ibits) - 1;
    for (uint32_t doff = 0, soff = 0; doff < 64; ) {
        uint64_t bits;
        memcpy(&bits, tmp + soff, 8);
        soff += ibits;
        for (int i = 0; i < 8; ++i) {
            uint32_t toff = ((bits & imask) * ebytes) & 0x3f; // NB: & 0x3f only comes into play when ibits==4 and ebits==64
            memcpy(dst + doff, table + toff, ebytes);
            bits >>= ibits;
            doff += ebytes;
        }
    }
}
```
