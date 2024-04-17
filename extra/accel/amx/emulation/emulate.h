#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef union amx_reg {
    uint8_t  u8 [64];
    uint16_t u16[32];
    uint32_t u32[16];
    int8_t   i8 [64];
    int16_t  i16[32];
    int32_t  i32[16];
    _Float16 f16[32];
    float    f32[16];
    double   f64[ 8];
} amx_reg;

typedef __attribute__((aligned(128))) struct amx_state {
    amx_reg x[ 8];
    amx_reg y[ 8];
    amx_reg z[64];
} amx_state;

extern uint32_t AMX_VER;
#define AMX_VER_M1 1
#define AMX_VER_M2 2
#define AMX_VER_M3 3

// Common helpers:

static inline int64_t load_int(const void* src, uint32_t nbytes, uint32_t signext) {
    int64_t val = 0;
    memcpy(&val, src, nbytes);
    return (val << signext) >> signext;
}

static inline void store_int(void* dst, uint32_t nbytes, int64_t val) {
    memcpy(dst, &val, nbytes);
}

static inline void load_xy_reg(void* dst, const void* src, uint64_t offset) {
    uint64_t avail = 512 - offset;
    memcpy(dst, ((const uint8_t*)src) + offset, avail >= 64 ? 64 :      avail);
    memcpy(((uint8_t*)dst) + avail, src       , avail >= 64 ?  0 : 64 - avail);
}

void load_xy_reg_indexed(uint8_t* dst, const uint8_t* table, uint32_t ibits, uint32_t ebits);

static inline void xy_shuffle(uint8_t* dst, uint32_t shuffle, uint32_t ebytes) {
    if (shuffle != 0) {
        uint8_t src[64];
        memcpy(src, dst, 64); // As we modify dst in-place
        uint32_t step = 64 >> shuffle;
        for (uint32_t doff = 0, soff = 0; doff < 64; doff += ebytes) {
            memcpy(dst + doff, src + soff, ebytes);
            soff += step;
            if (soff & 64) {
                soff += ebytes;
            }
            soff &= 63;
        }
    }
}

static inline uint64_t parse_writemask(uint32_t val, uint32_t g, uint32_t maskbits) {
    uint32_t mode = (maskbits >= 9) ? (val >> 6) & 7 : (val >> 5) & 3;
    if (mode != 0) val *= g;
    val &= 0x3F;
    uint64_t all = ~0ull;
    switch (mode) {
    case 0:
        if (val == 1 || val == 2) {
            // odd/even groups of g
            all = ~(all << g) << (g & -(val & 1));
            while ((g <<= 1) < 64) {
                all |= all << g;
            }
        }
        // sometimes additional meanings:
        // 3: x,y,z are all zero (i.e. write zeros)
        // 4: x is zero
        // 5: y is zero
        return (val < (maskbits >= 9 ? 6 : 3)) ? all : 0;
    case 1:
        // sometimes instead means broadcasting from n'th
        return (~(all << g)) << val; // only n'th group of g
    case 2:
        if (val == 0) return all;
        // fallthrough
    case 4:
        return ~(all << val); // first n groups of g
    case 3:
        if (val == 0) return all;
        // fallthrough
    case 5:
        return ~(all >> val); // last n groups of g
    default:
        return 0; // nothing enabled
    }
}

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#define bit_select(hi, lo, mask) (((hi) & ~(mask)) | ((lo) & (mask)))

#define FMA_WIDEN_16_32 (1ull << 62)
#define FMA_VECTOR_PRODUCT (1ull << 63)

#define BF16_ONE 0x3f80
#define BF16_NAN 0x7fc0
