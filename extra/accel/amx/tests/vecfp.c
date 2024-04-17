#include "emulate.h"

#define VECFP_INDEXED_LOAD (1ull << 53)
#define VECFP_INDEXED_LOAD_Y (1ull << 47)
#define VECFP_INDEXED_LOAD_4BIT (1ull << 48)

#define bf16_isnan(x) (((x) & 0x7fff) > 0x7f80)

float bf16_to_f32(uint32_t x) {
    if (bf16_isnan(x)) x = BF16_NAN;
    x <<= 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

uint16_t bf16_fma(uint16_t x, uint16_t y, uint16_t z) { // Compute x * y + z
    // Unpack inputs
#define bf16_unpack(var) \
        int32_t var##_e = (var >> 7) & 255; /* (biased) exponent */ \
        uint32_t var##_m = (var & 127) ^ 128; /* mantissa (including implicit bit) */ \
        if (var##_e == 0) { /* convert denormal to something closer to normal */ \
            var##_e = 25 - __builtin_clz(var##_m ^= 128); \
            var##_m <<= 1 - var##_e; \
        }
    bf16_unpack(x)
    bf16_unpack(y)
    bf16_unpack(z)
#undef bf16_unpack
    uint16_t z_sign = z & 0x8000;
    uint16_t p_sign = (x ^ y) & 0x8000;

    // Handle NaN or Inf input
    if (x_e == 255 || y_e == 255 || z_e == 255) {
        if ((x_e == 255 && (x_m != 128 || y_m == 0)) // x NaN or x Inf times y zero
        ||  (y_e == 255 && (y_m != 128 || x_m == 0)) // y NaN or y Inf times x zero
        ||  (z_e == 255 && z_m != 128) // z NaN
        ||  (z_e == 255 && (x_e == 255 || y_e == 255) && (z_sign != p_sign))) { // z Inf and (x * y) Inf and signs differ
            return BF16_NAN; // NaN output
        } else if (z_e == 255) { // z Inf
            return z; // Inf output
        } else { // (x * y) Inf
            return p_sign | 0x7f80; // Inf output
        }
    }

    // p = x * y
    uint32_t p_m = x_m * y_m;
    int32_t p_e = x_e + y_e - 7 - 127;

    // r = z + p
    if (p_m == 0) return z & (z_m ? z : p_sign);
    z_m <<= 7, z_e -= 7; // Give z_m similar precision to p_m
    p_m <<= 3, z_m <<= 3; // Three extra bits of precision (for rounding etc.)
#define sticky_shift(var, amount) \
    do { \
        int32_t s = amount; \
        if (s >= 32) { \
            var = (var != 0); \
        } else { \
            uint32_t orig = var; \
            var >>= s; \
            var |= ((var << s) != orig); \
        } \
    } while(0)
    int32_t r_e = p_e > z_e ? p_e : z_e;
    if (p_e < r_e) sticky_shift(p_m, r_e - p_e); // Discard low bits from p_m
    if (z_e < r_e) sticky_shift(z_m, r_e - z_e); // Discard low bits from z_m
    uint16_t r_sign = p_m >= z_m ? p_sign : z_sign;
    if (z_sign != r_sign) z_m = ~z_m;
    if (p_sign != r_sign) p_m = ~p_m;
    uint32_t r_m = z_m + p_m + (p_sign != z_sign);
    if (r_m == 0) return z_sign & p_sign;

    // Normalise result to 21 zero bits, 1 one bit, 10 fractional bits
    int32_t n = 21 - __builtin_clz(r_m);
    r_e += n;
    if (n < 0) r_m <<= -n; else sticky_shift(r_m, n);
    if (r_e >= 255) return r_sign | 0x7f80; // Inf
    if (r_e <= 0) { // Denorm or zero
        sticky_shift(r_m, 1 - r_e);
        r_e = 0;
    }
#undef sticky_shift
    uint16_t r = r_sign | (r_e << 7) | ((r_m >> 3) & 0x7f);

    // Round result
    return r + (((r_m & 7) + (r & 1)) > 4);
}

#define ord(x) ((int16_t)x ^ ((((int16_t)x) >> 15) & 0x7fff))
static uint16_t bf16_min(uint16_t x, uint16_t z) {
    if (bf16_isnan(x) || bf16_isnan(z)) return BF16_NAN;
    return ord(x) < ord(z) ? x : z;
}

static uint16_t bf16_max(uint16_t x, uint16_t z) {
    if (bf16_isnan(x) || bf16_isnan(z)) return BF16_NAN;
    return ord(x) > ord(z) ? x : z;
}
#undef ord

uint16_t vecfp_alu_bf16(uint16_t x, uint16_t y, uint16_t z, int alumode) {
    switch (alumode) {
    case 0: z = bf16_fma(x, y, z); break;
    case 1: z = bf16_fma(x ^ 0x8000, y, z); break;
    case 4: z = (x == 0 || (int16_t)x <= -128) ? 0 : y; break;
    case 5: z = bf16_min(x, z); break;
    case 7: z = bf16_max(x, z); break;
    case 10: z = bf16_fma(x, y, 0x8000); break;
    case 11: z = bf16_fma(x, BF16_ONE, z); break;
    case 12: z = bf16_fma(BF16_ONE, y, z); break;
    }
    return z;
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

float vecfp_alu32(float x, float y, float z, int alumode) {
    switch (alumode) {
    case 0: __asm("fmadd %s0, %s1, %s2, %s3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 1: __asm("fmsub %s0, %s1, %s2, %s3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 4: z = (x <= 0.f) ? 0.f : y; break;
    case 5: __asm("fmin %s0, %s1, %s2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 7: __asm("fmax %s0, %s1, %s2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 10: z = x * y; break;
    case 11: z = z + x; break;
    case 12: z = z + y; break;
    }
    return z;
}

double vecfp_alu64(double x, double y, double z, int alumode) {
    switch (alumode) {
    case 0: __asm("fmadd %d0, %d1, %d2, %d3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 1: __asm("fmsub %d0, %d1, %d2, %d3" : "=w"(z) : "w"(x), "w"(y), "w"(z)); break;
    case 4: z = (x <= 0.) ? 0. : y; break;
    case 5: __asm("fmin %d0, %d1, %d2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 7: __asm("fmax %d0, %d1, %d2" : "=w"(z) : "w"(x), "w"(z)); break;
    case 10: z = x * y; break;
    case 11: z = z + x; break;
    case 12: z = z + y; break;
    }
    return z;
}

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
                for (uint32_t i = 0; i < 32; i += 1) {
                    if (!((x_enable >> (i*xybytes)) & 1)) continue;
                    uint32_t j = broadcast_y ? ((operand >> 32) & 0x1f) : i;
                    uint16_t* z = &state->z[z_row].u16[i];
                    *z = omask ? vecfp_alu_bf16(x.u16[i & ximask], y.u16[j], *z, alumode) : 0;
                }
            } else {
                for (uint32_t i = 0; i < 32; i += 1) {
                    if (!((x_enable >> (i*xybytes)) & 1)) continue;
                    uint32_t j = broadcast_y ? ((operand >> 32) & 0x1f) : i;
                    _Float16* z = &state->z[z_row].f16[i];
                    *z = omask ? vecfp_alu16(x.f16[i & ximask], y.f16[j], *z, alumode) : 0;
                }
            }
        } else if (zbits == 32 && xybits == 16) {
            for (uint32_t i = 0; i < 32; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                uint32_t j = broadcast_y ? ((operand >> 32) & 0x1f) : i;
                float* z = &state->z[bit_select(z_row, i, 1)].f32[i >> 1];
                float xf = bf16 ? bf16_to_f32(x.u16[i & ximask]) : x.f16[i & ximask];
                float yf = bf16 ? bf16_to_f32(y.u16[j]) : y.f16[j];
                *z = omask ? vecfp_alu32(xf, yf, *z, alumode) : 0;
            }
        } else if (zbits == 32 && xybits == 32) {
            for (uint32_t i = 0; i < 16; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                uint32_t j = broadcast_y ? ((operand >> 32) & 0xf) : i;
                float* z = &state->z[z_row].f32[i];
                *z = omask ? vecfp_alu32(x.f32[i & ximask], y.f32[j], *z, alumode) : 0;
            }
        } else {
            for (uint32_t i = 0; i < 8; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                uint32_t j = broadcast_y ? ((operand >> 32) & 0x7) : i;
                double* z = &state->z[z_row].f64[i];
                *z = omask ? vecfp_alu64(x.f64[i & ximask], y.f64[j], *z, alumode) : 0;
            }
        }
    }
}
