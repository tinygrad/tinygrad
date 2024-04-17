#include "emulate.h"

#define MATFP_INDEXED_LOAD (1ull << 53)
#define MATFP_INDEXED_LOAD_Y (1ull << 47)
#define MATFP_INDEXED_LOAD_4BIT (1ull << 48)

float bf16_to_f32(uint32_t x);
_Float16 vecfp_alu16(_Float16 x, _Float16 y, _Float16 z, int alumode);
uint16_t vecfp_alu_bf16(uint16_t x, uint16_t y, uint16_t z, int alumode);
float vecfp_alu32(float x, float y, float z, int alumode);
double vecfp_alu64(double x, double y, double z, int alumode);

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
            for (uint32_t j = 0; j < 32; j += 1) {
                if (!((y_enable >> (j*xybytes)) & 1)) continue;
                for (uint32_t i = 0; i < 32; i += 1) {
                    if (!((x_enable >> (i*xybytes)) & 1)) continue;
                    uint16_t* z = &state->z[bit_select(j*2, z_row, 1)].u16[i];
                    *z = omask ? vecfp_alu_bf16(x.u16[i], y.u16[j], *z, alumode) : 0;
                }
            }
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
    } else if (zbits == 32 && xybits == 16) {
        for (uint32_t j = 0; j < 32; j += 1) {
            if (!((y_enable >> (j*xybytes)) & 1)) continue;
            for (uint32_t i = 0; i < 32; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                float* z = &state->z[bit_select(j*2, i, 1)].f32[i >> 1];
                float xf = bf16 ? bf16_to_f32(x.u16[i]) : x.f16[i];
                float yf = bf16 ? bf16_to_f32(y.u16[j]) : y.f16[j];
                *z = omask ? vecfp_alu32(xf, yf, *z, alumode) : 0;
            }
        }
    } else if (zbits == 32 && xybits == 32) {
        for (uint32_t j = 0; j < 16; j += 1) {
            if (!((y_enable >> (j*xybytes)) & 1)) continue;
            for (uint32_t i = 0; i < 16; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                float* z = &state->z[bit_select(j*4, z_row, 3)].f32[i];
                *z = omask ? vecfp_alu32(x.f32[i], y.f32[j], *z, alumode) : 0;
            }
        }
    } else {
        for (uint32_t j = 0; j < 8; j += 1) {
            if (!((y_enable >> (j*xybytes)) & 1)) continue;
            for (uint32_t i = 0; i < 8; i += 1) {
                if (!((x_enable >> (i*xybytes)) & 1)) continue;
                double* z = &state->z[bit_select(j*8, z_row, 7)].f64[i];
                *z = omask ? vecfp_alu64(x.f64[i], y.f64[j], *z, alumode) : 0;
            }
        }
    }
}
