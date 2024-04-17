#include "emulate.h"

#define FMA32_X_FP16 (1ull << 61)
#define FMA32_Y_FP16 (1ull << 60)

static double fma64_alu(double x, double y, double z, uint64_t operand) {
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

static float fma32_alu(float x, float y, float z, uint64_t operand) {
    switch ((operand >> 27) & 7) {
    case 1: return x * y;
    case 2: return z + x;
    case 3: return x;
    case 4: return z + y;
    case 5: return y;
    case 6: return z;
    case 7: return 0.f;
    }
    float result;
    __asm("fmadd %s0, %s1, %s2, %s3" : "=w"(result) : "w"(x), "w"(y), "w"(z));
    return result;
}

static _Float16 fma16_alu(_Float16 x, _Float16 y, _Float16 z, uint64_t operand) {
    switch ((operand >> 27) & 7) {
    case 1: return x * y;
    case 2: return z + x;
    case 3: return x;
    case 4: return z + y;
    case 5: return y;
    case 6: return z;
    case 7: return 0.f;
    }
    _Float16 result;
    __asm("fmadd %h0, %h1, %h2, %h3" : "=w"(result) : "w"(x), "w"(y), "w"(z));
    return result;
}

static void load_xy_reg_fma32(float* dst, const void* src, uint64_t offset, uint64_t fp16) {
    load_xy_reg(dst, src, offset);
    if (fp16) {
        for (uint32_t i = 0; i < 16; ++i) {
            float val = dst[i];
            __asm("fcvt %s0, %h0" : "=w"(val) : "0"(val));
            dst[i] = val;
        }
    }
}

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

void emulate_AMX_FMA32(amx_state* state, uint64_t operand) {
    uint64_t y_offset = operand & 0x1FF;
    uint64_t x_offset = (operand >> 10) & 0x1FF;
    uint64_t z_row = (operand >> 20) & 63;
    uint64_t x_enable = parse_writemask(operand >> 41, 4, 7);
    uint64_t y_enable = parse_writemask(operand >> 32, 4, 7);

    float x[16];
    float y[16];
    load_xy_reg_fma32(x, state->x, x_offset, operand & FMA32_X_FP16);
    load_xy_reg_fma32(y, state->y, y_offset, operand & FMA32_Y_FP16);

    for (int i = 0; i < 16; i++) {
        if (!((x_enable >> (i * 4)) & 1)) continue;
        if (operand & FMA_VECTOR_PRODUCT) {
            float* z = &state->z[z_row].f32[i];
            *z = fma32_alu(x[i], y[i], *z, operand);
        } else {
            for (int j = 0; j < 16; j++) {
                if (!((y_enable >> (j * 4)) & 1)) continue;
                float* z = &state->z[(j * 4) + (z_row & 3)].f32[i];
                *z = fma32_alu(x[i], y[j], *z, operand);
            }
        }
    }
}

void emulate_AMX_FMA16(amx_state* state, uint64_t operand) {
    uint64_t y_offset = operand & 0x1FF;
    uint64_t x_offset = (operand >> 10) & 0x1FF;
    uint64_t z_row = (operand >> 20) & 63;
    uint64_t x_enable = parse_writemask(operand >> 41, 2, 7);
    uint64_t y_enable = parse_writemask(operand >> 32, 2, 7);

    _Float16 x[32];
    _Float16 y[32];
    load_xy_reg(x, state->x, x_offset);
    load_xy_reg(y, state->y, y_offset);

    for (int i = 0; i < 32; i++) {
        if (!((x_enable >> (i * 2)) & 1)) continue;
        if (operand & FMA_VECTOR_PRODUCT) {
            _Float16* z = &state->z[z_row].f16[i];
            *z = fma16_alu(x[i], y[i], *z, operand);
        } else {
            for (int j = 0; j < 32; j++) {
                if (!((y_enable >> (j * 2)) & 1)) continue;
                if (operand & FMA_WIDEN_16_32) {
                    float* z = &state->z[(j * 2) + (i & 1)].f32[i >> 1];
                    *z = fma32_alu(x[i], y[j], *z, operand);
                } else {
                    _Float16* z = &state->z[(j * 2) + (z_row & 1)].f16[i];
                    *z = fma16_alu(x[i], y[j], *z, operand);
                }
            }
        }
    }
}
