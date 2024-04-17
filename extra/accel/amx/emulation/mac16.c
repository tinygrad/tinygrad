#include "emulate.h"

#define MAC_Y_INT8 (1ull << 60)
#define MAC_X_INT8 (1ull << 61)

#define MAC_SKIP_X_INPUT (1ull << 29)
#define MAC_SKIP_Y_INPUT (1ull << 28)
#define MAC_SKIP_Z_INPUT (1ull << 27)

static int64_t mac32_alu(int64_t x, int64_t y, int64_t z, uint64_t operand) {
    if (operand & MAC_X_INT8) x = (int8_t)x;
    if (operand & MAC_Y_INT8) y = (int8_t)y;
    int64_t val;
    switch ((operand >> 28) & 3) {
    default: val = x * y; break;
    case  1: val = x; break;
    case  2: val = y; break;
    case  3: val = 0; break;
    }
    uint32_t shift = (operand >> 55) & 0x1f;
    val >>= shift;
    if (!(operand & MAC_SKIP_Z_INPUT)) {
        val += z;
    }
    return val;
}

void emulate_AMX_MAC16(amx_state* state, uint64_t operand) {
    uint64_t y_offset = operand & 0x1FF;
    uint64_t x_offset = (operand >> 10) & 0x1FF;
    uint64_t z_row = (operand >> 20) & 63;
    uint64_t x_enable = parse_writemask(operand >> 41, 2, 7);
    uint64_t y_enable = parse_writemask(operand >> 32, 2, 7);

    int16_t x[32];
    int16_t y[32];
    load_xy_reg(x, state->x, x_offset);
    load_xy_reg(y, state->y, y_offset);

    for (int i = 0; i < 32; i++) {
        if (!((x_enable >> (i * 2)) & 1)) continue;
        if (operand & FMA_VECTOR_PRODUCT) {
            int16_t* z = &state->z[z_row].i16[i];
            *z = mac32_alu(x[i], y[i], *z, operand);
        } else {
            for (int j = 0; j < 32; j++) {
                if (!((y_enable >> (j * 2)) & 1)) continue;
                if (operand & FMA_WIDEN_16_32) {
                    int32_t* z = &state->z[(j * 2) + (i & 1)].i32[i >> 1];
                    *z = mac32_alu(x[i], y[j], *z, operand);
                } else {
                    int16_t* z = &state->z[(j * 2) + (z_row & 1)].i16[i];
                    *z = mac32_alu(x[i], y[j], *z, operand);
                }
            }
        }
    }
}
