#include "emulate.h"

#define EXTR_SIGNED_INPUT (1ull << 57)
#define EXTR_SIGNED_OUTPUT (1ull << 56)
#define EXTR_SATURATE (1ull << 55)
#define EXTR_ROUNDING_SHIFT (1ull << 54)
#define EXTR_BETWEEN_XY (1ull << 27)
#define EXTR_HV (1ull << 26)
#define EXTR_HV_TO_Y (1ull << 10)

static uint32_t bf16_from_f32(uint32_t f) {
    if ((f & 0x7fffffff) > 0x7f800000) return BF16_NAN; // Any NaN -> Default NaN
    uint32_t low = f & 0xffff;
    f >>= 16;
    return f + (low + (f & 1) > 0x8000);
}

static int64_t extr_alu(int64_t val, uint64_t operand, uint32_t outbits) {
    uint32_t shift = (operand >> 58) & 0x1f;
    if (operand & (1ull << 63)) {
        if (shift >= 16) {
            val = bf16_from_f32((uint32_t)val);
        } else {
            __asm("fcvt %h0, %s0" : "=w"(val) : "0"(val));
        }
        return val;
    }
    if (shift && (operand & EXTR_ROUNDING_SHIFT)) {
        val += 1 << (shift - 1);
    }
    val >>= shift;
    if (operand & EXTR_SATURATE) {
        if (operand & EXTR_SIGNED_OUTPUT) outbits -= 1;
        int64_t hi = 1ull << outbits;
        if (operand & EXTR_SIGNED_INPUT) {
            int64_t lo = (operand & EXTR_SIGNED_OUTPUT) ? -hi : 0;
            if (val < lo) val = lo;
            if (val >= hi) val = hi - 1;
        } else {
            if ((uint64_t)val >= (uint64_t)hi) val = hi - 1;
        }
    }
    return val;
}

static void store_xy_row(void* dst, uint64_t offset, const void* src, uint64_t write_mask) {
    for (uint64_t i = 0; i < 64; ++i) {
        if (!((write_mask >> i) & 1)) continue;
        ((uint8_t*)dst)[(offset + i) & 0x1ff] = ((const uint8_t*)src)[i];
    }
}

void emulate_AMX_EXTRX(amx_state* state, uint64_t operand) {
    void* dst;
    uint64_t dst_offset;
    uint64_t z_row = operand >> 20;
    uint64_t z_step = 64;
    uint64_t store_enable = ~(uint64_t)0;
    uint8_t buffer[64];
    uint32_t stride = 0;
    uint32_t zbytes, xybytes;

    if (operand & EXTR_HV) {
        dst = (operand & EXTR_HV_TO_Y) ? state->y : state->x;
        dst_offset = operand;
        switch (((operand >> 63) << 4) | ((operand >> 11) & 0xF)) {
        case  0: xybytes = 1; zbytes = 1; break;
        case  8: xybytes = 4; zbytes = 4; break;
        case  9: xybytes = 2; zbytes = 4; stride = 1; break;
        case 10: xybytes = 2; zbytes = 4; stride = 2; break;
        case 11: xybytes = 1; zbytes = 4; stride = 1; break;
        case 13: xybytes = 1; zbytes = 2; stride = 1; break;
        case 17: xybytes = 8; zbytes = 8; break;
        case 24: xybytes = 4; zbytes = 4; break;
        case 25: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 1; } else { zbytes = 2; } break;
        case 26: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 2; } else { zbytes = 2; } break;
        default: xybytes = 2; zbytes = 2; break;
        }
        if ((AMX_VER >= AMX_VER_M2) && (operand & (1ull << 31))) {
            operand &=~ (0x1ffull << 32);
            z_step = z_row & 32 ? 16 : 32;
        }
        store_enable &= parse_writemask(operand >> 32, xybytes, 9);
    } else if (operand & EXTR_BETWEEN_XY) {
        memcpy(state->x + ((operand >> 16) & 7),
               state->y + (          z_row & 7), 64);
        return;
    } else {
        dst = state->x;
        dst_offset = operand >> 10;
        xybytes = 8 >> ((operand >> 28) & 3);
        if (xybytes == 1) {
            xybytes = 2;
            store_enable &= 0x5555555555555555ull;
        }
        store_enable &= parse_writemask(operand >> 41, xybytes, 7);
        zbytes = xybytes;
    }

    uint32_t signext = (operand & EXTR_SIGNED_INPUT) ? 64 - zbytes*8 : 0;
    for (z_row &= z_step - 1; z_row <= 63; z_row += z_step) {
        for (uint32_t i = 0; i < 64; i += xybytes) {
            uint64_t zoff = (i & (zbytes - 1)) / xybytes * stride;
            int64_t val = load_int(&state->z[bit_select(z_row, z_row + zoff, zbytes - 1)].u8[i & -zbytes], zbytes, signext);
            if (stride) val = extr_alu(val, operand, xybytes*8);
            store_int(buffer + i, xybytes, val);
        }
        if ((operand & EXTR_HV) && (((operand >> 32) & 0x1ff) == 3)) {
            memset(buffer, 0, sizeof(buffer));
        }
        store_xy_row(dst, dst_offset & 0x1FF, buffer, store_enable);
        dst_offset += 64;
    }
}

void emulate_AMX_EXTRY(amx_state* state, uint64_t operand) {
    void* dst;
    uint64_t dst_offset = operand;
    uint64_t z_col = operand >> 20;
    uint64_t z_step = 64;
    uint64_t store_enable = ~(uint64_t)0;
    uint8_t buffer[64];
    uint32_t stride = 0;
    uint32_t zbytes, xybytes;

    if (operand & EXTR_HV) {
        dst = (operand & EXTR_HV_TO_Y) ? state->y : state->x;
        switch (((operand >> 63) << 4) | ((operand >> 11) & 0xF)) {
        case  0: xybytes = 1; zbytes = 1; break;
        case  8: xybytes = 4; zbytes = 4; break;
        case  9: xybytes = 2; zbytes = 4; stride = 1; break;
        case 10: xybytes = 2; zbytes = 4; stride = 2; break;
        case 11: xybytes = 1; zbytes = 4; stride = 1; break;
        case 13: xybytes = 1; zbytes = 2; stride = 1; break;
        case 17: xybytes = 8; zbytes = 8; break;
        case 24: xybytes = 4; zbytes = 4; break;
        case 25: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 1; } else { zbytes = 2; } break;
        case 26: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 2; } else { zbytes = 2; } break;
        default: xybytes = 2; zbytes = 2; break;
        }
        if ((AMX_VER >= AMX_VER_M2) && (operand & (1ull << 31))) {
            operand &=~ (0x1ffull << 32);
            z_step = z_col & 32 ? 16 : 32;
        }
        store_enable &= parse_writemask(operand >> 32, xybytes, 9);
    } else if (operand & EXTR_BETWEEN_XY) {
        memcpy(state->y + ((operand >> 6) & 7),
               state->x + (z_col & 7), 64);
        return;
    } else {
        dst = state->y;
        xybytes = 8 >> ((operand >> 28) & 3);
        if (xybytes == 1) {
            xybytes = 2;
            store_enable &= 0x5555555555555555ull;
        }
        store_enable &= parse_writemask(operand >> 32, xybytes, 7);
        zbytes = xybytes;
    }

    uint32_t signext = (operand & EXTR_SIGNED_INPUT) ? 64 - zbytes*8 : 0;
    for (z_col &= z_step - 1; z_col <= 63; z_col += z_step) {
        for (uint32_t j = 0; j < 64; j += xybytes) {
            uint64_t zoff = (j & (zbytes - 1)) / xybytes * stride;
            int64_t val = load_int(&state->z[bit_select(j, z_col + zoff, zbytes - 1)].u8[z_col & -zbytes], zbytes, signext);
            if (stride) val = extr_alu(val, operand, xybytes*8);
            store_int(buffer + j, xybytes, val);
        }
        if (((operand >> 32) & 0x1ff) == 3) {
            memset(buffer, 0, sizeof(buffer));
        }
        store_xy_row(dst, dst_offset & 0x1FF, buffer, store_enable);
        dst_offset += 64;
    }
}
