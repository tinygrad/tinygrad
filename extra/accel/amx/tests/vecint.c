#include "emulate.h"

#define VECINT_ROUNDING_SHIFT (1ull << 29)
#define VECINT_SATURATE (1ull << 30)
#define VECINT_SIGNED_OUTPUT (1ull << 26)
#define VECINT_SIGNED_Z (1ull << 63)

#define VECINT_INDEXED_LOAD (1ull << 53)
#define VECINT_INDEXED_LOAD_Y (1ull << 47)
#define VECINT_INDEXED_LOAD_4BIT (1ull << 48)

#define VECINT_SIGNED_X (1ull << 63)
#define VECINT_SIGNED_Y (1ull << 26)

int64_t vecint_alu_mode4(int64_t val, uint32_t satbits, uint64_t operand) {
    uint32_t shift = (operand >> 58) & 0x1f;
    if (shift && (operand & VECINT_ROUNDING_SHIFT)) {
        val += 1ull << (shift - 1);
    }
    val >>= shift;
    if (operand & VECINT_SATURATE) {
        if (operand & VECINT_SIGNED_OUTPUT) {
            satbits -= 1;
        }
        int64_t hi = 1ull << satbits;
        if (operand & VECINT_SIGNED_Z) {
            int64_t lo = (operand & VECINT_SIGNED_OUTPUT) ? -hi : 0;
            if (val < lo) val = lo;
            if (val >= hi) val = hi - 1;
        } else {
            if ((uint64_t)val >= (uint64_t)hi) val = hi - 1;
        }
    }
    return val;
}

int64_t vecint_alu(int64_t x, int64_t y, int64_t z, int alumode, uint32_t shift) {
    int64_t val = x * y;
    if (alumode == 5 || alumode == 6) {
        val += 1ull << (shift - 1);
    } else if (alumode == 2 || alumode == 3) {
        val = x + y;
    } else if (alumode == 9) {
        return z + __builtin_popcountll((~(x ^ y)) << shift);
    } else if (alumode == 11) {
        val = x;
    } else if (alumode == 12) {
        val = y;
    }
    val >>= shift;
    if (alumode == 1 || alumode == 3 || alumode == 6) {
        val = -val;
    }
    if (alumode != 10) {
        val += z;
    }
    if (alumode == 5 || alumode == 6) {
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
    }
    return val;
}

void emulate_AMX_VECINT(amx_state* state, uint64_t operand) {
    if ((operand >> 54) & 7) {
        return;
    }

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
    z_row &= z_step - 1;
    int32_t omask = (((operand >> 32) & 0x1ff) == 3) ? 0 : -1;
    bool broadcast_y = ((operand >> (32+6)) & 7) == 1;
    int alumode = (operand & VECINT_INDEXED_LOAD) ? 0 : (operand >> 47) & 0x3f;
    uint32_t shift = (operand >> 58) & 0x1f;

    uint32_t xbits = 0, ybits = 0, zbits, satbits;
    if (alumode == 4) {
        switch ((operand >> 42) & 0xf) {
        case  3: zbits = 32; satbits = 16; break;
        case  4: zbits = 32; satbits = 32; break;
        case  9: zbits =  8; satbits =  8; break;
        case 10: zbits = 32; satbits =  8; break;
        case 11: zbits = 16; satbits =  8; break;
        default: zbits = 16; satbits = 16; break;
        }
    } else if (alumode == 5 || alumode == 6) {
        xbits = 16; ybits = 16; zbits = 16;
        shift = 15;
    } else {
        switch ((operand >> 42) & 0xf) {
        case  3: xbits = 16; ybits = 16; zbits = 32; break;
        case 10: xbits =  8; ybits =  8; zbits = 32; break;
        case 11: xbits =  8; ybits =  8; zbits = 16; break;
        case 12: xbits =  8; ybits = 16; zbits = 32; break;
        case 13: xbits = 16; ybits =  8; zbits = 32; break;
        default: xbits = 16; ybits = 16; zbits = 16; break;
        }
    }
    uint32_t xbytes = xbits / 8;
    uint32_t ybytes = ybits / 8;
    uint32_t zbytes = zbits / 8;

    if (alumode == 4) {
        // z = f(z), where f is [rounding] shift followed by [saturate]
        // with various options for width and signedness
        uint32_t zsignext = (operand & VECINT_SIGNED_Z) ? (64 - zbits) : 0;
        uint64_t col_enable = parse_writemask(operand >> 32, zbytes, 9);
        if (broadcast_y) {
            col_enable = ~(uint64_t)0;
            // NB: There is no y input to the operation
        }
        for (; z_row <= 63; z_row += z_step) {
            for (uint32_t i = 0; i < 64; i += zbytes) {
                if (!((col_enable >> i) & 1)) continue;
                int64_t val = load_int(&state->z[z_row].u8[i], zbytes, zsignext);
                val = vecint_alu_mode4(val, satbits, operand);
                store_int(&state->z[z_row].u8[i], zbytes, val & omask);
            }
        }
        return;
    } else if ((AMX_VER >= AMX_VER_M2) && (alumode == 10 || alumode == 11 || alumode == 12)) {
    } else if (alumode >= 7) {
        return;
    }

    uint64_t x_offset = operand >> 10;
    uint64_t y_offset = operand;
    for (; z_row <= 63; z_row += z_step) {
        uint8_t x[64];
        uint8_t y[64];
        load_xy_reg(x, state->x, x_offset & 0x1FF); x_offset += x_step;
        load_xy_reg(y, state->y, y_offset & 0x1FF); y_offset += y_step;
        if (operand & VECINT_INDEXED_LOAD) {
            uint32_t src_reg = (operand >> 49) & 7;
            uint32_t ibits = (operand & VECINT_INDEXED_LOAD_4BIT) ? 4 : 2;
            if (operand & VECINT_INDEXED_LOAD_Y) {
                load_xy_reg_indexed(y, state->y[src_reg].u8, ibits, ybits);
                y_offset -= y_step - y_step * ibits / ybits;
            } else {
                load_xy_reg_indexed(x, state->x[src_reg].u8, ibits, xbits);
                x_offset -= x_step - x_step * ibits / xbits;
            }
        }
        xy_shuffle(x, (operand >> 29) & 3, xbytes);
        xy_shuffle(y, (operand >> 27) & 3, ybytes);

        // z =         z +/- (f(x, y) >>  s)  for f being * or +
        // z = sat_i16(z +/- (f(x, y) >> 16)) for f being SQRDMLAH / SQRDMLSH
        // with various width/sign/shuffle arrangements for x and y
        // and various width arrangements for z (interleaving of z dependent on widths of x/y/z)
        // write-mask, or broadcast from y, or x=0, or y=0

        uint64_t x_enable = parse_writemask(operand >> 32, xbytes, 9);
        uint64_t y_enable = parse_writemask(operand >> 32, ybytes, 9);
        if (broadcast_y) {
            x_enable = ~(uint64_t)0;
            y_enable = ~(uint64_t)0;
        } else if (((operand >> (32+6)) & 7) == 0) {
            uint32_t val = (operand >> 32) & 0x3F;
            if (val == 4) {
                memset(x, 0, 64);
            } else if (val == 5) {
                memset(y, 0, 64);
            }
        }

        uint32_t xsignext = (operand & VECINT_SIGNED_X) ? (64 - xbits) : 0;
        uint32_t ysignext = (operand & VECINT_SIGNED_Y) ? (64 - ybits) : 0;
        uint32_t zsignext = 64 - zbits;
        uint32_t step = min(xbytes, ybytes);
        uint32_t zmask = (zbytes / step) - 1;
        for (uint32_t i = 0; i < 64; i += step) {
            uint32_t xi = i & -xbytes & ximask;
            if (!((x_enable >> xi) & 1)) continue;
            uint32_t yj = broadcast_y ? ((operand >> 32) * ybytes) & 0x3f : i & -ybytes;
            if (!((y_enable >> yj) & 1)) continue;

            int64_t xv = load_int(x + xi, xbytes, xsignext);
            int64_t yv = load_int(y + yj, ybytes, ysignext);
            void* z = &state->z[bit_select(z_row, i / step, zmask)].u8[i & -zbytes];
            int64_t zv = load_int(z, zbytes, zsignext);
            int64_t result = vecint_alu(xv, yv, zv, alumode, shift) & omask;
            store_int(z, zbytes, result);
        }
    }
}
