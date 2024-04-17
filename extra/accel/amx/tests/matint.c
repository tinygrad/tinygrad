#include "emulate.h"

#define MATINT_SIGNED_Z (1ull << 63)

#define MATINT_INDEXED_LOAD (1ull << 53)
#define MATINT_INDEXED_LOAD_Y (1ull << 47)
#define MATINT_INDEXED_LOAD_4BIT (1ull << 48)

#define MATINT_SIGNED_X (1ull << 63)
#define MATINT_SIGNED_Y (1ull << 26)

#define MATINT_ENABLE_MASK_IS_Y (1ull << 25)

int64_t vecint_alu_mode4(int64_t val, uint32_t satbits, uint64_t operand);
int64_t vecint_alu(int64_t x, int64_t y, int64_t z, int alumode, uint32_t shift);

void emulate_AMX_MATINT(amx_state* state, uint64_t operand) {
    if ((operand >> 55) & 3) {
        return;
    }
    if ((operand & (1ull << 54)) && !(operand & MATINT_INDEXED_LOAD)) {
        return;
    }

    uint64_t z_row = (operand >> 20) & 3;
    int32_t omask = (((operand >> 32) & 0x1ff) == 3) ? 0 : -1;
    int alumode = (operand & MATINT_INDEXED_LOAD) ? (operand & (1ull << 54)) >> 51 : (operand >> 47) & 0x3f;
    uint32_t shift = (operand >> 58) & 0x1f;

    uint32_t xbits = 0, ybits = 0, zbits, satbits;
    if (alumode == 4) {
        switch ((operand >> 42) & 0xf) {
        case  3: zbits = 32; satbits = 16; break;
        case  4: zbits = 32; satbits = 32; break;
        case 10: zbits = 32; satbits =  8; break;
        case 11: zbits = 16; satbits =  8; break;
        default: zbits = 16; satbits = 16; break;
        }
    } else if (alumode == 5 || alumode == 6) {
        xbits = ybits = 16; zbits = 16;
        shift = 15;
    } else if (alumode == 8) {
        switch ((operand >> 42) & 0xf) {
        case 10: xbits = ybits = 8; zbits = 32; break;
        case 12: xbits = 8; if (AMX_VER >= AMX_VER_M3) { ybits = 16; zbits = 32; } else { ybits = 8; zbits = 16; } break;
        default: xbits = ybits = 8; zbits = 16; break;
        }
    } else if (alumode == 9) {
        switch ((operand >> 42) & 0xf) {
        case  3: xbits = ybits = 16; zbits = 32; break;
        case  4: xbits = ybits = 32; zbits = 32; break;
        default: xbits = ybits = 16; zbits = 16; break;
        }
        shift = 64 - xbits; // Not actually used as a shift
    } else {
        switch ((operand >> 42) & 0xf) {
        case  3: xbits = ybits = 16; zbits = 32; break;
        default: xbits = ybits = 16; zbits = 16; break;
        }
    }
    uint32_t xbytes = xbits / 8;
    uint32_t ybytes = ybits / 8;
    uint32_t zbytes = zbits / 8;
    
    if (alumode == 4) {
        // z = f(z), where f is [rounding] shift followed by [saturate]
        // with various options for width and signedness
        uint32_t zsignext = (operand & MATINT_SIGNED_Z) ? (64 - zbits) : 0;
        uint64_t col_enable = parse_writemask(operand >> 32, zbytes, 9);
        for (uint32_t j = 0; j < 64; j += zbytes) {
            for (uint32_t i = 0; i < 64; i += zbytes) {
                if (!((col_enable >> (operand & MATINT_ENABLE_MASK_IS_Y ? j : i)) & 1)) continue;
                void* z = &state->z[bit_select(j, z_row, zbytes - 1)].u8[i];
                int64_t val = load_int(z, zbytes, zsignext);
                val = vecint_alu_mode4(val, satbits, operand);
                store_int(z, zbytes, val & omask);
            }
        }
        return;
    } else if (alumode == 7 || alumode >= 10) {
        return;
    }

    uint8_t x[64];
    uint8_t y[64];
    load_xy_reg(x, state->x, (operand >> 10) & 0x1FF);
    load_xy_reg(y, state->y, operand & 0x1FF);
    if (operand & MATINT_INDEXED_LOAD) {
        uint32_t src_reg = (operand >> 49) & 7;
        uint32_t ibits = (operand & MATINT_INDEXED_LOAD_4BIT) ? 4 : 2;
        if (operand & MATINT_INDEXED_LOAD_Y) {
            load_xy_reg_indexed(y, state->y[src_reg].u8, ibits, ybits);
        } else {
            load_xy_reg_indexed(x, state->x[src_reg].u8, ibits, xbits);
        }
    }
    xy_shuffle(x, (operand >> 29) & 3, xbytes);
    xy_shuffle(y, (operand >> 27) & 3, ybytes);

    // z =         z +/- (f(x, y) >>  s)  for f being * or + or weird xor/popcnt thing
    // z = sat_i16(z +/- (f(x, y) >> 16)) for f being SQRDMLAH / SQRDMLSH
    // with various width/sign/shuffle arrangements for x and y
    // and various width arrangements for z (interleaving of z dependent on widths of x/y/z)
    // write-mask, or broadcast from y, or x=0, or y=0

    uint64_t x_enable, y_enable;
    if (operand & MATINT_ENABLE_MASK_IS_Y) {
        x_enable = ~(uint64_t)0;
        y_enable = parse_writemask(operand >> 32, ybytes, 9);
    } else {
        x_enable = parse_writemask(operand >> 32, xbytes, 9);
        y_enable = ~(uint64_t)0;
    }
    if (((operand >> (32+6)) & 7) == 0) {
        uint32_t val = (operand >> 32) & 0x3F;
        if (val == 4 || val == 5) {
            memset((operand & MATINT_ENABLE_MASK_IS_Y) ? y : x, 0, 64);
        }
    }

    uint32_t xsignext = (operand & MATINT_SIGNED_X) ? (64 - xbits) : 0;
    uint32_t ysignext = (operand & MATINT_SIGNED_Y) ? (64 - ybits) : 0;
    uint32_t zsignext = 64 - zbits;
    uint32_t zmask = (zbytes / xbytes) - 1;
    uint32_t step = xbytes == 1 ? zbytes : xbytes;
    for (uint32_t j = 0; j < 64; j += step) {
        if (!((y_enable >> j) & 1)) continue;
        for (uint32_t i = 0; i < 64; i += xbytes) {
            if (!((x_enable >> i) & 1)) continue;
            int64_t xv = load_int(x + i, xbytes, xsignext);
            int64_t yv = load_int(y + j, ybytes, ysignext);
            void* z = &state->z[bit_select(bit_select(j, z_row, xbytes - 1), i / xbytes, zmask)].u8[i & -zbytes];
            int64_t zv = load_int(z, zbytes, zsignext);
            int64_t result = vecint_alu(xv, yv, zv, alumode, shift) & omask;
            store_int(z, zbytes, result);
        }
    }
}
