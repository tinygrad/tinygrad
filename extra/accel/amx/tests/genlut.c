#include "emulate.h"

#define GENLUT_TABLE_Y (1ull << 59)
#define GENLUT_BF16 (1ull << 30)
#define GENLUT_DEST_Z (1ull << 26)
#define GENLUT_DEST_Y (1ull << 25)
#define GENLUT_SOURCE_Y (1ull << 10)

float bf16_to_f32(uint32_t x);

static void find_first_greater_than(uint8_t* vs, uint32_t mode, const amx_reg* xy, const amx_reg* table, uint64_t operand) {
    switch (mode) {
#define SCAN_TABLE(t, n, f) \
        for (uint32_t i = 0; i < 512/n; ++i) { \
            uint32_t v = 0; \
            for (; v < 512/n; ++v) { if (f(table->t##n[v]) > f(xy->t##n[i])) break; } \
            vs[i] = v - 1; \
        }
    case 0: SCAN_TABLE(f, 32, ) break;
    case 1: if ((AMX_VER >= AMX_VER_M2) && (operand & GENLUT_BF16)) SCAN_TABLE(u, 16, bf16_to_f32) else SCAN_TABLE(f, 16, ) break;
    case 2: SCAN_TABLE(f, 64, ) break;
    case 3: SCAN_TABLE(i, 32, ) break;
    case 4: SCAN_TABLE(i, 16, ) break;
    case 5: SCAN_TABLE(u, 32, ) break;
    case 6: SCAN_TABLE(u, 16, ) break;
#undef SCAN_TABLE
    }
}

static void pack_bits(uint8_t* dst, const uint8_t* bytes, uint32_t ibits, uint32_t ebits) {
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
