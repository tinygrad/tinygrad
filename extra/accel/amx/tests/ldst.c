#include "emulate.h"
#include <stdio.h>

#define LDST_MULTIPLE (1ull << 62)
#define LDST_NON_CONSECUTIVE (1ull << 61)
#define LDST_MULTIPLE_MEANS_FOUR (1ull << 60)

static void ld_common(amx_reg* regs, uint64_t operand, uint32_t regmask) {
    uint32_t rn = (operand >> 56) & regmask;
    const uint8_t* src = (uint8_t*)((operand << 8) >> 8);
    memcpy(regs + rn, src, 64);
    if (operand & LDST_MULTIPLE) {
        uint32_t rs = 1;
        if ((AMX_VER >= AMX_VER_M3) && (operand & LDST_NON_CONSECUTIVE) && (regmask <= 15)) {
            rs = (operand & LDST_MULTIPLE_MEANS_FOUR) ? 2 : 4;
        }
        memcpy(regs + ((rn + rs) & regmask), src + 64, 64);
        if ((AMX_VER >= AMX_VER_M2) && (operand & LDST_MULTIPLE_MEANS_FOUR) && (regmask <= 15)) {
            memcpy(regs + ((rn + rs*2) & regmask), src + 128, 64);
            memcpy(regs + ((rn + rs*3) & regmask), src + 192, 64);
        }
    }
}

static void st_common(const amx_reg* regs, uint64_t operand, uint32_t regmask) {
    uint32_t rn = (operand >> 56) & regmask;
    uint8_t* dst = (uint8_t*)((operand << 8) >> 8);
    memcpy(dst, regs + rn, 64);
    if (operand & LDST_MULTIPLE) {
        memcpy(dst + 64, regs + ((rn + 1) & regmask), 64);
    }
}

void emulate_AMX_LDX(amx_state* state, uint64_t operand) {
    ld_common(state->x, operand, 7);
}

void emulate_AMX_LDY(amx_state* state, uint64_t operand) {
    ld_common(state->y, operand, 7);
}

void emulate_AMX_LDZ(amx_state* state, uint64_t operand) {
    ld_common(state->z, operand, 63);
}

void emulate_AMX_LDZI(amx_state* state, uint64_t operand) {
    uint32_t rn = (operand >> 56) & 63;
    uint32_t half = (rn & 1) << 3;
    const uint32_t* src = (const uint32_t*)((operand << 8) >> 8);
    for (uint32_t i = 0; i < 16; ++i) {
        state->z[bit_select(rn, i, 1)].u32[half + (i >> 1)] = src[i];
    }
}

void emulate_AMX_STX(amx_state* state, uint64_t operand) {
    st_common(state->x, operand, 7);
}

void emulate_AMX_STY(amx_state* state, uint64_t operand) {
    st_common(state->y, operand, 7);
}

void emulate_AMX_STZ(amx_state* state, uint64_t operand) {
    st_common(state->z, operand, 63);
}

void emulate_AMX_STZI(amx_state* state, uint64_t operand) {
    uint32_t rn = (operand >> 56) & 63;
    uint32_t half = (rn & 1) << 3;
    uint32_t* dst = (uint32_t*)((operand << 8) >> 8);
    for (uint32_t i = 0; i < 16; ++i) {
        dst[i] = state->z[bit_select(rn, i, 1)].u32[half + (i >> 1)];
    }
}
