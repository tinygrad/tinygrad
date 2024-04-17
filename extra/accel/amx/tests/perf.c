#include "aarch64.h"
#include <time.h>
#include <dispatch/dispatch.h>
#include <stdio.h>

typedef struct thread_context_t {
    uint64_t spacer0[16];
    void (*kernel)(uint64_t, const uint64_t*);
    uint64_t count;
    uint64_t args[16];
    uint64_t spacer1[16];
    uint64_t times[6];
    uint64_t spacer2[16];
} thread_context_t;

typedef struct perfmode_t {
    const char* name;
    uint64_t encoding;
    uint16_t op_scale;
    uint8_t z_mask;
    uint8_t flags;
} perfmode_t;

typedef struct perfcase_t {
    const char* name;
    uint64_t encoding;
    uint16_t op_scale;
    uint8_t z_mask;
    uint8_t flags;
    uint8_t opcode;
    const perfmode_t* width_modes;
    const perfmode_t* alu_modes;
} perfcase_t;

#define FLAG_VEC 1
#define FLAG_INT 2
#define FLAG_M2 4

static const perfmode_t fma_alu_modes[] = {
    {"x*y+z",          0, 2, 0x3f, 0},
    {"x*y"  , 1ull << 27, 1, 0x3f, 0},
    {"x+z"  , 1ull << 28, 1, 0x3f, 0},
    {"y+z"  , 1ull << 29, 1, 0x3f, 0},
    {},
};

static const perfmode_t fma16_width_modes_mat[] = {
    {"f16f16",          0, 32, 0x01, 0},
    {"f16f32", 1ull << 62, 32, 0x00, 0},
    {},
};

static const perfmode_t fma16_width_modes_vec[] = {
    {"f16f16",          0, 32, 0x3f, 0},
    {},
};

static const perfmode_t fma32_width_modes[] = {
    {"f16f32", 3ull << 60, 16, 0x3f, 0},
    {"f32f32",          0, 16, 0x3f, 0},
    {},
};

static const perfmode_t fma64_width_modes[] = {
    {"f64f64",          0,  8, 0x3f, 0},
    {},
};

static const perfmode_t mac16_width_modes_mat[] = {
    {"i8i16" , 3ull << 60, 32, 0x01, 0},
    {"i8i32" , 7ull << 60, 32, 0x00, 0},
    {"i16i16",          0, 32, 0x01, 0},
    {"i16i32", 4ull << 60, 32, 0x00, 0},
    {},
};

static const perfmode_t mac16_width_modes_vec[] = {
    {"i8i16" , 3ull << 60, 32, 0x3f, 0},
    {"i16i16",          0, 32, 0x3f, 0},
    {},
};

static const perfmode_t matfp_width_modes[] = {
    {"bf16bf16", 0ull << 42, 32 * 32, 0x01, FLAG_M2},
    {"bf16f32" , 1ull << 42, 32 * 32, 0x00, FLAG_M2},
    {"f16f16"  , 2ull << 42, 32 * 32, 0x01, 0},
    {"f16f32"  , 3ull << 42, 32 * 32, 0x00, 0},
    {"f32f32"  , 4ull << 42, 16 * 16, 0x03, 0},
    {"f64f64"  , 7ull << 42,  8 *  8, 0x07, 0},
    {},
};

static const perfmode_t matfp_alu_modes[] = {
    {"x*y+z",          0, 2, 0x3f, 0},
    {},
};

static const perfmode_t vecfp_width_modes[] = {
    {"bf16bf16", 0ull << 42, 1 * 32, 0x3f, FLAG_M2},
    {"bf16f32" , 1ull << 42, 1 * 32, 0x3e, FLAG_M2},
    {"f16f16"  , 2ull << 42, 1 * 32, 0x3f, 0},
    {"f16f32"  , 3ull << 42, 1 * 32, 0x3e, 0},
    {"f32f32"  , 4ull << 42, 1 * 16, 0x3f, 0},
    {"f64f64"  , 7ull << 42, 1 *  8, 0x3f, 0},
    {},
};

static const perfmode_t vecfp_alu_modes[] = {
    {"x*y+z"   ,                           0, 1 * 2, 0x3f, 0},
    {"x*y+z_x2",  1ull << 31                , 2 * 2, 0x1f, FLAG_M2},
    {"x*y+z_x4", (1ull << 31) | (1ull << 25), 4 * 2, 0x0f, FLAG_M2},
    {},
};

static const perfmode_t matint_width_modes[] = {
    {"i8i16",  (8ull << 47) | ( 9ull << 42), 32 * 64, 0x00, 0},
    {"i8i32",  (8ull << 47) | (10ull << 42), 16 * 64, 0x00, 0},
    {"i16i16",                  2ull << 42,  32 * 32, 0x01, 0},
    {"i16i32",                  3ull << 42,  32 * 32, 0x00, 0},
    {},
};

static const perfmode_t matint_alu_modes[] = {
    {"x*y+z",          0, 2, 0x3f, 0},
    {},
};

static const perfmode_t vecint_width_modes[] = {
    {"i8i16",  (11ull << 42), 64, 0x3e, 0},
    {"i8i32",  (10ull << 42), 64, 0x3c, 0},
    {"i16i16",   2ull << 42,  32, 0x3f, 0},
    {"i16i32",   3ull << 42,  32, 0x3e, 0},
    {},
};

static const perfmode_t vecint_alu_modes[] = {
    {"x*y+z"   ,                           0, 1 * 2, 0x3f, 0},
    {"x*y+z_x2",  1ull << 31                , 2 * 2, 0x1f, FLAG_M2},
    {"x*y+z_x4", (1ull << 31) | (1ull << 25), 4 * 2, 0x0f, FLAG_M2},
    {},
};

static const perfcase_t perfcases[] = {
    {"fma16_mat", 0, 32, 0x3f, 0, 15, fma16_width_modes_mat, fma_alu_modes},
    {"fma32_mat", 0, 16, 0x03, 0, 12, fma32_width_modes, fma_alu_modes},
    {"fma64_mat", 0,  8, 0x07, 0, 10, fma64_width_modes, fma_alu_modes},

    {"fma16_vec", 1ull << 63, 1, 0x3f, FLAG_VEC, 15, fma16_width_modes_vec, fma_alu_modes},
    {"fma32_vec", 1ull << 63, 1, 0x3f, FLAG_VEC, 12, fma32_width_modes, fma_alu_modes},
    {"fma64_vec", 1ull << 63, 1, 0x3f, FLAG_VEC, 10, fma64_width_modes, fma_alu_modes},
    
    {"mac16_mat",          0, 32, 0x01, FLAG_INT, 14, mac16_width_modes_mat, fma_alu_modes},
    {"mac16_vec", 1ull << 63,  1, 0x3f, FLAG_INT | FLAG_VEC, 14, mac16_width_modes_vec, fma_alu_modes},

    {"matfp", 0, 1, 0x07, 0, 21, matfp_width_modes, matfp_alu_modes},
    {"vecfp", 0, 1, 0x3f, FLAG_VEC, 19, vecfp_width_modes, vecfp_alu_modes},

    {"matint", 0, 1, 0x03, FLAG_INT, 20, matint_width_modes, matint_alu_modes},
    {"vecint", 0, 1, 0x3f, FLAG_INT | FLAG_VEC, 18, vecint_width_modes, vecint_alu_modes},
    {},
};

#define barrier __asm volatile("dsb ish" : : : "memory")

static uint64_t now(clockid_t which) {
    barrier;
    uint64_t result = clock_gettime_nsec_np(which);
    barrier;
    return result;
}

static void thread_func(void* ctx_, size_t rank) {
    thread_context_t* ctx = (thread_context_t*)ctx_;
    AMX_SET();
    uint64_t t0 = now(CLOCK_THREAD_CPUTIME_ID);
    ctx->kernel(ctx->count, ctx->args);
    uint64_t t1 = now(CLOCK_THREAD_CPUTIME_ID);
    AMX_CLR();
    ctx->times[rank] = (t1 - t0);
}

typedef void (*perf_kernel_t)(uint64_t, const uint64_t*);
perf_kernel_t get_perf_kernel(uint32_t op, uint32_t nregs);

static uint64_t add_one_using_bitmask_hi(uint64_t x, uint64_t bitmask) {
    for (uint64_t bit = 1ull << 63; bit; bit >>= 1) {
        if (bitmask & bit) {
            x ^= bit;
            if (x & bit) {
                break;
            }
        }
    }
    return x;
}

static uint64_t add_one_using_bitmask_lo(uint64_t x, uint64_t bitmask) {
    for (uint64_t bit = 1; bit; bit <<= 1) {
        if (bitmask & bit) {
            x ^= bit;
            if (x & bit) {
                break;
            }
        }
    }
    return x;
}

#define PTR_ROW_FLAGS(ptr, row, flags) (((uint64_t)&*(ptr)) + (((uint64_t)((row) + (flags) * 64)) << 56))

static uint8_t detect_hardware_m2_flag() {
    __attribute__((aligned(256))) uint8_t buf[256];
    buf[128] = 1;
    AMX_SET(); // Set x[0:8] to zero
    AMX_LDX(PTR_ROW_FLAGS(buf, 16, 1)); // On M1: copy buf[0:128] to x[0:2], on M2: copy buf[0:256] to x[0:4]
    AMX_STX(PTR_ROW_FLAGS(buf,  2, 0)); // Copy x[2] to buf[0:64]
    AMX_CLR();
    return buf[0] == 1 ? FLAG_M2 : 0;
}

int main() {
    uint8_t allowed_flags = FLAG_VEC | FLAG_INT | detect_hardware_m2_flag();
    printf("test_name,z_spacing,num_z,bytes_per_z,num_threads,insn_total,ops_total,ns_elapsed,ops_unit\n");
    thread_context_t ctx;
    for (const perfcase_t* perfcase = perfcases; perfcase->name; ++perfcase) {
        for (const perfmode_t* width_mode = perfcase->width_modes; width_mode->name; ++width_mode) {
            for (const perfmode_t* alu_mode = perfcase->alu_modes; alu_mode->name; ++alu_mode) {
                uint8_t flags = perfcase->flags | width_mode->flags | alu_mode->flags;
                if (flags & ~allowed_flags) goto next_mode;
                uint8_t z_mask = perfcase->z_mask & width_mode->z_mask & alu_mode->z_mask;
                uint32_t max_num_z = 1u << __builtin_popcount(z_mask);
                uint32_t bytes_per_z = 4096 / max_num_z;
                if (max_num_z > 16) {
                    max_num_z = 16;
                }
                uint32_t ops_per_insn = (uint32_t)perfcase->op_scale * (uint32_t)width_mode->op_scale * (uint32_t)alu_mode->op_scale;
                uint64_t op_count_target = 4096ull * 4096ull * 5760ull;
                if (!(flags & FLAG_VEC)) {
                    op_count_target *= 20ull;
                }
                if (ops_per_insn > 64) {
                    op_count_target *= 2;
                }
                for (uint32_t n_z = 1; n_z <= max_num_z; ++n_z) {
                    ctx.kernel = get_perf_kernel(perfcase->opcode, n_z);
                    for (uint32_t far_z = 0; far_z <= 1; ++far_z) {
                        uint64_t encoding = perfcase->encoding ^ width_mode->encoding ^ alu_mode->encoding;
                        for (uint32_t z = 0; z < n_z; ++z) {
                            ctx.args[z] = encoding;
                            encoding = (far_z ? add_one_using_bitmask_hi : add_one_using_bitmask_lo)(encoding, ((uint64_t)z_mask) << 20);
                        }
                        for (uint32_t n_threads = 1; n_threads <= 6; ++n_threads) {
                            ctx.count = op_count_target / (ops_per_insn * n_threads * n_z);
                            ctx.count = (ctx.count + 4) & ~(uint64_t)7;
                            uint64_t t0 = now(CLOCK_REALTIME);
                            dispatch_apply_f(n_threads, DISPATCH_APPLY_AUTO, (void*)&ctx, thread_func);
                            uint64_t t1 = now(CLOCK_REALTIME);
                            uint64_t wall_elapsed = (t1 - t0);
                            uint64_t insn_total = n_threads * ctx.count * n_z;
                            printf("%s_%s_%s,%s,%u,%u,%u,%llu,%llu,%llu,%s\n",
                                perfcase->name, width_mode->name, alu_mode->name,
                                far_z ? "far" : "near", (unsigned)n_z, (unsigned)bytes_per_z, (unsigned)n_threads,
                                (long long unsigned)insn_total, (long long unsigned)(insn_total * ops_per_insn), (long long unsigned)wall_elapsed,
                                flags & FLAG_INT ? "GOPS" : "GFLOPS");
                            fflush(stdout);
                        }
                    }
                }
                next_mode:;
            }
        }
    }
    return 0;
}
