#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>
#include <cmath>

using namespace kittens;

#define SIZE 8192

#define HipCheckError() do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); exit(1); } \
} while(0)

struct TimingResult {
    float best_time_ms, avg_time_ms;
    double best_tflops, avg_tflops;
    int timing_iterations;
};


// Kernel
constexpr int NUM_WARPS  = 8;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;

using G = kittens::group<NUM_WARPS>;

template <int M, int N, int K>
__global__ __launch_bounds__(512, 2)
void mxfp8_gemm_kernel(
    const gl<fp8e4m3, 1, 1, M, K> A,
    const gl<fp8e4m3, 1, 1, N, K> B,
    const gl<float,   1, 1, M, N> C,
    const uint32_t *__restrict__ scale_A_iter,
    const uint32_t *__restrict__ scale_B_iter) {
    constexpr int k_iters      = K / BLOCK_K;
    constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

    using ST_A = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ uint8_t smem_scales[2048];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M  = M / BLOCK_ROW;
    constexpr int tiles_N  = N / BLOCK_COL;
    const int NUM_XCDS     = 8;
    const int WGM          = 8;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;
    int block_m      = block_row * BLOCK_ROW;
    int block_n      = block_col * BLOCK_COL;

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    int a_row_h0 = warp_m * REG_M;
    int a_row_h1 = HALF_ROW + warp_m * REG_M;
    int b_row_h0 = warp_n * REG_N;
    int b_row_h1 = HALF_COL + warp_n * REG_N;

    int tic = 0, toc = 1;

    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A);

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A);
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        load_scales_to_lds(smem_scales, scale_A_iter, scale_B_iter, block_m, block_n, k, M, N);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(smem_scales, 0, a_row_h0);
        fp8e8m0_4 sb_h0 = pack_scales(smem_scales, 1024, b_row_h0);

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fp8e8m0_4 sb_h1 = pack_scales(smem_scales, 1024, b_row_h1);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h1 = pack_scales(smem_scales, 0, a_row_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    { // Epilogue k = k_iters - 2
        int k = k_iters - 2;
        load_scales_to_lds(smem_scales, scale_A_iter, scale_B_iter, block_m, block_n, k, M, N);
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(smem_scales, 0, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(smem_scales, 0, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(smem_scales, 1024, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(smem_scales, 1024, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs0_next = subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load(b0, bs0_next);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1;
    }

    { // Epilogue k = k_iters - 1 (final)
        int k = k_iters - 1;
        load_scales_to_lds(smem_scales, scale_A_iter, scale_B_iter, block_m, block_n, k, M, N);
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(smem_scales, 0, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(smem_scales, 0, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(smem_scales, 1024, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(smem_scales, 1024, b_row_h1);

        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}


// Host: quantization + scale packing
static uint8_t compute_e8m0(const float *vals, int count) {
    float mx = 0.0f;
    for (int i = 0; i < count; i++) mx = std::max(mx, std::abs(vals[i]));
    if (mx == 0.0f) return 0;
    return (uint8_t)std::clamp((int)std::floor(std::log2(mx)) + 127, 0, 254);
}

template <int M, int N, int K>
void random_init_mxfp8(
    std::vector<float> &a_f, std::vector<float> &b_f,
    std::vector<fp8e4m3> &a_q, std::vector<fp8e4m3> &b_q,
    std::vector<uint8_t> &sa_raw, std::vector<uint8_t> &sb_raw,
    std::vector<uint32_t> &sa_iter, std::vector<uint32_t> &sb_iter,
    uint32_t seed = 42) {
    constexpr int scale_K  = K / 32;
    constexpr int k_iters  = K / BLOCK_K;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    a_f.resize(M * K); b_f.resize(N * K);
    for (auto &v : a_f) v = dist(rng);
    for (auto &v : b_f) v = dist(rng);

    a_q.resize(M * K); b_q.resize(N * K);
    sa_raw.resize(M * scale_K); sb_raw.resize(N * scale_K);

    auto quantize_matrix = [](const std::vector<float> &src, std::vector<fp8e4m3> &dst,
                              std::vector<uint8_t> &scales, int rows, int cols, int sk) {
        for (int row = 0; row < rows; row++)
            for (int kb = 0; kb < sk; kb++) {
                float blk[32];
                for (int i = 0; i < 32; i++) blk[i] = src[row * cols + kb * 32 + i];
                uint8_t s = compute_e8m0(blk, 32);
                scales[row * sk + kb] = s;
                float inv = std::ldexp(1.0f, 127 - (int)s);
                for (int i = 0; i < 32; i++) dst[row * cols + kb * 32 + i] = fp8e4m3(blk[i] * inv);
            }
    };
    quantize_matrix(a_f, a_q, sa_raw, M, K, scale_K);
    quantize_matrix(b_f, b_q, sb_raw, N, K, scale_K);

    sa_iter.resize(k_iters * M); sb_iter.resize(k_iters * N);
    auto pack_scales_host = [](const std::vector<uint8_t> &raw, std::vector<uint32_t> &iter,
                               int dim, int scale_K, int k_iters) {
        for (int ki = 0; ki < k_iters; ki++) {
            int kb_base = ki * 4;
            for (int row = 0; row < dim; row++) {
                uint32_t p = 0;
                for (int j = 0; j < 4; j++)
                    p |= (uint32_t)raw[row * scale_K + kb_base + j] << (j * 8);
                iter[ki * dim + row] = p;
            }
        }
    };
    pack_scales_host(sa_raw, sa_iter, M, scale_K, k_iters);
    pack_scales_host(sb_raw, sb_iter, N, scale_K, k_iters);
}


// Benchmark with rotating buffers
constexpr int ROTATING_BUFFER_COUNT = 4;

template <int M, int N, int K>
TimingResult run_benchmark(int warmup_iters, int timing_iters) {
    constexpr int k_iters = K / BLOCK_K;
    constexpr int grid    = (M / BLOCK_ROW) * (N / BLOCK_COL);

    std::vector<float> a_f, b_f;
    std::vector<fp8e4m3> a_q, b_q;
    std::vector<uint8_t> sa_raw, sb_raw;
    std::vector<uint32_t> sa_iter, sb_iter;

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a, (size_t)ROTATING_BUFFER_COUNT * M * K);
    hipMalloc(&d_b, (size_t)ROTATING_BUFFER_COUNT * N * K);
    hipMalloc(&d_c, (size_t)M * N * sizeof(float));
    hipMalloc(&d_sa, (size_t)ROTATING_BUFFER_COUNT * k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, (size_t)ROTATING_BUFFER_COUNT * k_iters * N * sizeof(uint32_t));
    HipCheckError();

    for (int buf = 0; buf < ROTATING_BUFFER_COUNT; buf++) {
        random_init_mxfp8<M, N, K>(a_f, b_f, a_q, b_q, sa_raw, sb_raw, sa_iter, sb_iter, 42 + buf);
        hipMemcpy(d_a  + (size_t)buf * M * K, a_q.data(), (size_t)M * K, hipMemcpyHostToDevice);
        hipMemcpy(d_b  + (size_t)buf * N * K, b_q.data(), (size_t)N * K, hipMemcpyHostToDevice);
        hipMemcpy(d_sa + (size_t)buf * k_iters * M, sa_iter.data(), (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
        hipMemcpy(d_sb + (size_t)buf * k_iters * N, sb_iter.data(), (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    }
    HipCheckError();

    for (int i = 0; i < warmup_iters; i++) {
        int buf = i % ROTATING_BUFFER_COUNT;
        gl<fp8e4m3, 1, 1, M, K> A(d_a + (size_t)buf * M * K, nullptr, nullptr, nullptr, nullptr);
        gl<fp8e4m3, 1, 1, N, K> B(d_b + (size_t)buf * N * K, nullptr, nullptr, nullptr, nullptr);
        gl<float,   1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
        mxfp8_gemm_kernel<M, N, K><<<grid, 512>>>(A, B, C,
            d_sa + (size_t)buf * k_iters * M, d_sb + (size_t)buf * k_iters * N);
        hipDeviceSynchronize();
    }

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    std::vector<float> times;
    for (int r = 0; r < timing_iters; r++) {
        int buf = r % ROTATING_BUFFER_COUNT;
        gl<fp8e4m3, 1, 1, M, K> A(d_a + (size_t)buf * M * K, nullptr, nullptr, nullptr, nullptr);
        gl<fp8e4m3, 1, 1, N, K> B(d_b + (size_t)buf * N * K, nullptr, nullptr, nullptr, nullptr);
        gl<float,   1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
        hipEventRecord(t0);
        mxfp8_gemm_kernel<M, N, K><<<grid, 512>>>(A, B, C,
            d_sa + (size_t)buf * k_iters * M, d_sb + (size_t)buf * k_iters * N);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        times.push_back(ms);
    }

    float best = *std::min_element(times.begin(), times.end());
    float avg  = 0; for (float t : times) avg += t; avg /= times.size();
    double ops = 2.0 * M * N * K;

    hipEventDestroy(t0); hipEventDestroy(t1);
    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);

    return { best, avg, ops / (best * 1e-3) / 1e12, ops / (avg * 1e-3) / 1e12, timing_iters };
}


// Correctness
template <int M, int N, int K>
bool run_correctness() {
    constexpr int scale_K = K / 32;
    constexpr int k_iters = K / BLOCK_K;
    constexpr int grid    = (M / BLOCK_ROW) * (N / BLOCK_COL);

    std::vector<float> a_f, b_f;
    std::vector<fp8e4m3> a_q, b_q;
    std::vector<uint8_t> sa_raw, sb_raw;
    std::vector<uint32_t> sa_iter, sb_iter;
    random_init_mxfp8<M, N, K>(a_f, b_f, a_q, b_q, sa_raw, sb_raw, sa_iter, sb_iter);

    printf("  Computing CPU reference...\n");
    constexpr size_t MN = (size_t)M * N;
    std::vector<float> c_ref(MN, 0.0f);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int kb = 0; kb < scale_K; kb++) {
                float sa = std::ldexp(1.0f, (int)sa_raw[i * scale_K + kb] - 127);
                float sb = std::ldexp(1.0f, (int)sb_raw[j * scale_K + kb] - 127);
                for (int ki = 0; ki < 32; ki++)
                    acc += (float)a_q[i * K + kb * 32 + ki] * sa
                         * (float)b_q[j * K + kb * 32 + ki] * sb;
            }
            c_ref[(size_t)i * N + j] = acc;
        }

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a,  (size_t)M * K);
    hipMalloc(&d_b,  (size_t)N * K);
    hipMalloc(&d_c,  MN * sizeof(float));
    hipMalloc(&d_sa, (size_t)k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, (size_t)k_iters * N * sizeof(uint32_t));
    hipMemcpy(d_a,  a_q.data(),     (size_t)M * K, hipMemcpyHostToDevice);
    hipMemcpy(d_b,  b_q.data(),     (size_t)N * K, hipMemcpyHostToDevice);
    hipMemcpy(d_sa, sa_iter.data(),  (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, sb_iter.data(),  (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, MN * sizeof(float));

    gl<fp8e4m3, 1, 1, M, K> A_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e4m3, 1, 1, N, K> B_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    gl<float,   1, 1, M, N> C_gl(d_c, nullptr, nullptr, nullptr, nullptr);
    mxfp8_gemm_kernel<M, N, K><<<grid, 512>>>(A_gl, B_gl, C_gl, d_sa, d_sb);
    hipDeviceSynchronize();

    std::vector<float> c_gpu(MN);
    hipMemcpy(c_gpu.data(), d_c, MN * sizeof(float), hipMemcpyDeviceToHost);

    float c_max = 0;
    for (size_t i = 0; i < MN; i++) c_max = std::max(c_max, std::abs(c_ref[i]));
    float atol = c_max * 0.001f;
    int fail_count = 0;
    for (size_t i = 0; i < MN; i++)
        if (std::abs(c_gpu[i] - c_ref[i]) > atol) fail_count++;

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);

    printf("  atol (0.1%%): %.2e -> %d / %zu failures -> %s\n",
           atol, fail_count, MN, fail_count == 0 ? "PASS" : "FAIL");
    return fail_count == 0;
}


// Main
int main() {
    constexpr int M             = SIZE;
    constexpr int N             = SIZE;
    constexpr int K             = SIZE;
    constexpr int warmup_iters  = 500;
    constexpr int timing_iters  = 100;

    printf("=== MXFP8 TN 8-wave GEMM ===\n");
    printf("Matrix dimensions: %dx%dx%d\n", M, N, K);
    printf("Warmup: %d, Timing: %d\n\n", warmup_iters, timing_iters);

    printf("Correctness check:\n");
    bool pass = run_correctness<M, N, K>();

    if (!pass) {
        printf("\nCorrectness FAILED — skipping benchmark.\n");
        return 1;
    }

    printf("\nRunning benchmark...\n");
    TimingResult res = run_benchmark<M, N, K>(warmup_iters, timing_iters);

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", res.best_time_ms, res.best_tflops);
    printf("Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", res.avg_time_ms, res.avg_tflops);
    printf("\nCorrectness: PASSED\n");

    return 0;
}
