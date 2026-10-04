#include "kittens.cuh"

using namespace kittens;

#ifndef TOKENS
#define TOKENS 2048
#endif
#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef HIDDEN
#define HIDDEN 3072
#endif
#ifndef PART_N
#define PART_N 32
#endif
#ifndef PART_M
#define PART_M 8
#endif

constexpr int TILE = 128;
constexpr int K_LOGIT = 64;
constexpr int K_GRAD = 16;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int THREADS = NUM_WARPS * WARP_THREADS;
constexpr int MTILES = TOKENS / TILE;
constexpr int NTILES = VOCAB / TILE;

using G = kittens::group<NUM_WARPS>;
using HLogST = st_bf<TILE, K_LOGIT, st_16x32_s>;
using WLogST = st_bf<TILE, K_LOGIT, st_16x32_s>;
using DZST = st_bf<TILE, TILE, st_16x16_s>;
using HGradST = st_bf<TILE, K_GRAD, st_16x16_s>;
using WGradST = st_bf<TILE, K_GRAD, st_16x16_s>;

__global__ __launch_bounds__(THREADS, 1) void lmhead_ce_bwd_hk(
    float *__restrict__ dh_partial, float *__restrict__ dw_partial,
    bf16 *__restrict__ hidden_ptr, bf16 *__restrict__ weight_ptr,
    const int *__restrict__ targets, const float *__restrict__ lse, const float *__restrict__ loss_scale) {
  const int wg = __builtin_amdgcn_workgroup_id_x();
  const int pn = wg % PART_N;
  const int pm = wg / PART_N;
  const int wid = warpid();
  const int warp_m = wid / WARPS_N;
  const int warp_n = wid % WARPS_N;
  const int lane = laneid();
  const int lrow_grp = 4 * (lane / 16);
  const int lcol = lane % 16;

  gl<bf16, 1, 1, TOKENS, HIDDEN> H{hidden_ptr, nullptr, nullptr, nullptr, nullptr};
  gl<bf16, 1, 1, VOCAB, HIDDEN> W{weight_ptr, nullptr, nullptr, nullptr, nullptr};

  __shared__ alignment_dummy __shm[MAX_SHARED_MEMORY / sizeof(alignment_dummy)];
  shared_allocator al((int*)&__shm[0]);
  DZST &DZs = al.allocate<DZST>();
  HLogST &HLs = al.allocate<HLogST>();
  WLogST &WLs = al.allocate<WLogST>();
  HGradST &HGs = al.allocate<HGradST>();
  WGradST &WGs = al.allocate<WGradST>();
  st_bf<512, K_GRAD, st_16x16_s> &GradScratch = al.allocate<st_bf<512, K_GRAD, st_16x16_s>>();
  bf16 *scratch = GradScratch.data;


  for (int mt = pm; mt < MTILES; mt += PART_M) {
    for (int nt = pn; nt < NTILES; nt += PART_N) {
      {
      rt_bf<64, K_LOGIT, row_l, rt_16x32_s> hlog;
      rt_bf<32, K_LOGIT, row_l, rt_16x32_s> wlog;
      rt_fl<64, 32, col_l, rt_16x16_s> z;
      rt_bf<64, 32, col_l, rt_16x16_s> zbf;
      zero(z);
      for (int kk = 0; kk < HIDDEN / K_LOGIT; kk++) {
        G::load(HLs, H, {0, 0, mt, kk});
        G::load(WLs, W, {0, 0, nt, kk});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        load(hlog, subtile_inplace<64, K_LOGIT>(HLs, {warp_m, 0}));
        load(wlog, subtile_inplace<32, K_LOGIT>(WLs, {warp_n, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(z, hlog, wlog, z);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
      }

      copy(zbf, z);     // round logits to the BF16 numerical contract
      copy(z, zbf);     // apply the derivative to those rounded values in FP32
      #pragma unroll
      for (int i = 0; i < 64/16; i++) {
        #pragma unroll
        for (int j = 0; j < 32/16; j++) {
          const int gr0 = mt * TILE + warp_m * 64 + i * 16 + lrow_grp;
          const int gc = nt * TILE + warp_n * 32 + j * 16 + lcol;
          float vals[4] = {z.tiles[i][j].data[0].x, z.tiles[i][j].data[0].y,
                           z.tiles[i][j].data[1].x, z.tiles[i][j].data[1].y};
          #pragma unroll
          for (int r = 0; r < 4; r++) {
            const float ex = expf(fminf(vals[r] - lse[gr0 + r], 0.0f));
            vals[r] = (ex - (targets[gr0 + r] == gc ? 1.0f : 0.0f)) * loss_scale[0];
          }
          z.tiles[i][j].data[0].x = vals[0]; z.tiles[i][j].data[0].y = vals[1];
          z.tiles[i][j].data[1].x = vals[2]; z.tiles[i][j].data[1].y = vals[3];
        }
      }
      copy(zbf, z);
      auto dzs_wave = subtile_inplace<64, 32>(DZs, {warp_m, warp_n});
      store(dzs_wave, zbf);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      }

      for (int hk = 0; hk < HIDDEN / K_GRAD; hk++) {
        G::load(HGs, H, {0, 0, mt, hk});
        G::load(WGs, W, {0, 0, nt, hk});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        {
          rt_bf<64, 32, row_l, rt_16x32_s> dzrow;
          rt_bf<32, K_GRAD, col_l, rt_32x16_s> wcol;
          rt_fl<64, K_GRAD, col_l, rt_16x16_s> dh;
          load(dzrow, subtile_inplace<64, 32>(DZs, {warp_m, warp_n}));
          load(wcol, subtile_inplace<32, K_GRAD>(WGs, {warp_n, 0}));
          asm volatile("s_waitcnt lgkmcnt(0)");
          zero(dh);
          __builtin_amdgcn_s_setprio(1);
          mma_AB(dh, dzrow, wcol, dh);
          __builtin_amdgcn_s_setprio(0);
          __builtin_amdgcn_sched_barrier(0);
          auto dh_slot = subtile_inplace<64, K_GRAD>(GradScratch, {wid, 0});
          store(dh_slot, dh);
          asm volatile("s_waitcnt lgkmcnt(0)");
          __builtin_amdgcn_s_barrier();
          for (int wm = 0; wm < WARPS_M; wm++) {
            for (int out_i = threadIdx.x; out_i < 64 * K_GRAD; out_i += THREADS) {
              const int rr = out_i / K_GRAD, cc = out_i % K_GRAD;
              float sum = 0.0f;
              #pragma unroll
              for (int wn = 0; wn < WARPS_N; wn++) sum += (float)scratch[((wm * WARPS_N + wn) * 64 + rr) * K_GRAD + cc];
              const long long oi = ((long long)pn * TOKENS + mt * TILE + wm * 64 + rr) * HIDDEN + hk * K_GRAD + cc;
              if (nt == pn) dh_partial[oi] = sum; else dh_partial[oi] += sum;
            }
            __builtin_amdgcn_s_barrier();
          }
        }

        {
          rt_bf<64, 32, col_l, rt_32x16_s> dzcol;
          rt_bf<64, K_GRAD, col_l, rt_32x16_s> hcol;
          rt_fl<32, K_GRAD, col_l, rt_16x16_s> dw;
          load(dzcol, subtile_inplace<64, 32>(DZs, {warp_m, warp_n}));
          load(hcol, subtile_inplace<64, K_GRAD>(HGs, {warp_m, 0}));
          asm volatile("s_waitcnt lgkmcnt(0)");
          zero(dw);
          __builtin_amdgcn_s_setprio(1);
          mma_AtB(dw, dzcol, hcol, dw);
          __builtin_amdgcn_s_setprio(0);
          __builtin_amdgcn_sched_barrier(0);
          auto dw_slot = subtile_inplace<32, K_GRAD>(GradScratch, {wid, 0});
          store(dw_slot, dw);
          asm volatile("s_waitcnt lgkmcnt(0)");
          __builtin_amdgcn_s_barrier();
          for (int out_i = threadIdx.x; out_i < TILE * K_GRAD; out_i += THREADS) {
            const int rr = out_i / K_GRAD, cc = out_i % K_GRAD;
            const int wn = rr / 32, r32 = rr % 32;
            const float sum = (float)scratch[((0 * WARPS_N + wn) * 32 + r32) * K_GRAD + cc]
                            + (float)scratch[((1 * WARPS_N + wn) * 32 + r32) * K_GRAD + cc];
            const long long oi = ((long long)pm * VOCAB + nt * TILE + rr) * HIDDEN + hk * K_GRAD + cc;
            if (mt == pm) dw_partial[oi] = sum; else dw_partial[oi] += sum;
          }
          __builtin_amdgcn_s_barrier();
        }
      }
    }
  }
}
