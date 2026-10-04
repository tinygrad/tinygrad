#include <hip/hip_runtime.h>

// Fast coalesced-LDS transpose of the mxfp8 block-scale tensor, replacing tinygrad's slow UOp
// permute(1,0).contiguous() (2D) / permute(0,2,1).contiguous() (3D) which codegens an UNCOALESCED transpose.
// The scale is ALREADY uint32 (the free e8.reshape(...,4).bitcast(uint32).reshape(...) view), so this is a pure
// transpose (no quantize): in (E, ROWS, K4) uint32 -> out (E, K4, ROWS) uint32. Byte-identical to mx_pack/mx_pack_3d.
#ifndef MXP_ROWS
#define MXP_ROWS 4096
#endif
#ifndef MXP_K4
#define MXP_K4 24
#endif
#ifndef MXP_E
#define MXP_E 1
#endif
#ifndef TILE_ROWS
#define TILE_ROWS 64
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int NRT        = MXP_ROWS / TILE_ROWS;   // row tiles per expert (ROWS is a multiple of TILE_ROWS)
constexpr int LDS_STRIDE = TILE_ROWS + 1;          // +1 pad -> conflict-free column reads
constexpr int TOTAL      = TILE_ROWS * MXP_K4;

static_assert(MXP_ROWS % TILE_ROWS == 0, "ROWS must be a multiple of TILE_ROWS");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
mxpack_transpose(uint8_t* __restrict__ out_b, const uint8_t* __restrict__ in_b)
{
  __shared__ uint32_t lds[MXP_K4 * LDS_STRIDE];
  const uint32_t* in  = reinterpret_cast<const uint32_t*>(in_b);   // (E, ROWS, K4)
  uint32_t*       out = reinterpret_cast<uint32_t*>(out_b);        // (E, K4, ROWS)
  const int tid = threadIdx.x;
  const int e   = blockIdx.x / NRT;
  const int rt  = blockIdx.x % NRT;
  const int r0  = rt * TILE_ROWS;
  const long long in_e  = (long long)e * MXP_ROWS * MXP_K4;
  const long long out_e = (long long)e * MXP_K4 * MXP_ROWS;

  // load: in[(r0+lr)*K4 + k4] -> lds[k4][lr]   (in-addr = in_e + r0*K4 + idx -> fully coalesced read)
  #pragma unroll
  for (int idx = tid; idx < TOTAL; idx += THREADS_PER_WG) {
    int lr = idx / MXP_K4, k4 = idx % MXP_K4;
    lds[k4 * LDS_STRIDE + lr] = in[in_e + (long long)(r0 + lr) * MXP_K4 + k4];
  }
  __syncthreads();
  // store: lds[k4][lr] -> out[k4*ROWS + (r0+lr)]   (per-warp 64 consecutive out-addrs -> coalesced write)
  #pragma unroll
  for (int idx = tid; idx < TOTAL; idx += THREADS_PER_WG) {
    int k4 = idx / TILE_ROWS, lr = idx % TILE_ROWS;
    out[out_e + (long long)k4 * MXP_ROWS + (r0 + lr)] = lds[k4 * LDS_STRIDE + lr];
  }
}
