#include <hip/hip_runtime.h>
#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef S_DIM
#define S_DIM 65536
#endif
#ifndef E_DIM
#define E_DIM 32
#endif
extern "C" __global__ void route_rank_counts(int* ranks, int* counts, const int* topi) {
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i >= (long long)G_DIM*S_DIM) return;
  int g=i/S_DIM, e=topi[i];
  ranks[i]=__hip_atomic_fetch_add(&counts[g*E_DIM+e],1,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
}
