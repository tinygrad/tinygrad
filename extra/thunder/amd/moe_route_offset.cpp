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
extern "C" __global__ void route_add_offsets(int* dest, const int* ranks, const int* topi, const int* off) {
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i >= (long long)G_DIM*S_DIM) return;
  int g=i/S_DIM, e=topi[i]; dest[i]=ranks[i]+off[g*(E_DIM+1)+e];
}
