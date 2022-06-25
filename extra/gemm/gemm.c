// clang -DFAST -ffast-math -march=native -O2 gemm.c && ./a.out

// https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

//#define DEBUG

#ifdef DEBUG
  #define N 8
#else
  //#define N 4096
  // L1 cache is 32 kB
  #define N 768
  // 8*768*4 = 24 kB
#endif

#define BLOCK_Y 1
#define BLOCK_X 1

// aligned?
float A[N*N] __attribute__ ((aligned (32)));
float B[N*N] __attribute__ ((aligned (32)));
float C[N*N] __attribute__ ((aligned (32)));
float val[N*N] __attribute__ ((aligned (32)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

float Bf[N*N] __attribute__ ((aligned (32)));
__m256 *Bfm = (__m256*)Bf;

#define BLOCK 8
void matmul() {
  // 136.77 GFLOPS on single core numpy
  // 4.59 GHz
  // 32 FLOPS/cycle (16 FMAs, aka 2x 8/32B wide FMAs)

  // A = (x, k)
  // B = (y, k)

  // Af = (x/8, k, 8)
  // Bf = (y/8, k, 8)

  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x += BLOCK) {

      __m256 acc = {};
      for (int k = 0; k < N; k++) {
        __m256 ta = _mm256_broadcast_ss(&A[y*N + k]);
        acc = _mm256_fmadd_ps(ta, Bfm[(x*N)/8 + k], acc);
      }
      Cm[(y*N + x)/8] = acc;


      /*float acc[BLOCK] = {};
      for (int k = 0; k < N; k++) {
        float ta = A[y*N + k];
        for (int ix = 0; ix < BLOCK; ix++) {
          //acc[ix] += ta * B[(x+ix)*N + k];
          acc[ix] += ta * Bf[x*N + k*8 + ix];
        }
      }

      // writeback
      for (int ix = 0; ix < BLOCK; ix++) {
        C[y*N + x + ix] = acc[ix];
      }*/

      /*__m256 acc = {};
      for (int k = 0; k < N; k++) {
        float ta = A[y*N + k];*/
    }
  }

  // 768*768*4 = 2.4 MB
  /*for (int y = 0; y < N; y++) {
    for (int bx = 0; bx < N; bx += BLOCK_X) {
      // this should all be in L1 cache

      // 16 YMM registers
      __m256 tc[1] = {};
      for (int k = 0; k < N; k += 8) {
        // this should all be in registers

        //__m256 ty = {2.0, 4.0, 8.0, 16.0, 1.0, 1.0, 1.0, 1.0};
        //__m256 ty = Am[(y*N + k)/8];
        __m256 tx = Bm[(bx*N + k)/8];

        for (int ik = 0; ik < 8; ik++) {
          //printf("%d %d\n", ((by+y)*N + k)/8, ((bx+x)*N + k)/8);
          //tc[x] = _mm256_fmadd_ps(ty, Bm[(x*N + k)/8], tc[x]);
          //tc[x] = _mm256_fmadd_ps(ty, Bm[((bx+x)*N + k)/8], tc[x]);
          __m256 ty = _mm256_broadcast_ss(&A[y*N + k + ik]);
          tc[0] = _mm256_fmadd_ps(ty, tx, tc[0]);
          //__builtin_prefetch(&Bm[((bx+x)*N + k)/8 + 1]);
          //tc[x] = _mm256_fmadd_ps(ty, Bl[(x*N + k)/8], tc[x]);
          //tc[x] = _mm256_fmadd_ps(ty, Bl[x+k], tc[x]);
          //tc[x] = _mm256_fmadd_ps(ty, ty, tc[x]);
        }

      }

      Cm[(y*N + bx)/8] = tc[0];
     
    }
  }*/
}

int main() {
  printf("hello\n");
  assert(N%BLOCK_X == 0);

#ifdef DEBUG
  for (int i = 0; i < N*N; i++) A[i] = i;
  for (int i = 0; i < N*N; i++) B[i] = i;
#else
  FILE *f = fopen("/tmp/matmul", "rb");
  fread(A, 1, sizeof(float)*N*N, f);
  fread(B, 1, sizeof(float)*N*N, f);
  fread(val, 1, sizeof(float)*N*N, f);
  fclose(f);
#endif

  // preswizzle
  for (int y = 0; y < N; y+=8) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 8; iy++) {
        Bf[y*N + x*8 + iy] = B[(y+iy)*N + x];
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    uint64_t start = nanos();
    matmul();
    uint64_t end = nanos();
    double gflop = (2.0*N*N*N)*1e-9;
    double s = (end-start)*1e-9;
    printf("%f GFLOP/S\n", gflop/s);
  }

#ifdef DEBUG
  for (int i = 0; i < N*N; i++) {
    if (i%N == 0 && i != 0) printf("\n");
    printf("%f ", C[i]);
  }
  printf("\n");
#else
  for (int k = 0; k < N*N; k++) {
    if (fabsf(C[k] - val[k]) > 1e-3) {
      printf("MISMATCH AT %d, %f != %f\n", k, C[k], val[k]);
      return -1;
    }
  }
  printf("match\n");
#endif

  return 0;
}