// clang -ffast-math -march=native -O3 gemm.c && ./a.out
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>

//#define DEBUG

#ifdef DEBUG
  #define N 8
#else
  //#define N 4096
  #define N 2048
#endif

#define BLOCK_Y 4
#define BLOCK_X 2

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

//#define FAST

void matmul() {
  // 136.77 GFLOPS on single core numpy
  // 4.59 GHz
  // 32 FLOPS/cycle (16 FMAs, aka 2x 8/32B wide FMAs)

  for (int by = 0; by < N; by += BLOCK_Y) {
    for (int bx = 0; bx < N; bx += BLOCK_X) {

#ifndef FAST
      float tc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k++) {
        for (int y = 0; y < BLOCK_Y; y++) {
          for (int x = 0; x < BLOCK_X; x++) {
            tc[y][x] += A[(by+y)*N + k] * B[(bx+x)*N + k];
          }
        }
      }

      // store
      for (int y = 0; y < BLOCK_Y; y++) {
        for (int x = 0; x < BLOCK_X; x++) {
          C[(by+y)*N + bx+x] = tc[y][x];
        }
      }
#else
      // 16 YMM registers
      __m256 tc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k += 8) {
        for (int y = 0; y < BLOCK_Y; y++) {
          for (int x = 0; x < BLOCK_X; x++) {
            //printf("%d %d\n", ((by+y)*N + k)/8, ((bx+x)*N + k)/8);
            tc[y][x] = _mm256_fmadd_ps(
              Am[((by+y)*N + k)/8],
              Bm[((bx+x)*N + k)/8],
              tc[y][x]);
          }
        }
      }

      // store
      for (int y = 0; y < BLOCK_Y; y++) {
        for (int x = 0; x < BLOCK_X; x++) {
          float ftmp = 0.0;
          for (int i = 0; i < 8; i++) ftmp += tc[y][x][i];
          C[(by+y)*N + bx+x] = ftmp;
        }
      }
#endif

    }
  }
}

int main() {
  printf("hello\n");
  assert(N%BLOCK_Y == 0);
  assert(N%BLOCK_X == 0);

/*#ifdef FAST
  assert(BLOCK_X == 8);
#endif*/

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

  for (int i = 0; i < 2; i++) {
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