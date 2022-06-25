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

#define BLOCK 8

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
  for (int by = 0; by < N; by += BLOCK) {
    for (int bx = 0; bx < N; bx += BLOCK) {

#ifndef FAST
      // compute
      float tc[BLOCK][BLOCK];
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          float acc = 0;
          for (int k = 0; k < N; k++) {
            acc += A[(by+y)*N + k] * B[(bx+x)*N + k];
          }
          tc[y][x] = acc;
        }
      }

      // store
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          C[(by+y)*N + bx+x] = tc[y][x];
        }
      }
#else
      float tc[BLOCK][BLOCK];
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          __m256 tmp = {};
          for (int k = 0; k < N; k += 8) {
            //printf("%d %d\n", ((by+y)*N + k)/8, ((bx+x)*N + k)/8);
            tmp = _mm256_fmadd_ps(
              Am[((by+y)*N + k)/8],
              Bm[((bx+x)*N + k)/8],
              tmp);
          }
          float ftmp = 0.0;
          for (int i = 0; i < 8; i++) ftmp += tmp[i];
          tc[y][x] = ftmp;
        }
      }

      // store
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          C[(by+y)*N + bx+x] = tc[y][x];
        }
      }
#endif

    }
  }
}

int main() {
  printf("hello\n");
  assert(N%BLOCK == 0);

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

  //matmul();
  uint64_t start = nanos();
  matmul();
  uint64_t end = nanos();
  double gflop = (2.0*N*N*N)*1e-9;
  double s = (end-start)*1e-9;
  printf("%f GFLOP/S\n", gflop/s);

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